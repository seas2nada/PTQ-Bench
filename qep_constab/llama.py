import copy
import math
import gc
from contextlib import contextmanager

import torch
import torch.nn as nn

from modelutils import *
from quant import *

from gptq import *
from modelutils import *
from quant import *
from resultutils import *
from awq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_tensor
from awq.quantize.pre_quant import run_awq
from qep_awq import run_awq_with_QEP
from zeroShot.utils import *
from zeroShot.main import get_result

# 너가 올린 Helper(SpQR/quip 버전)의 Helper 클래스를 사용한다고 가정
# from helper_file import Helper  # <-- 필요 시 실제 파일에서 import
# 여기서는 llama.py 내부에 Helper가 이미 있다고 가정(네 코드처럼)

# -----------------------------
# Stability utilities
# -----------------------------

@contextmanager
def sdp_math_only():
    """
    torch>=2.0: FlashAttention backward 미구현 이슈 회피용.
    Jacobian/grad 계산 시 math kernel만 사용.
    """
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True
        ):
            yield
    except Exception:
        yield


def _block_forward_last_token(layer, h, layer_kwargs):
    out = layer(h, **layer_kwargs)[0]
    return out[:, -1, :]  # (1, d)


def estimate_block_jacobian_norm(layer, h, layer_kwargs, n_power=1):
    """
    Power iteration으로 ||J||2 근사.
    J = d( block_out_last_token ) / d( block_in )
    """
    layer.eval()
    with torch.enable_grad():
        x = h.detach().clone().requires_grad_(True)

        def f(inp):
            with sdp_math_only():
                return _block_forward_last_token(layer, inp, layer_kwargs)  # (1, d)

        v = torch.randn_like(x)
        v = v / (v.norm() + 1e-8)

        sigma = None
        for _ in range(max(1, int(n_power))):
            with sdp_math_only():
                y, jv = torch.autograd.functional.jvp(f, (x,), (v,), create_graph=True)

            u = jv
            u_norm = u.norm() + 1e-8
            u = u / u_norm

            with sdp_math_only():
                (jt_u,) = torch.autograd.grad(
                    y, x, grad_outputs=u, retain_graph=True, create_graph=False
                )

            v = jt_u
            v = v / (v.norm() + 1e-8)
            sigma = u_norm

        return float(sigma) if sigma is not None else 0.0


def estimate_mlp_junction_rms(layer, h, layer_kwargs):
    """
    LLaMA MLP junction: z = act(gate(x)) * up(x)
    z RMS가 폭주하면 down_proj에서 NaN/Inf가 터지기 쉬움.
    """
    mlp = layer.mlp
    x = h  # (1, seqlen, d)

    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    z = mlp.act_fn(gate) * up

    rms = torch.sqrt(torch.mean(z.float() ** 2, dim=-1) + 1e-8)  # (1, seqlen)
    return float(rms.max())


def is_layer_finite(layer, inps, layer_kwargs, dev, nprobe=2):
    layer.eval()
    with torch.no_grad():
        n = min(int(nprobe), inps.shape[0])
        for j in range(n):
            h0 = inps[j].unsqueeze(0).to(dev)
            y = layer(h0, **layer_kwargs)[0]
            if not torch.isfinite(y).all():
                return False
    return True


# -----------------------------
# Model loader
# -----------------------------

def get_llama(model_name_or_path: str):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
    model.seqlen = 2048
    return model


# -----------------------------
# QEP + ConStab main
# -----------------------------

@torch.no_grad()
def llama_sequential_qep_constab(model, dataloader, dev, args):
    """
    QEP + ConStab wrapper (for GPTQ / QUIP / RTN path in Helper pipeline).

    - For each block i:
      - restore FP block
      - run one-pass (collect stats w/ add_batch_qep or add_batch)
      - apply QEP correction (run_weight_correct)
      - apply quantization (run_gptq / run_quip / rtn)
      - check stability (finite + junction + optional sigma)
      - if violated: increase damping, decrease perccorr, retry
    """
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype, device=dev)
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"].update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # move out
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    layer_kwargs = cache["layer_kwargs"]
    inps_true = inps.clone()

    sequential = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

    from tqdm import tqdm
    for i in tqdm(range(len(layers)), desc=f"Running {args.method}+QEP+ConStab"):
        # FP snapshot for rollback
        layer_fp_cpu = copy.deepcopy(layers[i]).to("cpu")

        # ConStab knobs (adaptive during retries)
        percdamp_now = float(args.percdamp)
        percdampqep_now = float(args.percdampqep)
        perccorr_now = float(args.perccorr)

        # FP junction reference (few probes)
        jac_n = min(args.stab_jac_nsamples, args.nsamples)
        fp_junc_ref = []
        layer_fp_dev = copy.deepcopy(layers[i]).to(dev).eval()
        for j in range(jac_n):
            h0 = inps[j].unsqueeze(0).to(dev)
            fp_junc_ref.append(estimate_mlp_junction_rms(layer_fp_dev, h0, layer_kwargs))
        fp_junc_ref_max = max(fp_junc_ref) if len(fp_junc_ref) else 0.0
        layer_fp_dev = layer_fp_dev.to("cpu")
        del layer_fp_dev
        torch.cuda.empty_cache()

        # A helper that runs "one pass" quantization for block i with current knobs
        def run_one_pass(layer_q, layer_true, percdamp, percdampqep, perccorr):
            """
            Mutates layer_q weights in-place.
            Uses Helper stats collection and quant routines.
            """
            full = find_layers(layer_q)
            full_true = find_layers(layer_true)

            # sequential sub-block processing
            for names in sequential:
                subset = {n: full[n] for n in names}
                subset_true = {n: full_true[n] for n in names}

                hook_inp = {}
                hook_inp_true = {}

                def make_hook(dst, name):
                    def hook(module, inp, out):
                        dst[name] = inp[0].detach().clone()
                    return hook

                handles = []
                for name, module in subset.items():
                    handles.append(module.register_forward_hook(make_hook(hook_inp, name)))
                for name, module in subset_true.items():
                    handles.append(module.register_forward_hook(make_hook(hook_inp_true, name)))

                helper = Helper(subset[names[0]])

                # stats collection
                if args.method in ["gptq", "quip"] or names[0] != "mlp.down_proj":
                    for j in range(args.nsamples):
                        _ = layer_q(inps[j].unsqueeze(0), **layer_kwargs)

                        # QEP stats for all except down_proj
                        if (not args.qep) or (names[0] == "mlp.down_proj"):
                            helper.add_batch(hook_inp[names[0]])
                        else:
                            _ = layer_true(inps_true[j].unsqueeze(0), **layer_kwargs)
                            helper.add_batch_qep(hook_inp[names[0]], hook_inp_true[names[0]])

                for h in handles:
                    h.remove()

                # quantization (with QEP correction first if enabled)
                for name, module in subset.items():
                    # QEP correction: skip down_proj by design
                    if args.qep and name != "mlp.down_proj":
                        helper.run_weight_correct(
                            module, percdamp=percdampqep, perccorr=perccorr
                        )

                    # quantizer backend
                    if args.method == "rtn":
                        module.weight.data = pseudo_quantize_tensor(
                            module.weight.data, n_bit=args.wbits, q_group_size=args.groupsize
                        )
                    elif args.method == "gptq":
                        # optional routing: if ConStab decides to raise down_proj bits
                        wbits_local = args.wbits
                        if getattr(args, "stab_route_active", False) and name == "mlp.down_proj":
                            wbits_local = max(wbits_local, args.stab_route_downproj_bit)

                        helper.run_gptq(
                            module,
                            percdamp=percdamp,
                            wbits=wbits_local,
                            groupsize=args.groupsize,
                            actorder=args.act_order
                        )
                    elif args.method == "quip":
                        helper.run_quip(
                            module, percdamp=percdamp, wbits=args.wbits, multigpu=args.multigpu
                        )
                    else:
                        raise NotImplementedError(f"Unsupported method: {args.method}")

                helper.free()

        # refinement loop
        ok = False
        for t in range(max(1, args.stab_iters)):
            # restore FP
            layer_q = copy.deepcopy(layer_fp_cpu).to(dev)
            layer_true = copy.deepcopy(layer_fp_cpu).to(dev)

            # (important) reset routing flag each trial (we may turn on later)
            args.stab_route_active = False

            # run QEP + quant
            run_one_pass(layer_q, layer_true, percdamp_now, percdampqep_now, perccorr_now)

            # check stability on the final block
            finite_ok = is_layer_finite(layer_q, inps, layer_kwargs, dev, nprobe=args.stab_jac_nsamples)

            # junction check (key for 2-bit)
            junc_ok = True
            if fp_junc_ref_max > 0:
                junc_vals = []
                for j in range(jac_n):
                    h0 = inps[j].unsqueeze(0).to(dev)
                    junc_vals.append(estimate_mlp_junction_rms(layer_q, h0, layer_kwargs))
                q_junc = max(junc_vals) if len(junc_vals) else 0.0
                if q_junc > args.stab_junc_mult * fp_junc_ref_max:
                    junc_ok = False

            # sigma check (optional)
            sigma_ok = True
            if args.stab_use_sigma:
                sigmas = []
                for j in range(jac_n):
                    h0 = inps[j].unsqueeze(0).to(dev)
                    sigmas.append(estimate_block_jacobian_norm(
                        layer_q, h0, layer_kwargs, n_power=args.stab_jac_power
                    ))
                sigma_hat = max(sigmas) if len(sigmas) else 0.0
                sigma_ok = (sigma_hat <= args.stab_tau)

            if finite_ok and junc_ok and sigma_ok:
                ok = True
                break

            # ---- refine policy (the "ConStab" part) ----
            # non-finite: strongest reaction
            if not finite_ok:
                percdamp_now = min(percdamp_now * max(2.0, args.stab_damp_mult), args.stab_max_percdamp)
                percdampqep_now = min(percdampqep_now * args.stab_damp_mult, args.stab_max_percdampqep)
                perccorr_now = max(perccorr_now * args.stab_corr_mult, args.stab_min_perccorr)
                continue

            # junction overshoot: usually QEP correction too aggressive / low-bit MLP sensitive
            if not junc_ok:
                # first: reduce correction strength
                perccorr_now = max(perccorr_now * args.stab_corr_mult, args.stab_min_perccorr)
                # also modestly increase damping
                percdamp_now = min(percdamp_now * args.stab_damp_mult, args.stab_max_percdamp)
                percdampqep_now = min(percdampqep_now * args.stab_damp_mult, args.stab_max_percdampqep)

                # optional routing after a few tries: only down_proj bits up
                if (t + 1) >= args.stab_route_after and args.stab_route_downproj_bit > 0:
                    args.stab_route_active = True
                continue

            # sigma violation only
            if not sigma_ok:
                percdamp_now = min(percdamp_now * args.stab_damp_mult, args.stab_max_percdamp)
                # sigma 문제는 correction보다 quant 안정성이 핵심이라 corr은 그대로 두는 편이 낫다
                continue

        if not ok:
            print(f"[QEP-ConStab] layer {i}: constraint not satisfied after {args.stab_iters} tries. "
                  f"Using last trial knobs: percdamp={percdamp_now}, percdampqep={percdampqep_now}, perccorr={perccorr_now}")

        # commit: layer_q is the accepted block on dev
        # update activations for next block
        for j in range(args.nsamples):
            inps[j] = layer_q(inps[j].unsqueeze(0), **layer_kwargs)[0]
            inps_true[j] = layer_true(inps_true[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer_q.cpu()
        del layer_q, layer_true
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache


# -----------------------------
# CLI / main
# -----------------------------

if __name__ == "__main__":
    import argparse
    from datautils import *
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("method", type=str, choices=["rtn", "gptq", "quip"])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)

    parser.add_argument("--wbits", type=int, default=4, choices=[2, 3, 4, 8, 16])
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument("--multigpu", action="store_true")

    # QEP
    parser.add_argument("--qep", action="store_true")
    parser.add_argument("--percdampqep", type=float, default=1.0)
    parser.add_argument("--perccorr", type=float, default=0.5)

    # base GPTQ damping
    parser.add_argument("--percdamp", type=float, default=0.01)

    # ConStab options
    parser.add_argument("--stab-iters", type=int, default=3)
    parser.add_argument("--stab-damp-mult", type=float, default=2.0)
    parser.add_argument("--stab-max-percdamp", type=float, default=0.2)
    parser.add_argument("--stab-max-percdampqep", type=float, default=10.0)

    parser.add_argument("--stab-use-sigma", type=int, default=0)
    parser.add_argument("--stab-tau", type=float, default=1.05)
    parser.add_argument("--stab-jac-nsamples", type=int, default=4)
    parser.add_argument("--stab-jac-power", type=int, default=1)

    parser.add_argument("--stab-junc-mult", type=float, default=3.0)

    # if unstable, reduce correction
    parser.add_argument("--stab-corr-mult", type=float, default=0.7,
                        help="Multiply perccorr by this when stability violated (should be < 1).")
    parser.add_argument("--stab-min-perccorr", type=float, default=0.05)

    # optional routing: after N failures, quantize down_proj with higher bit
    parser.add_argument("--stab-route-after", type=int, default=2)
    parser.add_argument("--stab-route-downproj-bit", type=int, default=3)

    parser.add_argument(
        '--save-model', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )

    args = parser.parse_args()
    args.sym = False  # Helper.run_gptq uses sym=False in your earlier code; keep consistent.

    # load
    model = get_llama(args.model)
    model.eval()

    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    llama_sequential_qep_constab(model, dataloader, DEV, args)

    if args.save_model:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained(args.model)
        enc.save_pretrained(args.save_model)
        model.save_pretrained(args.save_model)