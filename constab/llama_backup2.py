import copy
from contextlib import nullcontext, contextmanager
import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

def is_layer_finite(layer, inps, attention_mask, position_ids, dev, nprobe=2):
    layer.eval()
    with torch.no_grad():
        n = min(nprobe, inps.shape[0])
        for j in range(n):
            h0 = inps[j].unsqueeze(0).to(dev)
            out = layer(h0, attention_mask=attention_mask, position_ids=position_ids)[0]
            if not torch.isfinite(out).all():
                return False
    return True

def estimate_block_gain(layer_fp, layer_q, inps, attention_mask, position_ids, dev, nprobe=8):
    # returns alpha such that alpha * y_q matches y_fp in RMS
    layer_fp = layer_fp.to(dev)
    layer_q = layer_q.to(dev)
    with torch.no_grad():
        a_num, a_den = 0.0, 0.0
        for j in range(min(nprobe, inps.shape[0])):
            h0 = inps[j].unsqueeze(0).to(dev)
            y_fp = layer_fp(h0, attention_mask=attention_mask, position_ids=position_ids)[0]
            y_q  = layer_q (h0, attention_mask=attention_mask, position_ids=position_ids)[0]
            # RMS over (batch,seqlen,dim)
            num = torch.sqrt(torch.mean(y_fp.float()**2) + 1e-8)
            den = torch.sqrt(torch.mean(y_q .float()**2) + 1e-8)
            a_num += num.item()
            a_den += den.item()
        alpha = (a_num / max(1, min(nprobe, inps.shape[0]))) / max(1e-8, (a_den / max(1, min(nprobe, inps.shape[0]))))
        alpha = float(max(0.25, min(4.0, alpha)))  # clamp
    layer_fp = layer_fp.to("cpu")
    return alpha

def estimate_mlp_junction_rms(layer, h, attention_mask, position_ids):
    """
    LlamaDecoderLayer 내부 MLP의 junction z = act(gate(x)) * up(x) 의 RMS를 측정.
    폭주하면 down_proj에서 NaN이 터지는 전형적 원인.
    """
    mlp = layer.mlp
    x = h
    # forward는 no_grad에서 호출될 거라 괜찮음
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    z = mlp.act_fn(gate) * up
    # token-wise RMS를 평균내서 스칼라로
    rms = torch.sqrt(torch.mean(z.float() ** 2, dim=-1) + 1e-8)  # (1, seqlen)
    return float(rms.max())  # worst-token RMS


@contextmanager
def sdp_math_only():
    # torch>=2.0
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True
        ):
            yield
    except Exception:
        # fallback: do nothing if backend not available
        yield

def _block_forward_last_token(layer, h, attention_mask, position_ids):
    # h: (1, seqlen, d)
    out = layer(h, attention_mask=attention_mask, position_ids=position_ids)[0]
    return out[:, -1, :]  # (1, d)

def estimate_block_jacobian_norm(layer, h, attention_mask, position_ids, n_power=1):
    layer.eval()
    with torch.enable_grad():
        x = h.detach().clone().requires_grad_(True)

        def f(inp):
            # make sure attention runs in math kernel (no flash) inside JVP/VJP
            with sdp_math_only():
                return _block_forward_last_token(layer, inp, attention_mask, position_ids)  # (1, d)

        v = torch.randn_like(x)
        v = v / (v.norm() + 1e-8)

        sigma = None
        for _ in range(max(1, n_power)):
            # Jv
            with sdp_math_only():
                y, jv = torch.autograd.functional.jvp(f, (x,), (v,), create_graph=True)
            u = jv
            u_norm = u.norm() + 1e-8
            u = u / u_norm

            # J^T u
            with sdp_math_only():
                (jt_u,) = torch.autograd.grad(
                    y, x, grad_outputs=u, retain_graph=True, create_graph=False
                )

            v = jt_u
            v = v / (v.norm() + 1e-8)
            sigma = u_norm

        return float(sigma)

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    from tqdm import tqdm
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        layer_fp = copy.deepcopy(layers[i]).to('cpu')  # FP copy
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        def run_one_pass(percdamp_now: float):
            full_local = find_layers(layer)
            if args.true_sequential:
                sequential_local = [
                    ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                    ['self_attn.o_proj'],
                    ['mlp.up_proj', 'mlp.gate_proj'],
                    ['mlp.down_proj']
                ]
            else:
                sequential_local = [list(full_local.keys())]

            for names in sequential_local:
                subset = {n: full_local[n] for n in names}

                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)
                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))

                for j in range(args.nsamples):
                    _ = layer(inps[j].unsqueeze(0),
                              attention_mask=attention_mask,
                              position_ids=position_ids)[0]

                for h in handles:
                    h.remove()

                for name in subset:
                    gptq[name].fasterquant(
                        percdamp=percdamp_now,
                        groupsize=args.groupsize,
                        actorder=args.act_order,
                        static_groups=args.static_groups
                    )
                    quantizers[f'model.layers.{i}.{name}'] = gptq[name].quantizer
                    gptq[name].free()

        def requantize_downproj_only(percdamp_now: float, wbits_down: int):
            full_local = find_layers(layer)
            name = 'mlp.down_proj'
            if name not in full_local:
                return
            mod = full_local[name]

            g = GPTQ(mod)
            g.quantizer = Quantizer()
            g.quantizer.configure(wbits_down, perchannel=True, sym=args.sym, mse=False)

            def add_batch(_, inp, out):
                g.add_batch(inp[0].data, out.data)

            h = mod.register_forward_hook(add_batch)
            for j in range(args.nsamples):
                _ = layer(inps[j].unsqueeze(0),
                          attention_mask=attention_mask,
                          position_ids=position_ids)[0]
            h.remove()

            g.fasterquant(
                percdamp=percdamp_now,
                groupsize=args.groupsize,
                actorder=args.act_order,
                static_groups=args.static_groups
            )
            quantizers[f'model.layers.{i}.mlp.down_proj'] = g.quantizer
            g.free()

        # ---- FP junction baseline ----
        fp_junc = None
        if args.stab_junc_mult > 0:
            layer_fp_dev = copy.deepcopy(layers[i]).to(dev)
            layer_fp_dev.eval()
            jn = min(args.stab_jac_nsamples, args.nsamples)
            vals = []
            for j in range(jn):
                h0 = inps[j].unsqueeze(0).to(dev)
                vals.append(estimate_mlp_junction_rms(layer_fp_dev, h0, attention_mask, position_ids))
            fp_junc = max(vals) if vals else None
            del layer_fp_dev
            torch.cuda.empty_cache()

        # ---- NFSR baseline ----
        percdamp_now = float(args.percdamp)
        layer.load_state_dict(layer_fp.state_dict())
        torch.cuda.empty_cache()
        run_one_pass(percdamp_now)

        baseline_sd = {k: v.detach().cpu().clone() for k, v in layer.state_dict().items()}
        if not is_layer_finite(layer, inps, attention_mask, position_ids, dev, nprobe=args.stab_jac_nsamples):
            raise RuntimeError(f"[NFSR] baseline GPTQ already non-finite at layer {i}")

        down_proj_name = 'mlp.down_proj'
        constab_layers = [name for name in sequential if name != down_proj_name]
        # ---- refinement ----
        for t in range(max(1, args.stab_iters)):
            layer.load_state_dict(layer_fp.state_dict())
            torch.cuda.empty_cache()

            percdamp_try = min(percdamp_now * (args.stab_damp_mult ** (t + 1)),
                               args.stab_max_percdamp)
            run_one_pass(percdamp_try)

            # non-finite => rollback
            if not is_layer_finite(layer, inps, attention_mask, position_ids, dev,
                                   nprobe=args.stab_jac_nsamples):
                layer.load_state_dict(baseline_sd, strict=True)
                break

            # # junction overshoot => down_proj routing
            # if fp_junc is not None and args.stab_junc_mult > 0:
            #     jn = min(args.stab_jac_nsamples, args.nsamples)
            #     q_vals = []
            #     for j in range(jn):
            #         h0 = inps[j].unsqueeze(0).to(dev)
            #         q_vals.append(estimate_mlp_junction_rms(layer, h0, attention_mask, position_ids))
            #     q_junc = max(q_vals) if q_vals else 0.0

            #     if q_junc > args.stab_junc_mult * fp_junc:
            #         if args.stab_route_downproj_bit > 0:
            #             wbits_down = max(args.wbits, 0)

            #             if not is_layer_finite(layer, inps, attention_mask, position_ids, dev,
            #                                    nprobe=args.stab_jac_nsamples):
            #                 layer.load_state_dict(baseline_sd, strict=True)
            #                 break

            #             percdamp_now = percdamp_try
            #             break
            #         else:
            #             continue

            # sigma check (optional, soft)
            if args.stab_use_sigma:
                sigmas = []
                jac_n = min(args.stab_jac_nsamples, args.nsamples)
                for j in range(jac_n):
                    h0 = inps[j].unsqueeze(0).to(dev)
                    sigma = estimate_block_jacobian_norm(
                        layer, h0, attention_mask, position_ids,
                        n_power=args.stab_jac_power
                    )
                    sigmas.append(sigma)
                sigma_hat = max(sigmas) if sigmas else 0.0
                if sigma_hat <= args.stab_tau:
                    percdamp_now = percdamp_try
                    break
            else:
                percdamp_now = percdamp_try
                break

        # ---- gain calibration ----
        if args.stab_gain_calib:
            alpha = estimate_block_gain(
                layer_fp, layer, inps, attention_mask, position_ids,
                dev, nprobe=args.stab_gain_nsamples
            )
            layer.mlp.down_proj.weight.data.mul_(alpha)
            layer.self_attn.o_proj.weight.data.mul_(alpha)

        # update outs
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--ckpt', type=str,
        help='Whether to save quantized model'
    )
    # --- ConStab-GPTQ options ---
    parser.add_argument(
        '--stab-tau', type=float, default=1.05,
        help='Target Jacobian spectral norm upper bound (tau). Smaller => more stable, potentially more error.'
    )
    parser.add_argument(
        '--stab-iters', type=int, default=3,
        help='Max refinement iterations per block (re-quantize with higher damping if unstable).'
    )
    parser.add_argument(
        '--stab-damp-mult', type=float, default=2.0,
        help='Multiplier to increase percdamp when stability constraint is violated.'
    )
    parser.add_argument(
        '--stab-max-percdamp', type=float, default=0.2,
        help='Upper bound for percdamp during refinement.'
    )
    parser.add_argument(
        '--stab-jac-nsamples', type=int, default=4,
        help='How many calibration hidden states to probe Jacobian norm with.'
    )
    parser.add_argument(
        '--stab-jac-power', type=int, default=1,
        help='Power-iteration steps for Jacobian spectral norm estimate (1~2 recommended).'
    )
    parser.add_argument(
        '--stab-scope', type=str, default='block',
        choices=['block', 'o_down'],
        help="Where to enforce stability. 'block': whole block output(last token). 'o_down': only check after quantizing o_proj+down_proj (optional extension)."
    )
    parser.add_argument('--stab_use_sigma', type=int, default=1,
                        help='1이면 Jacobian sigma constraint도 사용, 0이면 junction+finite만 사용.')
    parser.add_argument('--stab_junc_mult', type=float, default=3.0,
                        help='MLP junction RMS allowed multiplier over FP reference.')
    parser.add_argument('--stab_route_after', type=int, default=2,
                        help='After this many failed refinements, enable stability routing for down_proj.')
    parser.add_argument('--stab_route_downproj_bit', type=int, default=3,
                        help='If >0, quantize mlp.down_proj with at least this many bits when routing is enabled (e.g., 3 or 4).')
    parser.add_argument('--stab-gain-calib', action='store_true',
                    help='Calibrate block output scale to reduce PPL degradation.')
    parser.add_argument('--stab-gain-nsamples', type=int, default=8,
                    help='How many calibration samples for gain estimation.')


    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)
    if args.ckpt:
        model.save_pretrained(args.ckpt)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.ckpt)
    torch.save(model.state_dict(), args.ckpt)
    datasets = ['wikitext2', 'ptb', 'c4'] 
    # if args.new_eval:
    #     datasets = ['wikitext2', 'ptb-new', 'c4-new']
    # for dataset in datasets:
    #     dataloader, testloader = get_loaders(
    #         dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    #     )
    #     print(dataset)
    #     llama_eval(model, testloader, DEV)
    # if args.save:
    #     llama_pack3(model, quantizers)
    #     torch.save(model.state_dict(), args.save)

