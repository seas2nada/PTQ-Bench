import copy
from contextlib import contextmanager
import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *


# ---------------------------
# Utils for Jacobian metric
# ---------------------------

@contextmanager
def sdp_math_only():
    # torch>=2.0: avoid flash-attn backward not implemented inside autograd
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True
        ):
            yield
    except Exception:
        yield


def estimate_kstep_diag_jtj(
    layers, i, x, attention_mask, position_ids,
    dev, kstep=2, nprobe=2
):
    """
    Hutchinson estimator for diag(J^T J) where
      f(x) = last_token_hidden after applying block i then next kstep blocks (FP copies).
    Returns g: (hidden_size,) float32 on dev.
    """
    L = len(layers)
    kstep = int(max(0, kstep))
    kmax = min(kstep, (L - 1) - i)
    if kmax <= 0:
        d = x.shape[-1]
        return torch.ones((d,), device=dev, dtype=torch.float32)

    # FP copies (do not touch quantized in-place weights)
    blocks = []
    for t in range(i, i + kmax + 1):
        blocks.append(copy.deepcopy(layers[t]).to(dev).eval())

    def f(inp):
        h = inp
        with sdp_math_only():
            for b in blocks:
                h = b(h, attention_mask=attention_mask, position_ids=position_ids)[0]
        return h[:, -1, :]  # (1, d)

    d = x.shape[-1]
    g = torch.zeros((d,), device=dev, dtype=torch.float32)

    with torch.enable_grad():
        xg = x.detach().clone().requires_grad_(True)
        y = f(xg)  # (1, d)

        for _ in range(max(1, int(nprobe))):
            r = torch.randn_like(y)  # (1, d)
            s = (y * r).sum()
            (jt_r,) = torch.autograd.grad(
                s, xg, retain_graph=True, create_graph=False
            )  # (1, seqlen, d)

            g += torch.mean(jt_r.float() ** 2, dim=(0, 1))

    g = g / max(1, int(nprobe))
    g = g / (g.mean() + 1e-8)
    g = torch.clamp(g, 1e-4, 1e4)

    # cleanup
    for b in blocks:
        b.to("cpu")
    del blocks
    torch.cuda.empty_cache()

    return g


# ---------------------------
# Baseline loader
# ---------------------------

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

        # ---- NEW: propagation metric g_hidden for this block ----
        g_hidden = None
        if getattr(args, "prop_kstep", 0) > 0:
            nprobe = min(int(getattr(args, "prop_nprobe", 2)), args.nsamples)
            hutch = int(getattr(args, "prop_hutch", 2))
            gs = []
            for j in range(nprobe):
                x = inps[j].unsqueeze(0).to(dev)
                g = estimate_kstep_diag_jtj(
                    layers, i, x, attention_mask, position_ids,
                    dev, kstep=int(args.prop_kstep), nprobe=hutch
                )
                gs.append(g)
            # conservative: max across probes
            g_hidden = torch.stack(gs, dim=0).max(dim=0).values

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

                # ---- NEW: inject propagation metric into Hessian building ----
                # Most projections have input dim == hidden_size (matches g_hidden).
                # down_proj input dim == ffn_dim, so skip by default.
                if g_hidden is not None and name != 'mlp.down_proj':
                    gptq[name].set_col_metric(g_hidden)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]

            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups
                )
                quantizers[f'model.layers.{i}.{name}'] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
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
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

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
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
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
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of huggingface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--nearest', action='store_true')

    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16])
    parser.add_argument('--groupsize', type=int, default=-1)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--new-eval', action='store_true')
    parser.add_argument('--act-order', action='store_true')
    parser.add_argument('--true-sequential', action='store_true')
    parser.add_argument('--static-groups', action='store_true')
    parser.add_argument('--ckpt', type=str, help='Whether to save quantized model')

    # ---- NEW: propagation-aware objective options ----
    parser.add_argument('--prop-kstep', type=int, default=0,
                        help='Downstream steps for diag(J^T J) metric. 0 disables.')
    parser.add_argument('--prop-nprobe', type=int, default=2,
                        help='How many calibration samples to probe g per block.')
    parser.add_argument('--prop-hutch', type=int, default=2,
                        help='Hutchinson random probes per sample.')

    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    if args.ckpt:
        model.save_pretrained(args.ckpt)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.ckpt)

    if args.ckpt:
        torch.save(model.state_dict(), args.ckpt)
