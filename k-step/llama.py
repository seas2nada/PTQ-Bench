import time

import torch
import torch.nn as nn
import copy
from datasets import load_dataset
from transformers import AutoTokenizer 
import tqdm
from accelerate import infer_auto_device_map, dispatch_model
import re

from gptq import *
from modelutils import *
from quant import *
from resultutils import *
from awq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_tensor
from awq.quantize.pre_quant import run_awq
from qep_awq import run_awq_with_QEP
from zeroShot.utils import *
from zeroShot.main import get_result

def headwise_expand(g: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    g: (d_model,)
    returns: (d_model,) where each head shares the same mean importance
    """
    d = g.numel()
    assert d % n_heads == 0, f"d_model={d} not divisible by n_heads={n_heads}"
    head_dim = d // n_heads
    gh = g.view(n_heads, head_dim).mean(dim=1, keepdim=True)  # (n_heads, 1)
    gh = gh.repeat(1, head_dim).reshape(-1)                   # (d_model,)
    return gh

def block_forward(block, x, layer_kwargs):
    # x: (seqlen, d) or (1, seqlen, d)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return block(x, **layer_kwargs)[0].squeeze(0)  # (seqlen, d)

# ------------------------------------------------------------
# (2) Δz 기반 k-step metric + (5) token-weight alpha_t 생성
# ------------------------------------------------------------
@torch.no_grad()
def estimate_kstep_metrics_delta(
    i,
    layers,
    layer,          # current block i (FP weights이지만 input은 inps로부터: prefix quant error 포함)
    layer_true,     # FP anchor block i (inps_true로부터)
    inps,           # current prefix states (nsamples, T, d)
    inps_true,      # FP prefix states (nsamples, T, d)
    layer_kwargs,
    K,
    lambdas,
    dev,
    eps: float = 1e-8,
):
    """
    Returns:
      g_block: (d_model,)  channel importance based on Δz_k = z_k(cur) - z_k(fp)
      token_alpha: (nsamples, T)  token-wise importance for weighted H (alpha_t)
    """

    K_eff = min(K, len(layers) - i - 1)
    if K_eff <= 0:
        return None, None

    ns = inps_true.shape[0]
    T = inps_true.shape[1]

    # d_model inferred from block output
    g_acc = None
    token_alpha = torch.zeros((ns, T), device=dev, dtype=torch.float32)

    for j in range(ns):
        # y_fp: FP prefix -> FP block i
        y_fp = block_forward(layer_true, inps_true[j], layer_kwargs)  # (T, d)
        # y_cur: current prefix -> current block i (prefix quant error reflected)
        y_cur = block_forward(layer, inps[j], layer_kwargs)           # (T, d)

        if g_acc is None:
            d = y_fp.shape[-1]
            g_acc = torch.zeros((d,), device=dev, dtype=torch.float32)

        z_fp = y_fp
        z_cur = y_cur

        for k in range(1, K_eff + 1):
            lam = float(lambdas[k - 1]) if lambdas is not None else 1.0

            # future blocks: 그대로 layers[i+k] 사용
            blk = layers[i + k].to(dev)

            z_fp = block_forward(blk, z_fp, layer_kwargs)   # (T, d)
            z_cur = block_forward(blk, z_cur, layer_kwargs) # (T, d)

            dz = (z_cur - z_fp).to(torch.float32)  # (T, d)

            # channel-wise importance: sum over tokens
            g_acc += lam * dz.pow(2).sum(dim=0)    # (d,)

            # token-wise importance: sum over channels
            token_alpha[j] += lam * dz.pow(2).sum(dim=1)  # (T,)

    # normalize: mean=1 (너무 공격적 스케일 방지)
    g_block = g_acc / (g_acc.mean() + eps)

    token_alpha = token_alpha / (token_alpha.mean() + eps)  # 전체 평균 1로
    return g_block, token_alpha

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
@torch.no_grad()
def llama_sequential(model, dev):
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

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
    cache = {'i': 0, 'layer_kwargs': {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'].update(kwargs)
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

    layer_kwargs = cache['layer_kwargs']
    inps_true = inps.clone()  # FP prefix state

    # lambdas
    if args.kstep > 0:
        if not hasattr(args, "kstep_lambdas") or args.kstep_lambdas is None:
            kstep_lambdas = [1.0] * args.kstep
        else:
            kstep_lambdas = args.kstep_lambdas
    else:
        kstep_lambdas = None

    sequential = [
        ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        ['self_attn.o_proj'],
        ['mlp.up_proj', 'mlp.gate_proj'],
        ['mlp.down_proj']
    ]
    module_order = [n for group in sequential for n in group]

    # (3) head info
    n_heads = getattr(model.config, "num_attention_heads", None)
    if n_heads is None:
        # LLaMA 계열이면 보통 num_attention_heads가 존재함
        raise ValueError("model.config.num_attention_heads not found; cannot do head-wise g for q/k/v")

    eps = 1e-8

    for i in tqdm.tqdm(range(len(layers)), desc=f'Running {args.method}'):
        # -----------------------------
        # ✅ anchor 순서 안정화: deepcopy 먼저
        # -----------------------------
        layer_cpu = layers[i]  # (대개 CPU)
        layer_true = copy.deepcopy(layer_cpu).to(dev)  # FP anchor
        layer = layer_cpu.to(dev)                      # quantize target (in-place)

        full = find_layers(layer)
        full_true = find_layers(layer_true)

        # ------------------------------------------------------------
        # (2) Δz 기반 g_block + (5) token_alpha 생성
        # ------------------------------------------------------------
        g_block = None
        token_alpha = None
        if args.kstep > 0:
            g_block, token_alpha = estimate_kstep_metrics_delta(
                i=i,
                layers=layers,
                layer=layer,
                layer_true=layer_true,
                inps=inps,                 # current prefix state
                inps_true=inps_true,       # FP prefix state
                layer_kwargs=layer_kwargs,
                K=args.kstep,
                lambdas=kstep_lambdas,
                dev=dev,
                eps=eps
            )

        # (3) q/k/v용 head-wise g
        g_qkv = None
        if g_block is not None:
            g_qkv = headwise_expand(g_block, n_heads=n_heads)
            # scale match (mean=1)
            g_qkv = g_qkv / (g_qkv.mean() + eps)

        # (기존) up/gate용 g_ff 생성은 유지 (d -> d_ff pull-back)
        g_ff = None
        if g_block is not None:
            Wdown_fp = full_true['mlp.down_proj'].weight.detach().to(torch.float32)  # (d, d_ff)
            g_block_f = g_block.detach().to(torch.float32)                           # (d,)
            g_ff = (Wdown_fp.pow(2).T @ g_block_f)                                   # (d_ff,)
            g_ff = g_ff / (g_ff.mean() + eps)

        # ------------------------------------------------------------
        # 모듈 단위 GPTQ (+ token-weighted H)
        # ------------------------------------------------------------
        for name in module_order:
            module = full[name]

            # row metric 선택
            row_metric = None
            if g_block is not None:
                if name.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")):
                    row_metric = g_qkv
                elif name.endswith(("self_attn.o_proj", "mlp.down_proj")):
                    row_metric = g_block
                elif name.endswith(("mlp.up_proj", "mlp.gate_proj")):
                    row_metric = g_ff
                else:
                    row_metric = None

            # hook으로 입력 activation 확보
            hook_inp = {}
            def _hook(mod, inp, out):
                hook_inp["x"] = inp[0].detach()  # (1, T, d_in)

            h = module.register_forward_hook(_hook)

            helper = Helper(module)

            # (5) token_alpha로 weighted H: x <- sqrt(alpha_t) * x
            for j in range(args.nsamples):
                _ = layer(inps[j].unsqueeze(0), **layer_kwargs)  # hook_inp["x"] 채워짐
                x = hook_inp["x"]

                if token_alpha is not None:
                    # alpha: (T,) -> (1, T, 1)
                    a = token_alpha[j].to(x.device, dtype=torch.float32)
                    a = a / (a.mean() + eps)  # per-sample mean=1 안정화
                    xw = x.to(torch.float32) * torch.sqrt(a.clamp_min(eps)).view(1, -1, 1)
                    helper.add_batch(xw)
                else:
                    helper.add_batch(x)

            h.remove()

            helper.set_row_metric(row_metric)

            # quantize
            if args.method == 'rtn':
                module.weight.data = pseudo_quantize_tensor(
                    module.weight.data, n_bit=args.wbits, q_group_size=args.groupsize
                )
            elif args.method == 'gptq':
                helper.run_gptq(
                    module,
                    percdamp=args.percdamp,
                    wbits=args.wbits,
                    groupsize=args.groupsize,
                    actorder=args.act_order
                )
            elif args.method == 'quip':
                helper.run_quip(
                    module,
                    percdamp=args.percdamp,
                    wbits=args.wbits,
                    multigpu=args.multigpu
                )
            else:
                helper.free()
                raise NotImplementedError

            helper.free()

        # 다음 블록을 위한 입력 업데이트
        for j in range(args.nsamples):
            inps[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            inps_true[j] = layer_true(inps_true[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del layer_true
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

@torch.no_grad()
def llama_eval(model, testenc, dev):
    #print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'].update(kwargs)
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
    # model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.model.norm = model.model.norm.to(dev)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs']

    for i in range(len(layers)):
        #print(i)
        layer = layers[i].to(dev)
        
        # if args.nearest:
        #     subset = find_layers(layer)
        #     for name in subset:
        #         quantizer = Quantizer()
        #         quantizer.configure(
        #             args.wbits, perchannel=True, sym=False, mse=False
        #         )
        #         W = subset[name].weight.data
        #         quantizer.find_params(W, weight=True)
        #         subset[name].weight.data = quantize(
        #             W, quantizer.scale, quantizer.zero, quantizer.maxq
        #         ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
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

    model.config.use_cache = use_cache

    return ppl.item()

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
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, help='Where to extract calibration data from.'
    )
    parser.add_argument(
        'method', type=str, choices=['fp16', 'rtn', 'gptq', 'awq', 'quip'],
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
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--qep', action='store_true',
        help='Whether to use QEP.'
    )
    parser.add_argument(
        '--percdampqep', type=float, default=1.0,
        help='Percent of the average Hessian diagonal to use for dampening of QEP.'
    )
    parser.add_argument(
        '--perccorr', type=float, default=0.5,
        help='Percent of the weight correction.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--save-model', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load-model', type=str, default='',
        help='Load quantized checkpoint from this pass.'
    )

    # Evalution
    parser.add_argument(
        '--save-result', action='store_true',
        help='Whether to save result.'
    )
    parser.add_argument(
        "--ppl", default=None,
        choices=MultiChoice(['wikitext2', 'ptb-new', 'c4-new'])
    )
    parser.add_argument(
        "--tasks", default=None,
        choices=MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument(
        '--generate', type=str, default='',
        help='Write input for model generation. example:"Hello, how are you today?"'
    )
    parser.add_argument(
        '--multigpu', action='store_true',
        help='Whether to use multigpu.'
    )
    parser.add_argument(
        '--kstep', type=int, default=0,
        help='Number of future blocks to use for k-step metric.'
    )
    parser.add_argument(
        '--kstep_ridge', type=float, default=0.1,
        help='Ridge regularization for k-step metric.'
    )
    parser.add_argument(
        '--kstep_lambdas', type=float, nargs='+', default=None,
        help='List of lambdas for k-step metric.'
    )

    args = parser.parse_args()
    args.batch_size = 1  # BS=1 is used for zeroShot tasks!

    model = get_llama(args.model)
    model.eval()

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    elif args.method != "fp16":
        print(f"method={args.method}", flush=True)
        if args.qep:
            print("Use QEP.", flush=True)
        print(args, flush=True)

        if args.method == 'rtn' or args.method == 'gptq' or args.method == 'quip':
            if args.method == 'rtn' and not args.qep:
                pseudo_quantize_model_weight(model, args.wbits, {"q_group_size":args.groupsize})
            else:
                if args.method == 'quip':
                    assert(args.groupsize == -1)
                llama_sequential(model, DEV)
        elif args.method == 'awq':
            enc = AutoTokenizer.from_pretrained(args.model, use_fast=False)
            if args.qep:
                run_awq_with_QEP(
                    model,
                    DEV,
                    args,
                    enc,
                    w_bit=args.wbits,
                    q_config={"q_group_size":args.groupsize},
                    n_samples=args.nsamples,
                    seqlen=512,
                    calib_data=args.dataset,
                )
            else:
                run_awq(
                    model,
                    enc,
                    w_bit=args.wbits,
                    q_config={"q_group_size":args.groupsize},
                    n_samples=args.nsamples,
                    seqlen=512,
                    calib_data=args.dataset,
                )
        else:
            raise NotImplementedError
    
    if args.save_model:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained(args.model)
        enc.save_pretrained(args.save_model)
        model.save_pretrained(args.save_model)
    
    # evalution
    # results = {}
    # if args.ppl is not None:
    #     datasets_names = args.ppl.split(",")
    #     model.seqlen = 2048
    #     for datasets_name in datasets_names:
    #         print(datasets_name, flush=True)
    #         dataloader, testloader = get_loaders(
    #             datasets_name, seed=args.seed, model=args.model, seqlen=model.seqlen
    #         )
    #         ppl = llama_eval(model, testloader, DEV)
    #         print(ppl, flush=True)
    #         results[f"ppl_{datasets_name}"] = ppl

    # if args.tasks is not None:
    #     tasks_results = get_result(args, model)["results"]
    #     flattened_results = {f"{outer}_{inner}": value 
    #                     for outer, inner_dict in tasks_results.items() 
    #                     for inner, value in inner_dict.items()}
    #     flattened_results = dict(sorted(flattened_results.items()))
    #     results.update(flattened_results)
    
    # if args.save_result:
    #     save_experiment_results(args, results)

    # # Text Generation
    # if args.generate:
    #     model = model.to(DEV)
    #     tokenizer = AutoTokenizer.from_pretrained(args.model)
    #     inputs = tokenizer(args.generate, return_tensors="pt").to(model.device)
    #     with torch.no_grad():
    #         output = model.generate(**inputs, max_new_tokens=512)
    #     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print("Generated text:\n", generated_text)
