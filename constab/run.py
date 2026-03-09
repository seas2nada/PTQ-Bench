import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *


if __name__ == '__main__':
    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, help='Where to extract calibration data from.'
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
        '--save', type=str, default=None,
        help='Where to save quantized model'
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
        '--stab-max-percdamp', type=float, default=0.03,
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
    parser.add_argument('--stab-use-sigma', type=int, default=0,
                        help='1이면 Jacobian sigma constraint도 사용, 0이면 junction+finite만 사용.')
    parser.add_argument('--stab-junc-mult', type=float, default=3.0,
                        help='MLP junction RMS allowed multiplier over FP reference.')
    parser.add_argument('--stab-route-after', type=int, default=2,
                        help='After this many failed refinements, enable stability routing for down_proj.')
    parser.add_argument('--stab-route-downproj-bit', type=int, default=3,
                        help='If >0, quantize mlp.down_proj with at least this many bits when routing is enabled (e.g., 3 or 4).')
    parser.add_argument('--stab-gain-calib', action='store_true',
                    help='Calibrate block output scale to reduce PPL degradation.')
    parser.add_argument('--stab-gain-nsamples', type=int, default=8,
                    help='How many calibration samples for gain estimation.')

    args = parser.parse_args()
    tokenizer = None
    if "llama" in args.model.lower():
        from llama import get_llama, llama_sequential
        model = get_llama(args.model)
        sequential = llama_sequential
    elif "llava" in args.model.lower() or "vila" in args.model.lower():
        from llava_gptq import get_llava, llava_sequential
        model, tokenizer = get_llava(args.model)
        sequential = llava_sequential
    elif "deepseek" in args.model.lower() or "mixtral" in args.model.lower() or "mistral" in args.model.lower():
        from mistral import get_mixtral, mixtral_sequential
        model = get_mixtral(args.model)
        sequential = mixtral_sequential
    elif "mamba" in args.model.lower():
        from mamba_gptq import get_mamba,  mamba_sequential
        model, tokenizer = get_mamba(args.model)
        sequential = mamba_sequential
    model.eval()
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    if args.save:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)
        model.save_pretrained(args.save)
