import argparse
import yaml
from utils.register import get_method, METHOD_REGISTRY
import importlib
import importlib.util
import os
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
def load_all_methods():
    methods_dir = os.path.join(os.path.dirname(__file__))
    for method_name in ['gptq', "QuIP", "OmniQuant", "awq", "c-gptq"]:
        method_path = os.path.join(methods_dir, method_name, 'register.py')
        if os.path.isfile(method_path):
            module_name = f"{method_name}.register"
            spec = importlib.util.spec_from_file_location(module_name, method_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="e.g., awq, gptq")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dataset", required=False, help="Override dataset in config")
    parser.add_argument("--save_path", required=False, help="Override save path in config")
    parser.add_argument("--bits", required=False, help="Override bits in config")
    parser.add_argument("--group_size", required=False, help="Override group_size in config")
    parser.add_argument("--nsamples", required=False, help="Override nsamples in config")
    parser.add_argument("--list", action="store_true", help="List all available methods")
    parser.add_argument("--h-in", required=False, help="Path to previous Hessian/Gram state (torch.save).")
    parser.add_argument("--h-out", required=False, help="Path to save updated Hessian/Gram state (torch.save).")
    parser.add_argument("--h-pi", required=False, help="Task weight π_t for current calibration set.")
    args = parser.parse_args()
    load_all_methods()

    if args.list:
        print("Available methods:")
        for name in METHOD_REGISTRY:
            print(f" - {name}")
        return
    config = load_config(args.config)
    if args.dataset:
        config["dataset"] = args.dataset
    if args.save_path:
        config["save_path"] = args.save_path
    if args.bits:
        config["w_bit"] = args.bits
        config["wbits"] = args.bits
    if args.group_size:
        config["group_size"] = args.group_size
    if args.nsamples:
        config["nsamples"] = args.nsamples
    if args.h_in:
        config["h_in"] = args.h_in
    if args.h_out:
        config["h_out"] = args.h_out
    if args.h_pi:
        config["h_pi"] = args.h_pi
    run_func = get_method(args.method)
    run_func(config)
if __name__ == "__main__":
    main()
