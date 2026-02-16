from utils.register import register_method
import os
import subprocess
@register_method("awq-proposed")
@register_method("gptq-proposed")
def run(config):
    model_path = config["model_path"]
    dataset = config["dataset"]
    wbits = str(config["wbits"])
    save_path = config["save_path"]
    group_size = config["group_size"]
    device = config.get("CUDA_VISIBLE_DEVICES", None)
    act_order = config.get("act_order", False)
    method_name = config.get("method", "awq")
    if method_name == "awq-proposed":
        method = "awq"
    elif method_name == "gptq-proposed":
        method = "gptq"
    cmd = [
        "python", "proposed/llama.py",
        model_path, dataset, method,
        "--wbits", wbits,
        "--save-model", save_path,
        "--groupsize", str(group_size),
        "--kstep", str(config["kstep"]),
        "--kstep_ridge", str(config["kstep_ridge"]),
        "--kstep_lambdas", *map(str, config["kstep_lambdas"]),
    ]
    if act_order:
        cmd.append("--act-order")
    # if device:
    #     env = {"CUDA_VISIBLE_DEVICES": device, **os.environ}
    # else:
    #     env = os.environ
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)