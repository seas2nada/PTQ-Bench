from utils.register import register_method
import os
import subprocess
@register_method("awq-qep")
@register_method("gptq-qep")
def run(config):
    model_path = config["model_path"]
    dataset = config["dataset"]
    wbits = str(config["wbits"])
    save_path = config["save_path"]
    group_size = config["group_size"]
    device = config.get("CUDA_VISIBLE_DEVICES", None)
    act_order = config.get("act_order", False)
    qep = config.get("qep", False)
    method_name = config.get("method", "awq")
    if method_name == "awq-qep":
        method = "awq"
    elif method_name == "gptq-qep":
        method = "gptq"
    cmd = [
        "python", "qep/llama.py",
        model_path, dataset, method,
        "--wbits", wbits,
        "--save-model", save_path,
        "--groupsize", str(group_size),
    ]
    if act_order:
        cmd.append("--act-order")
    if qep:
        cmd.append("--qep")
    # if device:
    #     env = {"CUDA_VISIBLE_DEVICES": device, **os.environ}
    # else:
    #     env = os.environ
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)