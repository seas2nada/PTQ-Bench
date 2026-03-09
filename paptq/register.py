from utils.register import register_method
import os
import subprocess
@register_method("paptq")
def run(config):
    model_path = config["model_path"]
    dataset = config["dataset"]
    wbits = str(config["wbits"])
    save_path = config["save_path"]
    group_size = config["group_size"]
    device = config.get("CUDA_VISIBLE_DEVICES", None)
    act_order = config.get("act_order", False)
    cmd = [
        "python", "paptq/run.py",
        model_path, dataset,
        "--wbits", wbits,
        "--save", save_path,
        "--groupsize", str(group_size),
        "--nsamples", str(config["nsamples"]),
        "--prop-kstep", str(config["prop_kstep"]),
        "--prop-nprobe", str(config["prop_nprobe"]),
        "--prop-hutch", str(config["prop_hutch"]),
    ]
    if act_order:
        cmd.append("--act-order")
    # if device:
    #     env = {"CUDA_VISIBLE_DEVICES": device, **os.environ}
    # else:
    #     env = os.environ
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)