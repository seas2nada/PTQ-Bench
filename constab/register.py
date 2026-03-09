from utils.register import register_method
import os
import subprocess
@register_method("constab")
def run(config):
    model_path = config["model_path"]
    dataset = config["dataset"]
    wbits = str(config["wbits"])
    save_path = config["save_path"]
    group_size = config["group_size"]
    device = config.get("CUDA_VISIBLE_DEVICES", None)
    act_order = config.get("act_order", False)
    cmd = [
        "python", "constab/run.py",
        model_path, dataset,
        "--wbits", wbits,
        "--save", save_path,
        "--groupsize", str(group_size),
        "--nsamples", str(config["nsamples"]),
        "--stab-tau", str(config["stab_tau"]),
        "--stab-iters", str(config["stab_iters"]),
        "--stab-damp-mult", str(config["stab_damp_mult"]),
        "--stab-max-percdamp", str(config["stab_max_percdamp"]),
        "--stab-jac-nsamples", str(config["stab_jac_nsamples"]),
        "--stab-jac-power", str(config["stab_jac_power"]),
        "--stab-scope", str(config["stab_scope"]),
        "--stab-use-sigma", str(config["stab_use_sigma"]),
        "--stab-junc-mult", str(config["stab_junc_mult"]),
        # "--stab-route-after", str(config["stab_route_after"]),
        # "--stab-route-downproj-bit", str(config["stab_route_downproj_bit"]),
        # "--stab-gain-nsamples", str(config["stab_gain_nsamples"]),
    ]
    if act_order:
        cmd.append("--act-order")
    if config.get("stab_gain_calib", False):
        cmd.append("--stab-gain-calib")
    # if device:
    #     env = {"CUDA_VISIBLE_DEVICES": device, **os.environ}
    # else:
    #     env = os.environ
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)