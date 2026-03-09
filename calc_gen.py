import numpy as np
from dataclasses import dataclass

# --------------------------------------------
# Define metrics from the (left) table:
# Compare: Single-domain (WikiText2, nsamples=512) vs Continual (Ours, ALL, nsamples=512)
# Metrics: PPL (WikiText2, lower=better) + Acc (BoolQ/PIQA/Wino, higher=better)
# --------------------------------------------

@dataclass
class Row:
    ppl: float
    boolq: float
    piqa: float
    wino: float

single = {
    4: Row(ppl=5.58, boolq=77.89, piqa=77.80, wino=68.35),
    3: Row(ppl=6.15, boolq=72.35, piqa=76.50, wino=66.46),
    2: Row(ppl=39.10, boolq=42.63, piqa=51.14, wino=49.01),
}

cont = {
    4: Row(ppl=5.58, boolq=79.88, piqa=77.75, wino=68.59),
    3: Row(ppl=6.18, boolq=74.74, piqa=76.39, wino=67.56),
    2: Row(ppl=49.44, boolq=49.56, piqa=55.50, wino=52.96),
}

# --------------------------------------------
# Relative improvement (%)
# - For higher-better metrics: (cont - single) / single * 100
# - For lower-better metrics (PPL): (single - cont) / single * 100
# --------------------------------------------

def rel_improve_higher_better(single_val, cont_val):
    return (cont_val - single_val) / single_val * 100.0

def rel_improve_lower_better(single_val, cont_val):
    return (single_val - cont_val) / single_val * 100.0

# Collect all relative improvements (PPL + 3 accuracies) over all wbits
improvements = []
detail = {}  # (wbits -> dict of metric improvements)

for w in [4, 3, 2]:
    s, c = single[w], cont[w]
    d = {
        "PPL":  rel_improve_lower_better(s.ppl, c.ppl),
        "BoolQ": rel_improve_higher_better(s.boolq, c.boolq),
        "PIQA":  rel_improve_higher_better(s.piqa, c.piqa),
        "Wino":  rel_improve_higher_better(s.wino, c.wino),
    }
    detail[w] = d
    improvements.extend(list(d.values()))

x = float(np.mean(improvements))

print(f"Overall average relative improvement including PPL + accuracies: x = {x:.2f}%\n")
print("Per-wbits breakdown (relative improvement %; + is better):")
for w in [4, 3, 2]:
    d = detail[w]
    print(f"  wbits={w}: " +
          ", ".join([f"{k}={d[k]:+.2f}%" for k in ["PPL","BoolQ","PIQA","Wino"]]))

# --------------------------------------------
# Optional: If you want to change weighting (e.g., give PPL less weight),
# set weights and compute weighted mean.
# Example: weight PPL=0.5, each QA task=1.0
# --------------------------------------------
weights = {"PPL": 1.0, "BoolQ": 1.0, "PIQA": 1.0, "Wino": 1.0}

weighted_vals = []
weighted_wts = []
for w in [4, 3, 2]:
    for k, v in detail[w].items():
        weighted_vals.append(v)
        weighted_wts.append(weights[k])

x_weighted = float(np.average(weighted_vals, weights=weighted_wts))
print(f"\nWeighted average (PPL weight={weights['PPL']}, others=1): x = {x_weighted:.2f}%")