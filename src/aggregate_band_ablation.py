"""
aggregate_band_ablation.py
 
Reads band ablation result JSONs from experiments/results/ and prints
a summary table of mean ± std accuracy per model and band.
 
Usage:
    python aggregate_band_ablation.py
    python aggregate_band_ablation.py --mode subject_dependent
    python aggregate_band_ablation.py --mode loso
    python aggregate_band_ablation.py --metric test_acc   # if evaluate_128 has been run
"""
 
import argparse
import json
import os
import re
from collections import defaultdict
 
import numpy as np
 
MODELS = [
    "transformer",
    "cnn_gru_alternative",
    "tcn",
    "cnn_lstm_alternative",
    "alternative_eegnet",
]
 
BANDS = ["delta", "theta", "mu", "beta", "gamma", "mu_beta", "full"]
 
RESULTS_DIR = os.path.join("experiments", "results")
 
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
REPS     = [0, 1, 2, 3]
 
 
def load_subject_dependent(model: str, band: str) -> list[float]:
    """
    Load best_val_acc for each subject from subject-dependent result JSONs.
    For 'full', falls back to the original filename without band suffix.
    Returns list of accuracies (one per subject found).
    """
    accs = []
    for subj in SUBJECTS:
        candidates = [f"{model}_{subj}_subject_dependent_128_{band}.json"]
        if band == "full":
            candidates.append(f"{model}_{subj}_subject_dependent_128.json")
        for fname in candidates:
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                accs.append(data["best_val_acc"])
                break
    return accs
 
 
def load_loso(model: str, band: str) -> list[float]:
    """
    Load best_val_acc for each fold from LOSO result JSONs.
    For 'full', falls back to the original filename without band suffix.
    Returns list of accuracies (one per fold found).
    """
    accs = []
    for subj in SUBJECTS:
        for rep in REPS:
            candidates = [f"{model}_{subj}_rep{rep}_loso_128_{band}.json"]
            if band == "full":
                candidates.append(f"{model}_{subj}_rep{rep}_loso_128.json")
            for fname in candidates:
                path = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    accs.append(data["best_val_acc"])
                    break
    return accs
 
 
def fmt(accs: list[float]) -> str:
    if not accs:
        return "   —   "
    mean = np.mean(accs) * 100
    std  = np.std(accs)  * 100
    n    = len(accs)
    return f"{mean:5.1f}±{std:4.1f} (n={n})"
 
 
def print_table(mode: str) -> None:
    loader = load_subject_dependent if mode == "subject_dependent" else load_loso
 
    # Collect all results
    # results[model][band] = list of accuracies
    results = defaultdict(dict)
    for model in MODELS:
        for band in BANDS:
            accs = loader(model, band)
            results[model][band] = accs
 
    # Print table
    band_cols = [b for b in BANDS if any(results[m][b] for m in MODELS)]
 
    col_w     = 18
    model_w   = 25
    header    = f"{'Model':<{model_w}}" + "".join(f"{b:>{col_w}}" for b in band_cols)
    separator = "-" * len(header)
 
    print(f"\n{'='*len(header)}")
    print(f" Mode: {mode}  |  Metric: best_val_acc (mean% ± std%)")
    print(f"{'='*len(header)}")
    print(header)
    print(separator)
 
    for model in MODELS:
        row = f"{model:<{model_w}}"
        for band in band_cols:
            row += f"{fmt(results[model][band]):>{col_w}}"
        print(row)
 
    print(separator)
 
    # Also print a compact version sorted by mu_beta for easy reading
    if "mu_beta" in band_cols and "full" in band_cols:
        print(f"\n--- Degradation vs full band (mu_beta - full, pp) ---")
        print(f"{'Model':<{model_w}} {'mu_beta':>10} {'full':>10} {'delta':>10}")
        print("-" * (model_w + 32))
        for model in MODELS:
            mb   = results[model].get("mu_beta", [])
            full = results[model].get("full",    [])
            d    = results[model].get("delta",   [])
            mb_m   = np.mean(mb)   * 100 if mb   else float("nan")
            full_m = np.mean(full) * 100 if full else float("nan")
            d_m    = np.mean(d)    * 100 if d    else float("nan")
            delta_str = f"{mb_m - full_m:+.1f}pp" if mb and full else "—"
            print(f"{model:<{model_w}} {mb_m:>9.1f}% {full_m:>9.1f}% {d_m:>9.1f}%  ({delta_str})")
 
 
def main() -> None:
    global RESULTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["subject_dependent", "loso", "both"],
                        default="both")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()
    
    RESULTS_DIR = args.results_dir
 
    if args.mode in ("subject_dependent", "both"):
        print_table("subject_dependent")
 
    if args.mode in ("loso", "both"):
        print_table("loso")
 
 
if __name__ == "__main__":
    main()