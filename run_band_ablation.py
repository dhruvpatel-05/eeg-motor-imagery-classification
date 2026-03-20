"""
run_band_ablation.py
 
Runs frequency band ablations using fixed best hyperparameters per model/mode.
The only variable that changes across runs is --band.
 
Internally calls run_subject_dependent_128.py and run_loso_128.py as subprocesses,
matching their CLI interfaces exactly.
 
Usage:
    # All models, all bands, both modes
    python run_band_ablation.py
 
    # Subset
    python run_band_ablation.py --models transformer tcn
    python run_band_ablation.py --bands mu beta mu_beta
    python run_band_ablation.py --mode subject_dependent
    python run_band_ablation.py --models alternative_eegnet --bands mu beta --mode loso
 
Prerequisites:
    Band-specific data directories must exist. For each band other than 'full':
        python src/data/band_preprocess.py --band <band>
    train_128.py must accept --band argument (see patch_train_128.py).
"""
 
import argparse
import os
import subprocess
import sys
import time
 
# ---------------------------------------------------------------------------
# Best hyperparameters — frozen from main experiment results.
# Only --band varies across ablation runs; everything else is fixed.
#
# Format: each value is a list of CLI tokens passed to the runner script.
# ---------------------------------------------------------------------------
 
BEST_CONFIGS = {
    "subject_dependent": {
        "transformer": [
            "--epochs", "500", "--lr", "0.0005", "--weight_decay", "1e-4",
            "--batch_size", "16", "--dropout", "0.3",
        ],
        "cnn_gru_alternative": [
            "--epochs", "300", "--lr", "0.001", "--weight_decay", "5e-4",
            "--batch_size", "8", "--dropout", "0.4",
        ],
        "tcn": [
            "--epochs", "300", "--lr", "0.001", "--weight_decay", "5e-4",
            "--batch_size", "8", "--dropout", "0.4",
        ],
        "cnn_lstm_alternative": [
            "--epochs", "300", "--lr", "0.001", "--weight_decay", "5e-4",
            "--batch_size", "16", "--dropout", "0.4",
        ],
        "alternative_eegnet": [
            "--epochs", "300", "--lr", "0.001", "--weight_decay", "5e-4",
            "--dropout", "0.4",
        ],
    },
    "loso": {
        "transformer": [
            "--reps", "4", "--epochs", "300", "--lr", "0.001",
            "--weight_decay", "1e-4", "--batch_size", "32", "--dropout", "0.25",
        ],
        "cnn_gru_alternative": [
            "--reps", "4", "--epochs", "300", "--lr", "0.001",
            "--weight_decay", "5e-4", "--batch_size", "32",
        ],
        "tcn": [
            "--reps", "4", "--epochs", "300", "--lr", "0.001",
            "--weight_decay", "1e-4", "--batch_size", "32",
        ],
        "cnn_lstm_alternative": [
            "--reps", "4", "--epochs", "300", "--lr", "0.001",
            "--weight_decay", "1e-4", "--batch_size", "32", "--dropout", "0.4",
        ],
        "alternative_eegnet": [
            "--reps", "4", "--epochs", "300", "--lr", "0.001",
            "--weight_decay", "1e-4", "--batch_size", "64", "--dropout", "0.25",
        ],
    },
}
 
ALL_MODELS = list(BEST_CONFIGS["subject_dependent"].keys())
ALL_BANDS  = ["delta", "theta", "mu", "beta", "gamma", "mu_beta"]
ALL_MODES  = ["subject_dependent", "loso"]
 
RUNNER = {
    "subject_dependent": "run_subject_dependent_128.py",
    "loso":              "run_loso_128.py",
}
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s"
 
 
def check_data_dirs(bands: list[str]) -> None:
    missing = []
    for band in bands:
        path = os.path.join("data", "processed", f"bci_competition_iv_2a_128_{band}")
        if not os.path.isdir(path):
            missing.append((band, path))
    if missing:
        print("[ERROR] Missing preprocessed data directories:")
        for band, path in missing:
            print(f"  {band}: {path}")
        print("\nRun for each missing band:")
        for band, _ in missing:
            print(f"  python src/data/band_preprocess.py --band {band}")
        sys.exit(1)
 
 
def run_one(model: str, mode: str, band: str) -> float:
    """Launch one runner subprocess. Returns wall-clock time in seconds."""
    script = RUNNER[mode]
    args   = BEST_CONFIGS[mode][model]
 
    cmd = [sys.executable, script, "--model", model, "--band", band, *args]
 
    print(f"\n{'='*60}")
    print(f"  model={model}  mode={mode}  band={band}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
 
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
 
    if result.returncode != 0:
        print(f"[WARN] Non-zero exit ({result.returncode}) for "
              f"{model}/{mode}/band={band} — continuing.")
 
    return elapsed
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frequency band ablation sweep with fixed best hyperparameters."
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
        metavar="MODEL",
        help=f"Models to ablate. Default: all ({', '.join(ALL_MODELS)})"
    )
    parser.add_argument(
        "--bands", nargs="+", default=ALL_BANDS, choices=ALL_BANDS,
        metavar="BAND",
        help=f"Bands to sweep. Default: all ({', '.join(ALL_BANDS)})"
    )
    parser.add_argument(
        "--mode", choices=ALL_MODES, default=None,
        help="Restrict to one mode. Default: both subject_dependent and loso."
    )
    args = parser.parse_args()
 
    modes = [args.mode] if args.mode else ALL_MODES
 
    # Validate data directories up front
    check_data_dirs(args.bands)
 
    total_runs = len(args.models) * len(args.bands) * len(modes)
    print(f"\nBand ablation sweep")
    print(f"  Models : {args.models}")
    print(f"  Bands  : {args.bands}")
    print(f"  Modes  : {modes}")
    print(f"  Total  : {total_runs} runs\n")
 
    run_i    = 0
    t_start  = time.time()
    timings  = []  # (model, mode, band, elapsed)
 
    for model in args.models:
        for mode in modes:
            for band in args.bands:
                run_i += 1
                print(f"\n[{run_i}/{total_runs}]", end=" ")
 
                elapsed = run_one(model, mode, band)
                timings.append((model, mode, band, elapsed))
 
                wall = time.time() - t_start
                avg  = wall / run_i
                eta  = avg * (total_runs - run_i)
                print(f"  Finished in {fmt_time(elapsed)} | "
                      f"total {fmt_time(wall)} | ETA {fmt_time(eta)}")
 
    # Summary table
    total_wall = time.time() - t_start
    print(f"\n{'='*60}")
    print(f" Ablation complete — {total_runs} runs in {fmt_time(total_wall)}")
    print(f"{'='*60}")
    print(f"\n{'Model':<25} {'Mode':<20} {'Band':<10} {'Time'}")
    print("-" * 65)
    for model, mode, band, elapsed in timings:
        print(f"{model:<25} {mode:<20} {band:<10} {fmt_time(elapsed)}")
 
 
if __name__ == "__main__":
    main()