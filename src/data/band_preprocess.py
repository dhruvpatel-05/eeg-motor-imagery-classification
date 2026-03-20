"""
Frequency-band ablation preprocessor for BCI Competition IV Dataset 2a.
 
Replicates the alternative_preprocess.py pipeline but with a parameterized
bandpass cutoff, so each band gets its own processed data directory.
 
Usage:
    python src/data/band_preprocess.py --band mu
    python src/data/band_preprocess.py --band beta
    # ... etc for all bands
 
Output:
    data/processed/bci_competition_iv_2a_128_{band}/
        {subject}{split}_X.npy   — (n_trials, 22, 256)  float32
        {subject}{split}_y.npy   — (n_trials,)           int32
        {subject}{split}_run.npy — (n_trials,)           int32
 
Ordering of operations (must match alternative_preprocess.py):
    1. Resample 250 → 128 Hz
    2. Scale V → µV
    3. Causal bandpass (band-specific, order 3, sosfilt)
    4. Exponential running standardization on continuous signal
    5. Epoch [2.5, 4.5) s after trial onset  →  (22, 256) per trial
    6. Drop artifact-flagged trials
 
NOTE: The 'full' band (4–38 Hz) reproduces alternative_preprocess.py exactly.
      You can symlink the existing processed dir instead of re-running:
          ln -s data/processed/bci_competition_iv_2a_128 \
                data/processed/bci_competition_iv_2a_128_full
"""

import math
import os
import sys
import threading
import time
import argparse

import numpy as np
import scipy.io
import scipy.signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANDS = {
    "delta": (1,4),
    "theta": (4,8),
    "mu": (8,13),
    "beta": (13,30),
    "gamma": (30, 40),
    "mu_beta": (8, 30),
    "full": (4, 38),
}

FS_RAW = 250          # native sampling rate (Hz)
FS_TARGET = 128       # target sampling rate after resampling (Hz)
FILTER_ORDER = 3
N_EEG = 22

TRIAL_START = 2.5     # s after trial onset
TRIAL_END = 4.5       # s after trial onset
N_TIMEPOINTS = int((TRIAL_END - TRIAL_START) * FS_TARGET)  # 256

ERS_FACTOR = 1e-3     # exponential decay coefficient α
ERS_EPS = 1e-4        # variance floor to avoid division by zero
ERS_INIT_SAMPLES = 1000  # samples used to initialise running mean/var

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
SPLITS = {"T": "Training", "E": "Evaluation"}
CLASS_LABELS = {1: "Left Hand", 2: "Right Hand", 3: "Both Feet", 4: "Tongue"}

# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------
 
def resample_signal(X: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Polyphase resample. X: (n_samples, n_channels)."""
    g    = math.gcd(fs_orig, fs_target)
    up   = fs_target // g
    down = fs_orig   // g
    return scipy.signal.resample_poly(X, up, down, axis=0)
 
 
def causal_bandpass(X: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Single-pass (causal) Butterworth bandpass. X: (n_samples, n_channels)."""
    sos = scipy.signal.butter(
        FILTER_ORDER, [low, high], btype="bandpass", fs=fs, output="sos"
    )
    return scipy.signal.sosfilt(sos, X, axis=0)
 
 
def ers_normalize(X: np.ndarray) -> np.ndarray:
    """
    Per-channel exponential running standardization on the continuous signal.
    Causal — no future-sample leakage.
    X: (n_samples, n_channels)
    """
    out  = np.empty_like(X)
    mean = X[:ERS_INIT_SAMPLES].mean(axis=0)
    var  = X[:ERS_INIT_SAMPLES].var(axis=0)
 
    for t in range(len(X)):
        x     = X[t]
        mean  = (1 - ERS_FACTOR) * mean + ERS_FACTOR * x
        var   = (1 - ERS_FACTOR) * var  + ERS_FACTOR * (x - mean) ** 2
        out[t] = (x - mean) / np.sqrt(var + ERS_EPS)
 
    return out
 

# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------
 
def load_mat(filepath: str) -> dict:
    """Load .mat, aggregate all 9 runs into a single continuous signal."""
    mat  = scipy.io.loadmat(filepath, simplify_cells=True)
    runs = mat["data"]
 
    X_list, trial_list, y_list, art_list, run_list = [], [], [], [], []
    offset = 0
 
    for run_idx, run in enumerate(runs):
        X         = run["X"]
        trial     = run["trial"].astype(np.int64)
        y         = np.asarray(run["y"]).ravel()
        artifacts = np.asarray(run["artifacts"]).ravel()
 
        X_list.append(X)
        trial_list.append(trial + offset)
        y_list.append(y)
        art_list.append(artifacts)
        run_list.append(np.full(len(trial), run_idx + 1, dtype=np.int32))
        offset += X.shape[0]
 
    first = runs[0]
    return {
        "X":         np.vstack(X_list).astype(np.float64),
        "trial":     np.concatenate(trial_list),
        "y":         np.concatenate(y_list).astype(np.int32),
        "artifacts": np.concatenate(art_list).astype(np.int32),
        "run":       np.concatenate(run_list).astype(np.int32),
        "fs":        int(first["fs"]),
    }

def process_subject(
    mat_path: str,
    low: float,
    high: float,
    out_dir: str,
    subject_split: str,
) -> None:
    mat = load_mat(mat_path)
 
    X         = mat["X"]          # (n_samples, 25)
    trials    = mat["trial"]
    y         = mat["y"]
    artifacts = mat["artifacts"]
    run_ids   = mat["run"]
 
    # 1. Resample 250 → 128 Hz (all channels; EOG channels discarded later)
    X      = resample_signal(X, FS_RAW, FS_TARGET)
    trials = (trials * FS_TARGET / FS_RAW).astype(np.int64)
 
    # 2. Scale V → µV
    X *= 1e6
 
    # 3. Causal bandpass on EEG channels only
    X[:, :N_EEG] = causal_bandpass(X[:, :N_EEG], low, high, FS_TARGET)
 
    # 4. ERS normalization on EEG channels
    X[:, :N_EEG] = ers_normalize(X[:, :N_EEG])
 
    # 5. Epoch [2.5, 4.5) s after trial onset
    start_samp = int(TRIAL_START * FS_TARGET)
    end_samp   = int(TRIAL_END   * FS_TARGET)
 
    epochs, ys, runs_out = [], [], []
    for i, onset in enumerate(trials):
        if artifacts[i]:
            continue
        s = onset + start_samp
        e = onset + end_samp
        if e > X.shape[0]:
            continue
        epochs.append(X[s:e, :N_EEG].T)   # → (22, 256)
        ys.append(y[i])
        runs_out.append(run_ids[i])
 
    X_out = np.stack(epochs, axis=0).astype(np.float32)  # (n_trials, 22, 256)
    y_out = np.array(ys,        dtype=np.int32)
    r_out = np.array(runs_out,  dtype=np.int32)
 
    np.save(os.path.join(out_dir, f"{subject_split}_X.npy"),   X_out)
    np.save(os.path.join(out_dir, f"{subject_split}_y.npy"),   y_out)
    np.save(os.path.join(out_dir, f"{subject_split}_run.npy"), r_out)
    print(f"  {subject_split}: {X_out.shape}")
 

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Band-specific EEG preprocessor")
    parser.add_argument(
        "--band", required=True, choices=list(BANDS.keys()),
        help="Frequency band to extract"
    )
    parser.add_argument(
        "--raw_dir", default="data/raw/bci_competition_iv_2a",
        help="Directory containing raw .mat files"
    )
    args = parser.parse_args()
 
    low, high = BANDS[args.band]
    out_dir   = f"data/processed/bci_competition_iv_2a_128_{args.band}"
    os.makedirs(out_dir, exist_ok=True)
 
    print(f"Band : {args.band}  ({low}–{high} Hz)")
    print(f"Output: {out_dir}\n")
 
    for subj in SUBJECTS:
        for split in SPLITS:
            mat_path = os.path.join(args.raw_dir, f"{subj}{split}.mat")
            if not os.path.exists(mat_path):
                print(f"  {subj}{split}: not found, skipping")
                continue
            process_subject(mat_path, low, high, out_dir, f"{subj}{split}")
 
    print("\nDone.")