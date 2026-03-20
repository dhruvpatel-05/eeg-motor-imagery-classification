"""
Preprocessing pipeline for BCI Competition IV Dataset 2a — 128 Hz variant.

Implements the desc_2a.pdf spec with the following differences from preprocess.py:
  - Resample 250 Hz → 128 Hz  (input shrinks from 1000 → 256 samples)
  - Bandpass 4–38 Hz, causal Butterworth order 3 (sosfilt, single-pass)
  - ×1e6 scaling applied before filtering (brings signal to µV range)
  - Exponential running standardisation (ERS) on the continuous signal
  - Trial window 2.5–4.5 s after trial onset  (0.5–2.5 s post-cue)

WARNING: Data produced by this pipeline is ERS-normalised on the continuous
signal.  When training with bci_competition_iv_2a_128/ data, set norm = None
(skip TrialNormalizer / Normalizer) in train.py to avoid double normalisation.

Notes on trial windowing (seconds relative to trial[] onset):
    t = 0.0 s : fixation cross appears        ← trial[] points here
    t = 2.0 s : cue arrow appears (MI onset)
    t = 2.0–6.0 s : motor imagery period
    We extract [2.5, 4.5) s after trial onset, i.e. [0.5, 2.5) s post-cue,
    giving exactly 256 samples at 128 Hz — the peak MI segment.
"""

import math
import os
import sys
import threading
import time

import numpy as np
import scipy.io
import scipy.signal


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "bci_competition_iv_2a"
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "processed",
    "bci_competition_iv_2a_128",
)

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
SPLITS = {"T": "Training", "E": "Evaluation"}
CLASS_LABELS = {1: "Left Hand", 2: "Right Hand", 3: "Both Feet", 4: "Tongue"}

FS_RAW = 250          # native sampling rate (Hz)
FS_TARGET = 128       # target sampling rate after resampling (Hz)
N_EEG = 22            # first 22 channels are EEG (23-25 are EOG)

BANDPASS_LOW = 4.0    # Hz
BANDPASS_HIGH = 38.0  # Hz
FILTER_ORDER = 3

ERS_FACTOR = 1e-3     # exponential decay coefficient α
ERS_EPS = 1e-4        # variance floor to avoid division by zero
ERS_INIT_SAMPLES = 1000  # samples used to initialise running mean/var

# Window 2.5–4.5 s after trial onset  (= 0.5–2.5 s post-cue at 128 Hz)
TRIAL_START = 2.5     # s after trial onset
TRIAL_END = 4.5       # s after trial onset
N_TIMEPOINTS = int((TRIAL_END - TRIAL_START) * FS_TARGET)  # 256


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class _Spinner:
    def __init__(self, msg: str = "Loading"):
        self._msg = msg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        dots = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self._msg}{'.' * dots}   ")
            sys.stdout.flush()
            dots = (dots % 3) + 1
            time.sleep(0.4)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self._msg) + 6) + "\r")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def resample_signal(X: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Polyphase resample X from fs_orig to fs_target Hz.

    Args:
        X: (n_samples, n_channels)
        fs_orig:  original sampling rate
        fs_target: target sampling rate

    Returns:
        (n_samples_new, n_channels) resampled array
    """
    g = math.gcd(fs_orig, fs_target)
    up = fs_target // g
    down = fs_orig // g
    return scipy.signal.resample_poly(X, up, down, axis=0)


def scale_to_microvolts(X: np.ndarray) -> np.ndarray:
    """Multiply signal by 1e6 (convert V → µV in-place)."""
    X *= 1e6
    return X


def causal_bandpass_filter(X: np.ndarray, fs: float) -> np.ndarray:
    """Single-pass (causal) Butterworth bandpass filter.

    Uses sosfilt (forward-only) to avoid any look-ahead / anti-causal
    artefacts at epoch boundaries.

    Args:
        X:  (n_samples, n_channels)
        fs: sampling rate in Hz

    Returns:
        Filtered array of the same shape.
    """
    sos = scipy.signal.butter(
        FILTER_ORDER,
        [BANDPASS_LOW, BANDPASS_HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return scipy.signal.sosfilt(sos, X, axis=0)


def exponential_running_standardize(X: np.ndarray) -> np.ndarray:
    """Online per-channel exponential running standardisation (ERS).

    Normalises the continuous signal using a causal running estimate of mean
    and variance, so no future-sample statistics leak into earlier time points.

    Initialisation: mean and variance are estimated from the first
    ERS_INIT_SAMPLES samples of each channel.

    Update rule (α = ERS_FACTOR):
        running_mean[t] = (1-α)*running_mean[t-1] + α*X[t]
        running_var[t]  = (1-α)*running_var[t-1]  + α*(X[t]-running_mean[t])²
        X_norm[t]       = (X[t] - running_mean[t]) / sqrt(running_var[t] + ε)

    Args:
        X: (n_samples, n_channels) continuous signal, float64 in µV

    Returns:
        X_norm: same shape, float64, ERS-normalised
    """
    n_samples, n_channels = X.shape
    alpha = ERS_FACTOR
    eps = ERS_EPS

    init = min(ERS_INIT_SAMPLES, n_samples)
    running_mean = X[:init].mean(axis=0).copy()          # (n_channels,)
    running_var = X[:init].var(axis=0).copy()            # (n_channels,)

    X_norm = np.empty_like(X, dtype=np.float64)

    for t in range(n_samples):
        running_mean = (1.0 - alpha) * running_mean + alpha * X[t]
        running_var = (1.0 - alpha) * running_var + alpha * (X[t] - running_mean) ** 2
        X_norm[t] = (X[t] - running_mean) / np.sqrt(running_var + eps)

    return X_norm


def extract_trials_128(
    X_norm: np.ndarray,
    trial_onsets_raw: np.ndarray,
    artifacts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice cue-locked windows from the ERS-normalised, resampled signal.

    trial_onsets_raw are in the original 250 Hz sample space; they are scaled
    to 128 Hz before use.

    Args:
        X_norm:          (n_samples_128, n_channels) normalised signal at 128 Hz
        trial_onsets_raw: (n_trials,) integer sample indices at 250 Hz
        artifacts:        (n_trials,) non-zero means artifact-flagged

    Returns:
        epochs:     (n_clean, N_EEG, N_TIMEPOINTS)  i.e. (n, 22, 256)
        clean_idx:  (n_clean,) original trial indices kept
    """
    scale = FS_TARGET / FS_RAW              # 128/250
    start_offset = int(TRIAL_START * FS_TARGET)   # 320 samples at 128 Hz

    kept_epochs = []
    kept_indices = []

    for i, onset_raw in enumerate(trial_onsets_raw):
        if artifacts[i] != 0:
            continue

        onset_128 = int(round(int(onset_raw) * scale))
        s = onset_128 + start_offset
        e = s + N_TIMEPOINTS

        if e > X_norm.shape[0]:
            continue

        window = X_norm[s:e, :N_EEG]       # (256, 22)
        kept_epochs.append(window.T)        # → (22, 256)
        kept_indices.append(i)

    if not kept_epochs:
        raise RuntimeError("No valid (non-artifact) trials found.")

    epochs = np.stack(kept_epochs, axis=0)                     # (n, 22, 256)
    clean_idx = np.array(kept_indices, dtype=np.int64)
    return epochs, clean_idx


# ---------------------------------------------------------------------------
# .mat loading  (same logic as preprocess.py)
# ---------------------------------------------------------------------------

def load_mat(filepath: str) -> dict:
    """Load a BCI Competition IV 2a .mat file, aggregate across all 9 runs.

    Returns a dict with keys: X, trial, y, artifacts, run, fs, gender, age.
    """
    mat = scipy.io.loadmat(filepath, simplify_cells=True)
    runs = mat["data"]

    X_list, trial_list, y_list, art_list, run_list = [], [], [], [], []
    offset = 0

    for run_idx, run in enumerate(runs):
        X = run["X"]                               # (n_samples, 25)
        trial = run["trial"].astype(np.int64)      # (n_trials,)
        y = np.asarray(run["y"]).ravel()
        artifacts = np.asarray(run["artifacts"]).ravel()

        X_list.append(X)
        trial_list.append(trial + offset)
        y_list.append(y)
        art_list.append(artifacts)
        run_list.append(np.full(len(trial), run_idx + 1, dtype=np.int32))
        offset += X.shape[0]

    first = runs[0]
    return {
        "X":         np.vstack(X_list),
        "trial":     np.concatenate(trial_list),
        "y":         np.concatenate(y_list).astype(np.int32),
        "artifacts": np.concatenate(art_list).astype(np.int32),
        "run":       np.concatenate(run_list).astype(np.int32),
        "fs":        int(first["fs"]),
        "gender":    first.get("gender", "unknown"),
        "age":       first.get("age",    "unknown"),
    }


# ---------------------------------------------------------------------------
# Processing & saving
# ---------------------------------------------------------------------------

def process(mat: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: resample → µV → causal filter → ERS → epoch.

    Returns:
        X_epochs: (n_clean_trials, 22, 256)  float64
        y:        (n_clean_trials,)           int32, values in {1,2,3,4}
        run:      (n_clean_trials,)           int32
    """
    X = mat["X"].astype(np.float64)        # (n_samples, 25)

    # 1. Resample 250 → 128 Hz (all 25 channels; only 22 are used later)
    X = resample_signal(X, FS_RAW, FS_TARGET)

    # 2. Scale to microvolts
    X = scale_to_microvolts(X)

    # 3. Causal bandpass 4–38 Hz (order 3, single-pass)
    X = causal_bandpass_filter(X, FS_TARGET)

    # 4. Exponential running standardisation on the continuous signal
    X = exponential_running_standardize(X)

    # 5. Epoch at 128 Hz (trial onsets are still in 250 Hz sample space)
    X_epochs, clean_idx = extract_trials_128(X, mat["trial"], mat["artifacts"])

    y = mat["y"][clean_idx]
    run = mat["run"][clean_idx]

    return X_epochs, y, run


def save(X: np.ndarray, y: np.ndarray, run: np.ndarray, fname_stem: str) -> None:
    """Save X, y, and run arrays as .npy files under OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"{fname_stem}_X.npy"), X)
    np.save(os.path.join(OUT_DIR, f"{fname_stem}_y.npy"), y)
    np.save(os.path.join(OUT_DIR, f"{fname_stem}_run.npy"), run)


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def summarize(
    mat: dict,
    X_epochs: np.ndarray,
    y: np.ndarray,
    n_dropped: int,
    subject: str,
    split_name: str,
) -> None:
    print(f"  Subject {subject} | {split_name}")
    print(f"  {'─' * 54}")
    print(f"  Channels      : {mat['X'].shape[1]} total  ({N_EEG} EEG + {25-N_EEG} EOG)")
    print(f"  Native fs     : {mat['fs']} Hz  →  resampled to {FS_TARGET} Hz")
    print(f"  Bandpass      : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz  (order {FILTER_ORDER}, causal)")
    print(f"  Normalisation : ERS  (α={ERS_FACTOR}, ε={ERS_EPS})")
    print(f"  Trial window  : {TRIAL_START}–{TRIAL_END} s after trial onset  ({N_TIMEPOINTS} samples)")
    print(f"  Epochs shape  : {X_epochs.shape}  (trials × channels × time)")
    print(f"  Subject       : {mat['gender']}, age {mat['age']}")
    print(f"  Trials kept   : {len(y)}  ({n_dropped} artifact-flagged, removed)")
    for code, label in CLASS_LABELS.items():
        count = int(np.sum(y == code))
        bar = "█" * (count // 4)
        print(f"    {label:<12}: {count:>3}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 58)
    print("  BCI Competition IV 2a — Preprocessing  (128 Hz / ERS)")
    print("=" * 58)
    print()

    for subject in SUBJECTS:
        for suffix, split_name in SPLITS.items():
            fname = f"{subject}{suffix}.mat"
            fpath = os.path.join(DATA_DIR, fname)

            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found — skipping\n")
                continue

            spinner = _Spinner(f"  Processing {fname}")
            spinner.start()
            mat = load_mat(fpath)
            X_epochs, y, run = process(mat)
            save(X_epochs, y, run, f"{subject}{suffix}")
            spinner.stop()

            n_dropped = int(np.sum(mat["artifacts"]))
            summarize(mat, X_epochs, y, n_dropped, subject, split_name)

    print(f"  Saved to: {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
