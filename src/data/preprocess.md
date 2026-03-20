# Preprocessing Pipeline — BCI Competition IV Dataset 2a

## Overview

`src/data/preprocess.py` loads all 18 `.mat` subject files (9 training + 9 evaluation), applies the EEGNet paper preprocessing pipeline (Lawhern et al. 2018), and writes the resulting tensors to `data/processed/bci_competition_iv_2a/`.

---

## Input Data Format

Each `.mat` file contains a `data` struct array of shape `(1, 9)` — one element per recording run. Each run element has:

| Field       | Shape             | Description                                               |
|-------------|-------------------|-----------------------------------------------------------|
| `X`         | `(n_samples, 25)` | Continuous EEG+EOG signal (V, float64)                    |
| `trial`     | `(n_trials,)`     | Sample index of each trial cue onset                      |
| `y`         | `(n_trials,)`     | Class label: 1=Left Hand, 2=Right Hand, 3=Feet, 4=Tongue |
| `artifacts` | `(n_trials,)`     | 0 = clean, 1 = artifact-contaminated                      |
| `fs`        | scalar            | Sampling rate (250 Hz)                                    |

Channels 1–22 are EEG; channels 23–25 are EOG and are discarded before filtering.

### Run structure (T session)

| Run indices | Content                   | Trials  |
|-------------|---------------------------|---------|
| 0–2         | Non-MI (fixation/rest)    | 0       |
| 3–8         | Motor imagery             | 48 each |

Total per T session: **288 MI trials** (6 runs × 48).

---

## Processing Steps

### 1. Aggregation across runs (`load_mat`)

All 9 runs are concatenated into a single continuous signal. Trial onset indices are offset by the cumulative sample count of prior runs. The run index (0-based) for each trial is recorded in a `run` array.

```
X_full    = vstack([run.X for run in runs])           # (N_total_samples, 25)
trial_abs = trial_local + cumulative_sample_offset    # absolute onset positions
run[i]    = run_index for trial i                     # (n_trials,)
```

### 2. Resample 250 Hz → 128 Hz

`scipy.signal.resample_poly` resamples the continuous signal before any epoching. Trial onset indices are scaled proportionally.

### 3. Convert V → µV

Multiply by 1e6 to match the scale expected by the EEGNet paper.

### 4. Causal Butterworth bandpass filter (`bandpass_filter`)

- **Passband:** 4–38 Hz
- **Order:** 3 (causal, forward-only via `sosfilt`)
- Applied to the full continuous signal on EEG channels only (22 channels)

### 5. Exponential running standardization (`exp_running_standardize`)

Per-channel online normalization on the continuous signal:

```
mean ← (1 - α) * mean + α * x
var  ← (1 - α) * var  + α * (x - mean)²
x_norm = (x - mean) / sqrt(var + ε)
```

- `α = 1e-3`, initialized from the first 1000 samples, `ε = 1e-4`

### 6. Trial window extraction (`extract_trials`)

```
window = X_norm[onset + 64 : onset + 320, :22]   # 0.5–2.5 s post cue @ 128 Hz
epoch  = window.T                                  # (22, 256)
```

- **Window:** 0.5–2.5 s after cue onset → **256 samples** at 128 Hz
- Trials whose window would extend past signal end are dropped (rare edge case)

---

## Output Files

Written to `data/processed/bci_competition_iv_2a/`:

| File           | Shape          | dtype   | Contents                              |
|----------------|----------------|---------|---------------------------------------|
| `A01T_X.npy`   | `(288, 22, 256)` | float64 | Preprocessed EEG epochs (T session) |
| `A01T_y.npy`   | `(288,)`       | int32   | Class labels 1–4                     |
| `A01T_run.npy` | `(288,)`       | int32   | Run index (0-based) per trial        |
| `A01E_X.npy`   | `(*, 22, 256)` | float64 | Evaluation session epochs            |
| `A01E_y.npy`   | `(*,)`         | int32   | Evaluation labels                    |
| `A01E_run.npy` | `(*,)`         | int32   | Run indices for evaluation session   |
| ...            | ...            | ...     | Repeated for A02–A09                 |

Labels use 1-indexed convention (raw dataset); `BCIDataLoader` shifts to 0-indexed at load time.

---

## Design Decisions

**Why 128 Hz?** Matches the EEGNet paper (Lawhern et al. 2018). Motor imagery bands (mu 8–13 Hz, beta 13–30 Hz) are well below the Nyquist limit of 64 Hz.

**Why 0.5–2.5 s window?** Follows EEGNet paper. Avoids the visual evoked potential at cue onset and captures the motor imagery period within the 4-second trial.

**Why causal filter?** Matches EEGNet paper's causal `sosfilt`. Zero-phase filters would leak future information into past samples — inappropriate for online BCI scenarios.

**Why keep artifact-flagged trials?** Artifact labels are preserved in output but not used to exclude trials. Downstream code can choose to filter or weight them.

**Why save `_run.npy`?** `splits.py` uses run membership to construct blockwise cross-validation folds aligned with the paper's protocol.

---

## Usage

```bash
python src/data/preprocess.py
```

Processes all 18 files with a per-file spinner. Output written to `data/processed/bci_competition_iv_2a/`.
