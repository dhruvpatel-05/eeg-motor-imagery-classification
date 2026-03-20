# MindReader: EEG Motor Imagery Classification

A benchmark comparing neural network architectures for classifying imagined movements from EEG signals, with a focus on cross-subject generalisation in brain-computer interfaces.

**Task:** 4-class motor imagery — left hand, right hand, both feet, tongue.
**Dataset:** BCI Competition IV Dataset 2a — 9 subjects, 22 EEG channels, 288 trials per subject.

---

## Repository Structure

```
data/
  raw/bci_competition_iv_2a/        # Raw .mat files (not tracked)
  processed/
    bci_competition_iv_2a/          # 250 Hz preprocessed .npy files
    bci_competition_iv_2a_128/      # 128 Hz / ERS preprocessed .npy files
configs/
  data_splits_TE.json               # 250 Hz split configuration
  data_splits_128.json              # 128 Hz split configuration
experiments/
  checkpoints/                      # Saved model weights (.pt)
  results/                          # Training/evaluation JSON outputs
src/
  train.py                          # Training entry point (128 Hz / ERS pipeline)
  evaluate.py                       # Evaluation entry point
  data/
    preprocess.py                   # 250 Hz preprocessing pipeline
    alternative_preprocess.py       # 128 Hz / ERS preprocessing pipeline
    splits.py                       # Split generator → data_splits_TE.json
    alternative_splits.py           # Split generator → data_splits_128.json
    dataloader.py                   # BCIDataLoader, Normalizer, TrialNormalizer
  models/
    eegnet.py                       # EEGNet-8,2 (128 Hz)
    cnn_lstm.py                     # CNN + LSTM (128 Hz)
    cnn_gru.py                      # CNN + Bidirectional GRU (128 Hz)
    lstm.py                         # Pure Bidirectional LSTM (128 Hz)
    transformer.py                  # EEGNet front-end + Transformer (128 Hz)
    tcn.py                          # Temporal Convolutional Network (250 Hz)
run_subject_dependent.py            # Batch runner: all 9 subjects
run_loso.py                         # Batch runner: all 36 LOSO folds with ETA timer
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the raw BCI Competition IV Dataset 2a `.mat` files in `data/raw/bci_competition_iv_2a/`.

---

## Preprocessing

There are two independent preprocessing pipelines. They produce different data and cannot be mixed. The 128 Hz / ERS pipeline is the primary one used by `src/train.py`.

### Pipeline 1 — 250 Hz (original)

```bash
python src/data/preprocess.py      # → data/processed/bci_competition_iv_2a/
python src/data/splits.py          # → configs/data_splits_TE.json
```

What `preprocess.py` does per subject/session:
1. Load raw `.mat`, aggregate all 9 runs
2. Scale ×1e6 (raw units → µV)
3. Zero-phase Butterworth bandpass 4–40 Hz, order 5 (`sosfiltfilt`)
4. Epoch window 2–6 s after trial onset → 1000 samples at 250 Hz
5. Save `{subject}{T|E}_X.npy` `(n, 22, 1000)`, `_y.npy`, `_run.npy`

What `splits.py` produces (`configs/data_splits_TE.json`):
- **Subject-dependent:** random 70/15 train/val index split of the T session per subject (seeded per subject); E session → test.
- **LOSO:** 36 folds (9 test subjects × 4 random repetitions). Keys like `A01_rep0`. Per fold: 5 train subjects (T session) / 3 val subjects (T session) / 1 test subject (E session).

### Pipeline 2 — 128 Hz / ERS (primary)

```bash
python src/data/alternative_preprocess.py   # → data/processed/bci_competition_iv_2a_128/
python src/data/alternative_splits.py       # → configs/data_splits_128.json
```

What `alternative_preprocess.py` does per subject/session:
1. Load raw `.mat`, aggregate all 9 runs
2. Polyphase resample 250 → 128 Hz (all 25 channels)
3. Scale ×1e6 (→ µV)
4. Causal Butterworth bandpass 4–38 Hz, order 3, single-pass (`sosfilt`)
5. Exponential Running Standardisation (ERS) on the continuous signal (α = 1e-3, ε = 1e-4, initialised on first 1000 samples)
6. Epoch window 2.5–4.5 s after trial onset → 256 samples at 128 Hz
7. Drop artifact-flagged trials
8. Save `{subject}{T|E}_X.npy` `(n, 22, 256)`, `_y.npy`, `_run.npy`

ERS normalises the *continuous* signal before epoching with a causal running mean/variance. Because data is already normalised, **do not apply `Normalizer` or `TrialNormalizer` when training on this data** — `train.py` skips normalisation for this reason.

What `alternative_splits.py` produces (`configs/data_splits_128.json`):
- **Subject-dependent:** 80/20 stratified random train/val split of the T session per subject (seed = subject index 1–9); E session → test.
- **LOSO:** 36 folds with identical fold assignments to `data_splits_TE.json` (same seed=42). Keys like `A01_rep0`.

---

## Architectures

All models take `(batch, 22, 256)` input (22 EEG channels, 256 time samples at 128 Hz) and output `(batch, 4)` class logits, except TCN which targets the 250 Hz pipeline.

### EEGNet (`eegnet`)

Compact depthwise-separable CNN from Lawhern et al. (2018). Minimal parameter count (~2K), strong regularisation via max-norm constraints and dropout.

```
Block 1: Temporal Conv2D (1×64, same) → BatchNorm
          DepthwiseConv2D (22×1, valid, D=2) → BatchNorm → ELU → AvgPool (×4)
Block 2: SeparableConv2D (1×16, same) → BatchNorm → ELU → AvgPool (×8)
Flatten → Dropout → Linear(n_classes)
```

- Temporal kernel = `FS//2 = 64` (≥ 2 Hz frequency resolution)
- F1=8, D=2, F2=16
- DepthwiseConv max-norm = 1.0; classifier max-norm = 0.25
- `~2K parameters`
- `dropout_rate`: 0.5 within-subject, 0.25 cross-subject

### CNN+LSTM (`cnn_lstm`)

EEGNet-style CNN front-end feeding a single-layer LSTM. Learns frequency/spatial features then models long-range temporal dependencies recurrently.

```
Temporal Conv → Depthwise Spatial Conv → SeparableConv → AvgPool (×4, ×8)
→ Reshape to sequence (B, T', F2)
→ LSTM (hidden=128, 1 layer)
→ Last hidden state → Linear(n_classes)
```

- Temporal kernel = `sfreq//2 = 64` at 128 Hz
- `track_running_stats=False` in all BatchNorm layers (correct for cross-subject eval)
- `dropout_rate` forwarded from CLI

### CNN+GRU (`cnn_gru`)

EEGNet-style CNN front-end feeding a bidirectional GRU with temporal attention pooling. Recommended for LOSO.

```
Temporal Conv (×64) → Depthwise Conv (×22) → SeparableConv (×16)
→ AvgPool ×4 → AvgPool ×8   (T=256 → T'=8)
→ BiGRU (hidden=48, 2 layers)
→ Temporal attention pooling (weighted sum over T'=8 hidden states)
→ Linear(n_classes)
```

- `temporal_kernel=64`, `sep_kernel=16`
- `track_running_stats=False` in BatchNorm
- `dropout_rate` forwarded from CLI
- Grid-searched defaults: `lr=0.001`, `weight_decay=5e-4`, `dropout=0.4`, `batch_size=16`

### LSTM (`lstm`)

Pure bidirectional LSTM with no CNN front-end. Uses learnable depthwise temporal subsampling + LayerNorm throughout (no BatchNorm running stats). Supports an optional adversarial DANN head for LOSO domain adaptation.

```
Input (B, 22, 256)
→ AvgPool1d (kernel=4, stride=4)    → (B, 22, 64)   [non-learnable]
→ Permute to (B, 64, 22)
→ Linear(22 → 24) + LayerNorm + ELU  → (B, 64, 24)
→ Dropout(0.2)
→ BiLSTM (hidden=48, 1 layer)        → (B, 64, 96)
→ Dropout(0.4) + LayerNorm
→ Temporal attention pooling (attn_dim=24) → (B, 96)
→ Dropout(0.5) → Linear(96, n_classes) + MaxNorm(0.25)
```

- `~15K parameters`
- LayerNorm everywhere — safe for cross-subject eval (no accumulated running stats)
- LOSO mode: optionally adds DANN adversarial subject head (GRL, ×2 spatial expansion)

### Transformer (`transformer`)

EEGNet-8,2 CNN front-end feeding a temporal Transformer encoder. The CNN reduces the 256-sample input to 8 time steps; the Transformer attends over those 8 steps.

```
EEGNet Block 1 + Block 2 (identical to EEGNet-8,2, 128 Hz)
→ (B, F2=16, 1, T'=8) → reshape (B, 8, 16)
→ Linear(16 → d_model=32) + positional encoding
→ Transformer encoder (n_layers=2, n_heads=4, ff_dim=64)
→ Mean pooling over T'=8 → Linear(32, n_classes)
```

- `~12K parameters` (~2K CNN + ~10K transformer)
- `get_temporal_attention()` → `(B, heads, 8, 8)` for visualisation

### TCN (`tcn`)

Temporal Convolutional Network based on EEG-TCNet (Ingolfsson et al., 2020). Stacked dilated causal convolutions for multi-timescale temporal modelling without recurrence.

```
EEGNet front-end (→ compact feature sequence)
→ TCN blocks (dilated causal conv, dilation 1, 2, 4, 8, ...)
→ Classifier Linear(n_classes)
```

- Input: `(batch, 22, 1000)` — designed for 250 Hz / 1000-sample data
- Each TCN block: two dilated causal convs + BatchNorm + ELU + Dropout + residual

---

## Training

`src/train.py` uses the 128 Hz / ERS pipeline by default.

### CLI reference

```bash
python -m src.train --model <model> --mode <mode> [options]
```

**Required arguments:**

| Argument | Values | Description |
|---|---|---|
| `--model` | `eegnet`, `cnn_lstm`, `cnn_gru`, `lstm`, `transformer`, `tcn` | Architecture |
| `--mode` | `subject_dependent`, `loso` | Evaluation protocol |
| `--subject` | `A01`–`A09` | Required for `subject_dependent` |
| `--fold` | `A01_rep0`–`A09_rep3` | Required for `loso` |

**Hyperparameters:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `300` | Maximum training epochs |
| `--lr` | `0.001` | Adam learning rate |
| `--batch_size` | `32` | Mini-batch size |
| `--weight_decay` | `1e-4` | Adam L2 regularisation |
| `--dropout` | `0.6` | Dropout rate passed to model |
| `--min_epoch` | `100` | Earliest epoch at which early stopping can trigger |
| `--aug_shift` | `64` | Max time-shift for sliding-window augmentation (samples) |
| `--label_smoothing` | `0.1` | CrossEntropyLoss label smoothing factor |
| `--sign_flip_p` | `0.0` | Probability of negating trial amplitude per sample (0 = off) |
| `--euclidean_align` | `False` | Apply Euclidean alignment before training (LOSO only) |
| `--num_workers` | `0` | DataLoader worker processes |
| `--data_path` | `data/processed/bci_competition_iv_2a_128/` | Processed data directory |
| `--split_config` | `configs/data_splits_128.json` | Split configuration file |

**Training loop details:**
- Optimizer: Adam
- LR schedule: CosineAnnealingLR (`T_max=epochs`, `eta_min=1e-5`)
- Loss: CrossEntropyLoss with label smoothing
- Gradient clipping: global L2 norm ≤ 1.0 per step
- Max-norm constraints applied after each optimiser step (`apply_max_norm_()`)
- Early stopping: patience = 40 epochs, only active after `--min_epoch`
- Checkpoint criterion: 5-epoch smoothed validation accuracy (saves on improvement)
- Data augmentation (subject_dependent mode only):
  - Time-shift: random crop of `aug_shift` samples from trial start
  - Amplitude scaling: multiply by `U(0.8, 1.2)` per batch
  - Sign flip: negate entire trial with probability `sign_flip_p`

**Outputs:**
- Checkpoint: `experiments/checkpoints/{model}_{subject|fold}_{mode}_128_best.pt`
- Training history: `experiments/results/{model}_{subject|fold}_{mode}_128.json`

### Single-subject example

```bash
# Subject-dependent — one subject
python -m src.train --model eegnet --mode subject_dependent --subject A01 \
    --epochs 300 --lr 0.001 --batch_size 32

# LOSO — one fold
python -m src.train --model cnn_gru --mode loso --fold A01_rep0 \
    --epochs 300 --lr 0.001 --weight_decay 5e-4 --batch_size 16 --dropout 0.4
```

### Batch runners

```bash
# Subject-dependent — all 9 subjects sequentially
python run_subject_dependent.py --model eegnet --epochs 300 --lr 0.001

# LOSO — all 36 folds (9 subjects × 4 reps) with ETA timer
python run_loso.py --model cnn_gru --epochs 300 --lr 0.001 \
    --weight_decay 5e-4 --batch_size 16 --dropout 0.4

# LOSO — fast run (1 rep = 9 folds)
python run_loso.py --model lstm --reps 1 --epochs 300 --euclidean_align
```

Both runners accept all the same hyperparameter flags as `src.train`.

---

## Evaluation

```bash
python -m src.evaluate --model <model> --mode <mode> [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Model name (must match training) |
| `--mode` | required | `subject_dependent` or `loso` |
| `--reps` | `4` | Number of LOSO repetitions per subject to evaluate |
| `--euclidean_align` | `False` | Apply Euclidean alignment to test data (LOSO only) |
| `--data_path` | `data/processed/bci_competition_iv_2a_128/` | |
| `--split_config` | `configs/data_splits_128.json` | |

**Subject-dependent:** loads one checkpoint per subject, reports train/val/test accuracy per subject plus mean ± std.

**LOSO:** loads all matching fold checkpoints, ensembles softmax probabilities across all repetitions for each test subject, then reports per-subject ensemble accuracy and overall mean ± std.

Results are saved to `experiments/results/{model}_{mode}_128_eval_{YYYYMMDD_HHMMSS}.json` (timestamped to prevent overwrites). Each file includes a `training_config` block with the hyperparameters used during training.

```bash
# Subject-dependent
python -m src.evaluate --model eegnet --mode subject_dependent

# LOSO
python -m src.evaluate --model cnn_gru --mode loso
python -m src.evaluate --model lstm --mode loso --euclidean_align
```

---

## BCIDataLoader API

```python
from src.data.dataloader import BCIDataLoader, Normalizer

# Subject-dependent
loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    split='train',   # 'train' | 'val' | 'test'
    batch_size=32,
    shuffle=True,
)
for X_batch, y_batch, subj_batch in loader:
    # X_batch:    (batch, 22, T)  float32
    # y_batch:    (batch,)        long  (0=Left, 1=Right, 2=Feet, 3=Tongue)
    # subj_batch: (batch,)        long  (0=A01 … 8=A09; -1 in subject_dependent mode)
    ...

# LOSO
loader = BCIDataLoader(mode='loso', fold='A01_rep0', split='train', batch_size=32)

# Normalizer (250 Hz pipeline only — skip for 128 Hz / ERS data)
norm = Normalizer()
norm.fit(train_loader.dataset.X)
norm.apply_(train_loader.dataset)
val_loader  = BCIDataLoader(..., split='val',  normalizer=norm)
test_loader = BCIDataLoader(..., split='test', normalizer=norm)
```

Optional constructor arguments: `data_path`, `split_config`, `num_workers`, `transform`, `normalizer`.

---

## References

- Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5).
- Ingolfsson et al. (2020). EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain–Machine Interfaces. *arXiv:2006.00622*.
- BCI Competition IV Dataset 2a: http://www.bbci.de/competition/iv/
