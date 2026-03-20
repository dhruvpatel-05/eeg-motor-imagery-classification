# Data Loading Guide — BCI Competition IV Dataset 2a

## Overview

`BCIDataLoader` wraps the preprocessed `.npy` files and `configs/data_splits.json` into a standard PyTorch `DataLoader`. Two evaluation protocols are supported, both aligned with the dataset paper (desc_2a.pdf):

- **Subject-dependent**: 4-fold blockwise cross-validation within a single subject
- **LOSO**: 90-fold leave-one-subject-out (9 subjects × 10 random repetitions)

---

## Prerequisites

Run the preprocessing pipeline once before using the dataloader (in case your data is not up-to-date):

```bash
python src/data/preprocess.py   # produces _X.npy, _y.npy, _run.npy
python src/data/splits.py       # produces configs/data_splits.json
```

---

## Subject-Dependent (4-Fold Blockwise CV)

### Protocol

The T session contains 6 motor imagery runs (run indices 3–8), each with 48 trials. The first 4 MI runs (indices 3–6) are used for cross-validation, giving 4 blocks of 48 trials. Each fold assigns the blocks as follows:

| Fold | Train blocks      | Val block | Test block |
|------|-------------------|-----------|------------|
| 0    | runs 5, 6 (96)    | run 4 (48)| run 3 (48) |
| 1    | runs 3, 6 (96)    | run 5 (48)| run 4 (48) |
| 2    | runs 3, 4 (96)    | run 6 (48)| run 5 (48) |
| 3    | runs 4, 5 (96)    | run 3 (48)| run 6 (48) |

All data comes from the T session. The E session is not used in this mode.

### Usage

```python
from src.data.dataloader import BCIDataLoader

# Training split — fold 0, subject A01
train_loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    fold=0,
    split='train',
    batch_size=32,
    shuffle=True,
)

# Validation split — same fold
val_loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    fold=0,
    split='val',
    batch_size=32,
)

# Test split — same fold
test_loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    fold=0,
    split='test',
    batch_size=32,
)

for X_batch, y_batch in train_loader:
    # X_batch: (batch_size, 22, 256)  float32
    # y_batch: (batch_size,)           long  (0-indexed: 0=Left, 1=Right, 2=Feet, 3=Tongue)
    pass
```

### Full 4-Fold Loop

```python
from src.data.dataloader import BCIDataLoader

subject = 'A01'

for fold in range(4):
    train_loader = BCIDataLoader(
        mode='subject_dependent', subject=subject, fold=fold,
        split='train', batch_size=32, shuffle=True,
    )
    val_loader = BCIDataLoader(
        mode='subject_dependent', subject=subject, fold=fold,
        split='val', batch_size=32,
    )
    test_loader = BCIDataLoader(
        mode='subject_dependent', subject=subject, fold=fold,
        split='test', batch_size=32,
    )
    # ... train and evaluate
```

### All Subjects × All Folds

```python
SUBJECTS = [f'A{i:02d}' for i in range(1, 10)]

for subject in SUBJECTS:
    for fold in range(4):
        train_loader = BCIDataLoader(
            mode='subject_dependent', subject=subject, fold=fold,
            split='train', batch_size=32, shuffle=True,
        )
        # ...
```

---

## LOSO (Leave-One-Subject-Out, 90 Folds)

### Protocol

90 folds total: 9 test subjects × 10 random repetitions. For each fold:

| Split | Subjects      | Session | Count     |
|-------|---------------|---------|-----------|
| Train | 5 subjects    | T only  | ~1440 trials |
| Val   | 3 subjects    | T only  | ~864 trials  |
| Test  | 1 subject     | E only  | ~288 trials  |

The test subject's T session is **never used** for training or validation — only their E (evaluation/competition test) session is used for testing. This mirrors the competition protocol where labels for the E session were withheld.

Fold keys follow the pattern `{subject}_rep{0–9}`, e.g. `A01_rep0` through `A09_rep9`.

### Usage

```python
from src.data.dataloader import BCIDataLoader

# Single fold
train_loader = BCIDataLoader(
    mode='loso',
    fold_key='A01_rep0',
    split='train',
    batch_size=32,
    shuffle=True,
)

val_loader = BCIDataLoader(
    mode='loso',
    fold_key='A01_rep0',
    split='val',
    batch_size=32,
)

test_loader = BCIDataLoader(
    mode='loso',
    fold_key='A01_rep0',
    split='test',
    batch_size=32,
)
```

### All 90 Folds

```python
SUBJECTS = [f'A{i:02d}' for i in range(1, 10)]

for subject in SUBJECTS:
    for rep in range(10):
        fold_key = f'{subject}_rep{rep}'
        train_loader = BCIDataLoader(
            mode='loso', fold_key=fold_key,
            split='train', batch_size=32, shuffle=True,
        )
        val_loader   = BCIDataLoader(mode='loso', fold_key=fold_key, split='val',   batch_size=32)
        test_loader  = BCIDataLoader(mode='loso', fold_key=fold_key, split='test',  batch_size=32)
        # ... train and evaluate
```

### Aggregating Results Across Repetitions

Since each test subject appears in 10 folds (one per rep), averaging test accuracy across all 10 reps for a subject gives a more stable estimate than a single fold:

```python
import numpy as np

results = {f'A{i:02d}': [] for i in range(1, 10)}

for subject in SUBJECTS:
    for rep in range(10):
        fold_key = f'{subject}_rep{rep}'
        # ... train model, get test_acc
        results[subject].append(test_acc)

for subject, accs in results.items():
    print(f'{subject}: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
```

---

## Optional Arguments

All modes accept these additional keyword arguments:

```python
BCIDataLoader(
    ...,
    data_path='data/processed/bci_competition_iv_2a',  # override data directory
    split_config='configs/data_splits.json',            # override split config
    num_workers=0,                                      # DataLoader worker processes
    transform=None,                                     # callable applied to each X tensor
)
```

### Custom transform example

```python
import torch

def normalize(x):
    # x: (22, 256) float32
    return (x - x.mean()) / (x.std() + 1e-6)

loader = BCIDataLoader(
    mode='subject_dependent', subject='A01', fold=0,
    split='train', batch_size=32, transform=normalize,
)
```

---

## Output Tensor Shapes

| Tensor    | Shape                   | dtype   | Notes                              |
|-----------|-------------------------|---------|------------------------------------|
| `X_batch` | `(batch_size, 22, 256)` | float32 | EEG channels × time samples        |
| `y_batch` | `(batch_size,)`         | long    | 0-indexed class labels (0–3)       |

Label mapping after 0-indexing shift:

| Value | Class      |
|-------|------------|
| 0     | Left Hand  |
| 1     | Right Hand |
| 2     | Both Feet  |
| 3     | Tongue     |

---

## Accessing the Underlying Dataset

```python
loader = BCIDataLoader(mode='subject_dependent', subject='A01', fold=0, split='train', batch_size=32)

dataset = loader.dataset          # torch.utils.data.Dataset
print(len(dataset))               # number of trials in this split
X_all = dataset.X                 # (n_trials, 22, 256) float32 tensor
y_all = dataset.y                 # (n_trials,) long tensor
```
