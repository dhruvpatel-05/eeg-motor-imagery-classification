"""
examples/data_loading.py — Demonstrates how to use BCIDataLoader.

Run from the project root:
    python examples/data_loading.py
"""

import torch
from src.data.dataloader import BCIDataLoader

# ---------------------------------------------------------------------------
# 1. Subject-dependent: load one subject, one split
# ---------------------------------------------------------------------------

print("=== Subject-dependent ===")

train_loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    split='train',
    batch_size=32,
    shuffle=True,
)

val_loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    split='val',
    batch_size=32,
)

test_loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    split='test',
    batch_size=32,
)

print(f"  Train batches : {len(train_loader)}  (~{len(train_loader.dataset)} trials)")
print(f"  Val batches   : {len(val_loader)}   (~{len(val_loader.dataset)} trials)")
print(f"  Test batches  : {len(test_loader)}   (~{len(test_loader.dataset)} trials)")

X_batch, y_batch, _ = next(iter(train_loader))
print(f"  Batch X shape : {tuple(X_batch.shape)}  (batch, channels, time)")
print(f"  Batch y shape : {tuple(y_batch.shape)}   labels 0–3")
print()

# ---------------------------------------------------------------------------
# 2. LOSO: fold 0 → A01 is test subject
# ---------------------------------------------------------------------------

print("=== LOSO (fold 0) ===")

loso_train = BCIDataLoader(mode='loso', fold='A01_rep0', split='train', batch_size=32, shuffle=True)
loso_val   = BCIDataLoader(mode='loso', fold='A01_rep0', split='val',   batch_size=32)
loso_test  = BCIDataLoader(mode='loso', fold='A01_rep0', split='test',  batch_size=32)

print(f"  Train batches : {len(loso_train)}  (~{len(loso_train.dataset)} trials)")
print(f"  Val batches   : {len(loso_val)}   (~{len(loso_val.dataset)} trials)")
print(f"  Test batches  : {len(loso_test)}   (~{len(loso_test.dataset)} trials)")

X_batch, y_batch, _ = next(iter(loso_train))
print(f"  Batch X shape : {tuple(X_batch.shape)}")
print(f"  Batch y shape : {tuple(y_batch.shape)}")
print()

# ---------------------------------------------------------------------------
# 3. Optional transform (e.g. normalization)
# ---------------------------------------------------------------------------

print("=== With transform ===")

def normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-score each channel independently."""
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std

normed_loader = BCIDataLoader(
    mode='subject_dependent',
    subject='A01',
    split='train',
    batch_size=32,
    shuffle=True,
    transform=normalize,
)

X_batch, _, _ = next(iter(normed_loader))
print(f"  Post-norm mean: {X_batch.mean().item():.4f}  (should be ~0)")
print(f"  Post-norm std : {X_batch.std().item():.4f}   (should be ~1)")
