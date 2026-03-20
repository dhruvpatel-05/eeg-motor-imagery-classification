"""
BCIDataLoader — PyTorch DataLoader for BCI Competition IV Dataset 2a.

## Overview

This module wraps the preprocessed .npy files and split config
(configs/data_splits.json) into standard PyTorch DataLoaders. Two evaluation
modes are supported:

  - **subject_dependent**: Train, validate, and test within a single subject
    using a 70/15/15 index split. All data comes from that subject's Training
    session (A##T).

  - **loso** (Leave-One-Subject-Out): Train on 7 subjects, validate on 1,
    test on 1. Both Training and Evaluation sessions are concatenated per
    subject.

## Normalization

EEGNet (and all other architectures in this project) expect z-scored input.
Normalization is per-channel, computed on training data only and applied to
val/test. Use the `Normalizer` helper.

The Normalizer computes mean and std across (trials, timepoints) independently
for each of the 22 channels, then stores them so they can be reused for val/
test without any leakage.
"""

import json
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DEFAULT_DATA_PATH = os.path.join(_ROOT, "data", "processed", "bci_competition_iv_2a")
_DEFAULT_CONFIG    = os.path.join(_ROOT, "configs", "data_splits.json")


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    """Per-channel z-score normalizer fitted on training data.

    Stats are computed across the (trials, timepoints) axes independently for
    each of the 22 EEG channels, so each channel is normalized to zero mean and
    unit variance.

    Usage:
        norm = Normalizer()
        norm.fit(X_train_tensor)        # shape: (n, 22, 1000)
        norm.apply_(train_dataset)
        norm.apply_(val_dataset)
        norm.apply_(test_dataset)       # always use training stats

    After apply_() the dataset's X tensor is modified in-place.
    """

    def __init__(self):
        self.mean_: Optional[torch.Tensor] = None   # (1, 22, 1)
        self.std_:  Optional[torch.Tensor] = None   # (1, 22, 1)

    def fit(self, X: torch.Tensor) -> "Normalizer":
        """Compute mean and std from X (n_trials, n_channels, n_timepoints).

        Args:
            X: float32 tensor of shape (n, 22, 1000).

        Returns:
            self, for chaining.
        """
        # Average over trial and time axes; keep channel axis
        self.mean_ = X.mean(dim=(0, 2), keepdim=True)   # (1, 22, 1)
        self.std_  = X.std(dim=(0, 2), keepdim=True).clamp(min=1e-6)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply stored normalization to X. Does NOT modify in place."""
        if self.mean_ is None:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def apply_(self, dataset: "BCIDataset") -> None:
        """Normalize dataset.X in-place using stored stats.

        Typical use:
            norm.fit(train_ds.X)
            norm.apply_(train_ds)
            norm.apply_(val_ds)
            norm.apply_(test_ds)
        """
        if self.mean_ is None:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")
        dataset.X = self.transform(dataset.X)


# ---------------------------------------------------------------------------
# Internal Dataset
# ---------------------------------------------------------------------------
class TrialNormalizer:
    """Normalize each trial independently (zero mean, unit variance per channel).

    Unlike Normalizer, this requires no fitting — each trial is normalized
    using its own statistics. This removes inter-subject baseline differences,
    which is critical for LOSO generalization: the val/test subject's signals
    will be at the same scale as training signals regardless of who they are.

    Usage:
        norm = TrialNormalizer()
        norm.apply_(train_loader.dataset)
        norm.apply_(val_loader.dataset)
        norm.apply_(test_loader.dataset)
    """

    def apply_(self, dataset: "BCIDataset") -> None:
        X = dataset.X                                    # (n_trials, 22, 1000)
        mean = X.mean(dim=2, keepdim=True)               # (n_trials, 22, 1)
        std  = X.std(dim=2,  keepdim=True).clamp(min=1e-6)
        dataset.X = (X - mean) / std

class BCIDataset(Dataset):
    """Holds the NumPy arrays for a single split and exposes them as tensors.

    Labels are shifted from 1-indexed (1–4) to 0-indexed (0–3) so they work
    directly with PyTorch's CrossEntropyLoss.

    Each item is a 3-tuple ``(x, y, subject_id)`` where ``subject_id`` is an
    integer in 0–8 (A01→0 … A09→8). It is -1 when subject identity is not
    tracked (e.g. subject-dependent mode).
    """

    def __init__(
        self,
        X:           np.ndarray,
        y:           np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        transform:   Optional[Callable]   = None,
    ):
        self.X         = torch.tensor(X, dtype=torch.float32)
        self.y         = torch.tensor(y - 1, dtype=torch.long)   # 1–4 → 0–3
        if subject_ids is not None:
            self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)
        else:
            self.subject_ids = torch.full((len(y),), -1, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx], self.subject_ids[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npy(data_path: str, subject: str, session: str):
    """Load X and y arrays for one subject/session pair."""
    stem = f"{subject}{session}"
    X = np.load(os.path.join(data_path, f"{stem}_X.npy"))
    y = np.load(os.path.join(data_path, f"{stem}_y.npy"))
    return X, y


def _concat_sessions(data_path: str, subject: str, sessions: list[str]):
    """Concatenate multiple sessions for one subject.

    Sessions that are missing from disk are skipped gracefully (e.g. if an
    Evaluation file hasn't been downloaded yet). If no sessions are found at
    all, a FileNotFoundError is raised rather than silently returning empty
    data.
    """
    Xs, ys = [], []
    for sess in sessions:
        x_path = os.path.join(data_path, f"{subject}{sess}_X.npy")
        if not os.path.exists(x_path):
            # Warn but continue — may have only T or only E session
            print(f"  [dataloader] WARNING: {subject}{sess}_X.npy not found, skipping.")
            continue
        X, y = _load_npy(data_path, subject, sess)
        Xs.append(X)
        ys.append(y)

    if not Xs:
        raise FileNotFoundError(
            f"No session files found for subject '{subject}' in {data_path}. "
            f"Looked for sessions: {sessions}"
        )

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BCIDataLoader:
    """PyTorch DataLoader for BCI Competition IV Dataset 2a.

    Supports two modes:
      - 'subject_dependent': single subject, index-based 70/15/15 split.
      - 'loso': cross-subject, 7 train / 1 val / 1 test subject per fold.

    Normalization workflow (recommended):

        train_loader = BCIDataLoader(mode=..., split='train', ...)
        norm = Normalizer()
        norm.fit(train_loader.dataset.X)
        norm.apply_(train_loader.dataset)

        val_loader  = BCIDataLoader(mode=..., split='val',  ...)
        test_loader = BCIDataLoader(mode=..., split='test', ...)
        norm.apply_(val_loader.dataset)
        norm.apply_(test_loader.dataset)

    See module docstring for full usage examples.
    """

    def __init__(
        self,
        mode:  str,
        split: str,
        *,
        # subject_dependent args
        subject: Optional[str] = None,
        # loso args
        fold: Optional[str] = None,
        # shared / optional
        data_path:    str = _DEFAULT_DATA_PATH,
        split_config: str = _DEFAULT_CONFIG,
        batch_size:   int = 32,
        shuffle:      bool = False,
        num_workers:  int = 0,
        transform:    Optional[Callable] = None,
        normalizer:   Optional[Normalizer] = None,
    ):
        if mode not in ("subject_dependent", "loso"):
            raise ValueError(f"mode must be 'subject_dependent' or 'loso', got '{mode}'")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        with open(split_config) as f:
            config = json.load(f)

        if mode == "subject_dependent":
            X, y, subject_ids = self._build_subject_dependent(config, data_path, subject, split)
        else:
            X, y, subject_ids = self._build_loso(config, data_path, fold, split)

        dataset = BCIDataset(X, y, subject_ids=subject_ids, transform=transform)

        # Apply normalizer if provided (caller fitted it on training data)
        if normalizer is not None:
            normalizer.apply_(dataset)

        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # Mode-specific builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subject_dependent(config, data_path, subject, split):
        if subject is None:
            raise ValueError("'subject' is required for mode='subject_dependent'")

        sid = int(subject[1:]) - 1   # "A01" → 0, "A09" → 8

        subj_cfg = config["subject_dependent"][subject]
        if split == "test":
            test_session = subj_cfg.get("test_session", subj_cfg.get("session", "T"))
            X, y = _load_npy(data_path, subject, test_session)
        else:
            indices = subj_cfg[split]
            train_session = subj_cfg.get("train_session", subj_cfg.get("session", "T"))
            X, y = _load_npy(data_path, subject, train_session)
            X, y = X[indices], y[indices]

        subject_ids = np.full(len(y), sid, dtype=np.int64)
        return X, y, subject_ids

    @staticmethod
    def _build_loso(config, data_path, fold, split):
        if fold is None:
            raise ValueError("'fold' is required for mode='loso'")

        fold_cfg = config["loso"][fold]

        if split == "train":
            subjects = fold_cfg["train_subjects"]
            sessions = fold_cfg["train_sessions"]
        elif split == "val":
            subjects = fold_cfg["val_subjects"]
            sessions = fold_cfg["val_sessions"]
        else:  # test
            subjects = [fold_cfg["test_subject"]]
            sessions = fold_cfg["test_sessions"]

        Xs, ys, sids = [], [], []
        for subj in subjects:
            X, y = _concat_sessions(data_path, subj, sessions)
            sid  = int(subj[1:]) - 1   # "A01" → 0, "A09" → 8
            Xs.append(X)
            ys.append(y)
            sids.append(np.full(len(y), sid, dtype=np.int64))
        return (
            np.concatenate(Xs,   axis=0),
            np.concatenate(ys,   axis=0),
            np.concatenate(sids, axis=0),
        )

    # ------------------------------------------------------------------
    # DataLoader pass-through
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self) -> BCIDataset:
        return self._loader.dataset
