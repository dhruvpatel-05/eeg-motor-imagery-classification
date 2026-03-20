"""
Training script for BCI Competition IV 2a — 128 Hz / ERS pipeline.

  - Default data directory: data/processed/bci_competition_iv_2a_128/
  - Default split config:   configs/data_splits_128.json
  - Normalization skipped:  data is already ERS-normalised on the continuous
    signal by alternative_preprocess.py; applying TrialNormalizer on top would
    double-normalise and degrade training signal.

"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataloader import BCIDataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DEFAULT_DATA_PATH  = os.path.join(_ROOT, "data", "processed", "bci_competition_iv_2a_128")
_DEFAULT_SPLIT_CFG  = os.path.join(_ROOT, "configs", "data_splits_128.json")


# ---------------------------------------------------------------------------
# Euclidean alignment
# ---------------------------------------------------------------------------

def compute_alignment_matrix(X_train: np.ndarray) -> np.ndarray:
    """Compute the Euclidean alignment matrix from combined training data.

    Args:
        X_train: (n_trials, C, T) — all training subjects concatenated.

    Returns:
        R: (C, C) whitening matrix — inv(cholesky(mean_cov)).
    """
    cov = np.mean([x @ x.T for x in X_train], axis=0) / X_train.shape[-1]
    cov += 1e-6 * np.eye(cov.shape[0])   # numerical regularisation
    return np.linalg.inv(np.linalg.cholesky(cov))


def apply_alignment(dataset, R: np.ndarray) -> None:
    """Apply a precomputed alignment matrix R to dataset.X in-place.

    Args:
        dataset: BCIDataset whose .X tensor will be modified.
        R:       (C, C) whitening matrix from compute_alignment_matrix().
    """
    X = dataset.X.numpy()
    dataset.X = torch.tensor(np.stack([R @ x for x in X]), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model registry (same as train.py)
# ---------------------------------------------------------------------------

def sliding_window_augment(
    X: torch.Tensor,
    y: torch.Tensor,
    max_shift: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random time-shift with zero padding — preserves input size for model compatibility."""
    if max_shift == 0:
        return X, y
    shift = torch.randint(0, max_shift + 1, (1,)).item()
    if shift == 0:
        return X, y
    pad = torch.zeros(*X.shape[:-1], shift, device=X.device, dtype=X.dtype)
    return torch.cat([pad, X[..., :-shift]], dim=-1), y


def get_model(model_name: str, n_channels: int = 22, n_classes: int = 4, dropout_rate: float = 0.5, mode: str = "subject_dependent") -> nn.Module:
    if model_name == "eegnet":
        from src.models.eegnet import EEGNet
        return EEGNet(n_channels=n_channels, n_classes=n_classes, n_timepoints=256)
    elif model_name == "cnn_lstm":
        from src.models.cnn_lstm import CNNLSTM
        return CNNLSTM(n_channels=n_channels, n_classes=n_classes,
                       n_timepoints=256, sfreq=128, dropout_rate=dropout_rate)
    elif model_name == "tcn":
        from src.models.tcn import TCN
        return TCN(n_channels=n_channels, n_classes=n_classes,
               n_timepoints=256, sfreq=128, track_running_stats=False)
    elif model_name == "lstm":
        from src.models.lstm import LSTM
        return LSTM(n_channels=n_channels, n_classes=n_classes, dropout_rate=dropout_rate, mode=mode)
    elif model_name == "cnn_gru":
        from src.models.cnn_gru import CNNGRU
        return CNNGRU(n_channels=n_channels, n_classes=n_classes,
                      cnn_dropout=dropout_rate, fc_dropout=dropout_rate)
    elif model_name == "transformer":
        from src.models.transformer import EEGTransformer
        return EEGTransformer(n_channels=n_channels, n_classes=n_classes, cnn_dropout=dropout_rate)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Options: eegnet, cnn_lstm, tcn, lstm, cnn_gru, transformer"
        )


# ---------------------------------------------------------------------------
# Training and validation loops (identical to train.py)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:         nn.Module,
    loader:        BCIDataLoader,
    optimizer:     torch.optim.Optimizer,
    criterion:     nn.Module,
    device:        torch.device,
    augment:       bool  = False,
    aug_shift:     int   = 64,
    sign_flip_p:   float = 0.0,
    alpha:         float = 0.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch, subj_batch in loader:
        X_batch    = X_batch.float().to(device)
        y_batch    = y_batch.long().to(device)
        subj_batch = subj_batch.long().to(device)

        if augment:
            X_batch, y_batch = sliding_window_augment(X_batch, y_batch, max_shift=aug_shift)
            scale = torch.empty(X_batch.size(0), 1, 1, device=X_batch.device).uniform_(0.8, 1.2)
            X_batch = X_batch * scale
            if sign_flip_p > 0.0:
                flip = torch.rand(X_batch.size(0), 1, 1, device=X_batch.device) < sign_flip_p
                X_batch = torch.where(flip, -X_batch, X_batch)

        optimizer.zero_grad()

        if getattr(model, "adversarial", False):
            logits, subject_logits = model(X_batch, alpha=alpha)
            task_loss    = criterion(logits, y_batch)
            subject_loss = F.cross_entropy(subject_logits, subj_batch)
            loss = task_loss + 0.1 * subject_loss
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if hasattr(model, "apply_max_norm_"):
            model.apply_max_norm_()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def validate(
    model:     nn.Module,
    loader:    BCIDataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            if getattr(model, "adversarial", False):
                logits, _ = model(X_batch, alpha=0.0)
            else:
                logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training loop (identical to train.py)
# ---------------------------------------------------------------------------

def train(
    model:           nn.Module,
    train_loader:    BCIDataLoader,
    val_loader:      BCIDataLoader,
    config:          dict,
    checkpoint_path: str,
    results_path:    str,
) -> dict:
    device = config["device"]
    model  = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-5
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "config":     config,
    }

    best_val_acc       = 0.0
    epochs_no_improve  = 0
    early_stop_patience = 40
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path),    exist_ok=True)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # DANN alpha schedule: ramps 0 → 1 over training (Ganin et al. 2016)
        p     = (epoch - 1) / config["epochs"]
        alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            augment=True,
            aug_shift=config["aug_shift"],
            sign_flip_p=config["sign_flip_p"],
            alpha=alpha,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.1f}s"
        )

        VAL_SMOOTH = 5
        if len(history["val_acc"]) >= VAL_SMOOTH:
            smoothed_val = sum(history["val_acc"][-VAL_SMOOTH:]) / VAL_SMOOTH
        else:
            smoothed_val = val_acc

        if smoothed_val > best_val_acc:
            best_val_acc = smoothed_val
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
            print(f"  ✓ Saved checkpoint (smoothed_val={smoothed_val:.2%})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience and epoch > config["min_epoch"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs)")
                break

    history["best_val_acc"] = best_val_acc
    print(f"\nBest val acc: {best_val_acc:.2%}")
    print(f"Checkpoint:   {checkpoint_path}")

    history["config"] = {
        k: str(v) if isinstance(v, torch.device) else v
        for k, v in config.items()
    }
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Results:      {results_path}\n")

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train on BCI Competition IV 2a (128 Hz / ERS pipeline)"
    )

    parser.add_argument("--data_path",    type=str, default=_DEFAULT_DATA_PATH)
    parser.add_argument("--split_config", type=str, default=_DEFAULT_SPLIT_CFG)

    parser.add_argument("--model", type=str, required=True,
                        choices=["eegnet", "cnn_lstm", "tcn", "lstm", "cnn_gru", "transformer"])
    parser.add_argument("--mode",  type=str, required=True,
                        choices=["subject_dependent", "loso"])

    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--fold",    type=str, default=None)

    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout",         type=float, default=0.6)
    parser.add_argument("--num_workers",     type=int,   default=0)
    parser.add_argument("--min_epoch",       type=int,   default=100)
    parser.add_argument("--aug_shift",       type=int,   default=64,
                        help="Max time-shift for sliding window augmentation (samples, default 64)")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--sign_flip_p",     type=float, default=0.0,
                        help="P(negate trial amplitude) per sample (0=off, 0.5 recommended)")
    parser.add_argument("--euclidean_align", action="store_true", default=False,
                        help="Apply Euclidean alignment per subject before training (LOSO only)")

    args = parser.parse_args()

    if args.mode == "subject_dependent" and args.subject is None:
        parser.error("--subject is required for subject_dependent mode")
    if args.mode == "loso" and args.fold is None:
        parser.error("--fold is required for loso mode")
    if args.euclidean_align and args.mode != "loso":
        parser.error("--euclidean_align is only supported with --mode loso")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {args.data_path}")
    print(f"Config: {args.split_config}")

    # ------------------------------------------------------------------
    # Build dataloaders — no normalizer applied (ERS done in preprocessing)
    # ------------------------------------------------------------------
    loader_kwargs = dict(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subject=args.subject,
        fold=args.fold,
        data_path=args.data_path,
        split_config=args.split_config,
    )
    train_loader = BCIDataLoader(**loader_kwargs, split="train", shuffle=True)
    val_loader   = BCIDataLoader(**loader_kwargs, split="val",   shuffle=False)

    if args.euclidean_align:
        R = compute_alignment_matrix(train_loader.dataset.X.numpy())
        apply_alignment(train_loader.dataset, R)
        apply_alignment(val_loader.dataset, R)
        print("Euclidean alignment applied (training reference, applied to train+val)")

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model    = get_model(args.model, dropout_rate=args.dropout, mode=args.mode)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    if args.mode == "subject_dependent":
        run_id = f"{args.model}_{args.subject}_subject_dependent_128"
    else:
        run_id = f"{args.model}_{args.fold}_loso_128"

    checkpoint_path = os.path.join("experiments", "checkpoints", f"{run_id}_best.pt")
    results_path    = os.path.join("experiments", "results",      f"{run_id}.json")

    config = {
        "model":        args.model,
        "mode":         args.mode,
        "subject":      args.subject,
        "fold":         args.fold,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "batch_size":   args.batch_size,
        "weight_decay": args.weight_decay,
        "device":       device,
        "data_path":    args.data_path,
        "split_config": args.split_config,
        "norm":            "none (ERS applied in preprocessing)",
        "min_epoch":       args.min_epoch,
        "aug_shift":       args.aug_shift,
        "label_smoothing":  args.label_smoothing,
        "sign_flip_p":      args.sign_flip_p,
        "euclidean_align":  args.euclidean_align,
    }

    if args.euclidean_align:
        ea_matrix_path = checkpoint_path.replace("_best.pt", "_ea_matrix.npy")
        np.save(ea_matrix_path, R)
        print(f"EA matrix saved: {ea_matrix_path}")

    train(model, train_loader, val_loader, config, checkpoint_path, results_path)


if __name__ == "__main__":
    main()
