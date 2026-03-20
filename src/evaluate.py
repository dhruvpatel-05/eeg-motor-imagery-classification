"""
Evaluation script for BCI Competition IV 2a — 128 Hz / ERS pipeline.

  - Checkpoint suffix:  …_subject_dependent_128_best.pt
  - Default data path:  data/processed/bci_competition_iv_2a_128/
  - Default split config: configs/data_splits_128.json
  - Normalization skipped: data is already ERS-normalised

"""

import argparse
import datetime
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from src.train import get_model, apply_alignment
from src.data.dataloader import BCIDataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS  = [f"A{i:02d}" for i in range(1, 10)]
LOSO_REPS = 4

_ROOT             = os.path.join(os.path.dirname(__file__), "..")
_DEFAULT_DATA_PATH = os.path.join(_ROOT, "data", "processed", "bci_competition_iv_2a_128")
_DEFAULT_SPLIT_CFG = os.path.join(_ROOT, "configs", "data_splits_128.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, checkpoint_path: str, device: torch.device,
               mode: str = "subject_dependent") -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run train_128.py first for this model/subject."
        )
    model = get_model(model_name, mode=mode)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_ensemble(
    models: list,
    loader: BCIDataLoader,
    device: torch.device,
) -> float:
    """Average softmax probabilities across a list of models, then argmax."""
    all_probs: list[torch.Tensor] = []
    y_true:    list[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch = X_batch.float().to(device)
            batch_probs = []
            for model in models:
                out    = model(X_batch, alpha=0.0) if getattr(model, "adversarial", False) \
                         else model(X_batch)
                logits = out[0] if isinstance(out, tuple) else out
                batch_probs.append(F.softmax(logits, dim=-1))
            ensemble = torch.stack(batch_probs).mean(dim=0)   # (B, n_classes)
            all_probs.append(ensemble.cpu())
            y_true.append(y_batch.cpu())

    probs  = torch.cat(all_probs)
    labels = torch.cat(y_true)
    return (probs.argmax(dim=-1) == labels).float().mean().item()


def evaluate_one(model: torch.nn.Module, loader: BCIDataLoader, device: torch.device) -> float:
    correct = 0
    total   = 0
    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            out      = model(X_batch, alpha=0.0) if getattr(model, "adversarial", False) \
                       else model(X_batch)
            logits   = out[0] if isinstance(out, tuple) else out
            preds    = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total


# ---------------------------------------------------------------------------
# Subject-dependent evaluation
# ---------------------------------------------------------------------------

def evaluate_subject_dependent(
    model_name:  str,
    device:      torch.device,
    data_path:   str,
    split_config: str,
) -> dict:
    results = {}

    print(f"\nSubject-dependent evaluation (128 Hz): {model_name}")
    print(f"{'Subject':>9}  {'Train Acc':>9}  {'Val Acc':>7}  {'Test Acc':>8}")
    print("-" * 42)

    loader_kwargs = dict(
        mode="subject_dependent",
        batch_size=64,
        shuffle=False,
        data_path=data_path,
        split_config=split_config,
    )

    for subject in SUBJECTS:
        checkpoint_path = os.path.join(
            "experiments", "checkpoints",
            f"{model_name}_{subject}_subject_dependent_128_best.pt",
        )

        try:
            model = load_model(model_name, checkpoint_path, device, mode="subject_dependent")
        except FileNotFoundError as e:
            print(f"{subject:>9}  MISSING CHECKPOINT")
            print(f"           {e}")
            continue

        # No normalizer — ERS already applied in alternative_preprocess.py
        train_loader = BCIDataLoader(subject=subject, split="train", **loader_kwargs)
        val_loader   = BCIDataLoader(subject=subject, split="val",   **loader_kwargs)
        test_loader  = BCIDataLoader(subject=subject, split="test",  **loader_kwargs)

        train_acc = evaluate_one(model, train_loader, device)
        val_acc   = evaluate_one(model, val_loader,   device)
        test_acc  = evaluate_one(model, test_loader,  device)

        results[subject] = {
            "train_acc": train_acc,
            "val_acc":   val_acc,
            "test_acc":  test_acc,
        }
        print(f"{subject:>9} {train_acc:>8.2%}  {val_acc:>6.2%}  {test_acc:>7.2%}")

    if results:
        train_accs = [v["train_acc"] for v in results.values()]
        val_accs   = [v["val_acc"]   for v in results.values()]
        test_accs  = [v["test_acc"]  for v in results.values()]
        results["mean"] = {
            "train_acc": float(np.mean(train_accs)),
            "val_acc":   float(np.mean(val_accs)),
            "test_acc":  float(np.mean(test_accs)),
        }
        results["std"] = {
            "train_acc": float(np.std(train_accs)),
            "val_acc":   float(np.std(val_accs)),
            "test_acc":  float(np.std(test_accs)),
        }
        print("-" * 42)
        print(f"{'Mean':>9} {results['mean']['train_acc']:>8.2%}  "
              f"{results['mean']['val_acc']:>6.2%}  {results['mean']['test_acc']:>7.2%}")
        print(f"{'Std':>9} {results['std']['train_acc']:>8.2%}  "
              f"{results['std']['val_acc']:>6.2%}  {results['std']['test_acc']:>7.2%}")

    return results


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def evaluate_loso(
    model_name:      str,
    device:          torch.device,
    data_path:       str,
    split_config:    str,
    reps:            int  = LOSO_REPS,
    euclidean_align: bool = False,
) -> dict:
    with open(split_config) as f:
        config = json.load(f)

    fold_keys = []
    for key in config["loso"].keys():
        match = re.search(r"_rep(\d+)$", key)
        if match and int(match.group(1)) < reps:
            fold_keys.append(key)
    fold_keys.sort()

    results = {}

    print(f"\nLOSO evaluation (128 Hz): {model_name}")
    print(f"Using {reps} permutations per subject ({len(fold_keys)} folds total)")
    print(f"{'Fold':>12}  {'Test Acc':>8}")
    print("-" * 24)

    loader_kwargs = dict(
        mode="loso",
        batch_size=64,
        shuffle=False,
        data_path=data_path,
        split_config=split_config,
    )

    for fold in fold_keys:
        checkpoint_path = os.path.join(
            "experiments", "checkpoints",
            f"{model_name}_{fold}_loso_128_best.pt",
        )

        try:
            model = load_model(model_name, checkpoint_path, device, mode="loso")
        except FileNotFoundError as e:
            print(f"{fold:>12}  MISSING CHECKPOINT")
            print(f"               {e}")
            continue

        # No normalizer — ERS already applied in alternative_preprocess.py
        test_loader = BCIDataLoader(fold=fold, split="test", **loader_kwargs)
        ea_path = checkpoint_path.replace("_best.pt", "_ea_matrix.npy")
        if os.path.exists(ea_path):
            apply_alignment(test_loader.dataset, np.load(ea_path))
        elif euclidean_align:
            print(f"  WARNING: EA matrix not found at {ea_path}, skipping alignment for {fold}")

        acc = evaluate_one(model, test_loader, device)
        results[fold] = acc
        print(f"{fold:>12} {acc:>7.2%}")

    if results:
        accs = list(results.values())
        results["mean"] = float(np.mean(accs))
        results["std"]  = float(np.std(accs))
        print("-" * 24)
        print(f"{'Mean':>12} {results['mean']:>7.2%}")
        print(f"{'Std':>12} {results['std']:>7.2%}")

    # -- Per-subject ensemble: average softmax across all reps --------------
    subject_folds: dict[str, list[str]] = {}
    for fold in fold_keys:
        subj = fold.split("_rep")[0]
        subject_folds.setdefault(subj, []).append(fold)

    ensemble_results: dict[str, float] = {}

    print(f"\n{'Subject':>9}  {'Ensemble Acc':>12}  {'# reps':>6}")
    print("-" * 34)

    for subj, folds in sorted(subject_folds.items()):
        models_for_subj = []
        for fold in folds:
            ckpt = os.path.join(
                "experiments", "checkpoints",
                f"{model_name}_{fold}_loso_128_best.pt",
            )
            try:
                models_for_subj.append(load_model(model_name, ckpt, device, mode="loso"))
            except FileNotFoundError:
                pass

        if not models_for_subj:
            continue

        # All reps share the same test subject — use the first fold's test loader.
        # EA: average the R matrices across all available reps for this subject,
        # then apply the mean reference to the test data.
        test_loader = BCIDataLoader(fold=folds[0], split="test", **loader_kwargs)
        Rs = []
        for fold in folds:
            ckpt = os.path.join(
                "experiments", "checkpoints",
                f"{model_name}_{fold}_loso_128_best.pt",
            )
            ea_path = ckpt.replace("_best.pt", "_ea_matrix.npy")
            if os.path.exists(ea_path):
                Rs.append(np.load(ea_path))
        if Rs:
            apply_alignment(test_loader.dataset, np.mean(Rs, axis=0))
        elif euclidean_align:
            print(f"  WARNING: no EA matrices found for {subj}, skipping alignment")
        acc = evaluate_ensemble(models_for_subj, test_loader, device)
        ensemble_results[subj] = acc
        print(f"{subj:>9} {acc:>11.2%}  {len(models_for_subj):>6}")

    if ensemble_results:
        ens_accs = list(ensemble_results.values())
        ensemble_results["mean"] = float(np.mean(ens_accs))
        ensemble_results["std"]  = float(np.std(ens_accs))
        print("-" * 34)
        print(f"{'Mean':>9} {ensemble_results['mean']:>11.2%}")
        print(f"{'Std':>9} {ensemble_results['std']:>11.2%}")
        results["ensemble"] = ensemble_results

    return results


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results: dict, model_name: str, mode: str) -> None:
    out_dir = os.path.join("experiments", "results")
    os.makedirs(out_dir, exist_ok=True)

    # Pull training hyperparams from the first available training results JSON.
    results_dir = out_dir
    training_config = None
    if mode == "subject_dependent":
        candidates = [
            os.path.join(results_dir, f"{model_name}_{s}_subject_dependent_128.json")
            for s in SUBJECTS
        ]
    else:
        candidates = [
            os.path.join(results_dir, f"{model_name}_{s}_rep0_loso_128.json")
            for s in SUBJECTS
        ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                training_config = json.load(f).get("config")
            break

    payload = {"training_config": training_config, **results}

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{model_name}_{mode}_128_eval_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models trained with the 128 Hz / ERS pipeline"
    )

    parser.add_argument("--data_path",    type=str, default=_DEFAULT_DATA_PATH)
    parser.add_argument("--split_config", type=str, default=_DEFAULT_SPLIT_CFG)

    parser.add_argument("--model", type=str, required=True,
                        choices=["eegnet", "cnn_lstm", "tcn", "lstm", "cnn_gru", "transformer"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["subject_dependent", "loso"])
    parser.add_argument("--reps", type=int, default=LOSO_REPS,
                        help="Number of LOSO repetitions per subject to evaluate (default 4)")
    parser.add_argument("--euclidean_align", action="store_true", default=False,
                        help="Apply Euclidean alignment per subject to test data (LOSO only)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {args.data_path}")
    print(f"Config: {args.split_config}")

    if args.mode == "subject_dependent":
        results = evaluate_subject_dependent(
            args.model, device, args.data_path, args.split_config
        )
    else:
        results = evaluate_loso(
            args.model, device, args.data_path, args.split_config,
            reps=args.reps, euclidean_align=args.euclidean_align,
        )

    save_results(results, args.model, args.mode)


if __name__ == "__main__":
    main()
