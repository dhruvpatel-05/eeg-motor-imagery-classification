"""
Create subject-dependent and LOSO data split configurations for the 128 Hz / ERS pipeline.

## Within-subject (subject_dependent)
80/20 stratified random train/val split of the T session per subject.
Seed is derived from the subject index (e.g. A01 → seed 1).
Test always uses the E session.

## LOSO
36 folds: 9 test subjects × 4 random repetitions.
Per fold: 5 train subjects (T session), 3 val subjects (T session), 1 test subject (E session).
Identical fold assignments to data_splits_TE.json (seed=42).

No sklearn dependency — uses numpy.random.RandomState only.
"""

import json
import os

import numpy as np


PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "processed",
    "bci_competition_iv_2a_128",
)
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]


def create_subject_dependent_splits() -> dict:
    """80/20 random train/val split within each subject's T session.

    - Seed is ``int(subject[1:])`` (1–9) for reproducibility per subject.
    - No sklearn: uses numpy.random.RandomState directly.
    - val indices  = first n//5  elements of the permutation
    - train indices = remaining elements

    Returns:
        dict mapping subject (e.g. 'A01') to split config dict with keys:
        'train', 'val', 'train_session', 'test_session'.
    """
    splits = {}
    for subject in SUBJECTS:
        x_path = os.path.join(PROCESSED_DIR, f"{subject}T_X.npy")
        if not os.path.exists(x_path):
            print(f"  WARNING: {x_path} not found — skipping {subject}")
            continue

        X = np.load(x_path, mmap_mode="r")
        n = X.shape[0]

        seed = int(subject[1:])          # A01 → 1, A09 → 9
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)

        n_val = n // 5
        val_indices = sorted(indices[:n_val].tolist())
        train_indices = sorted(indices[n_val:].tolist())

        splits[subject] = {
            "train": train_indices,
            "val":   val_indices,
            "train_session": "T",
            "test_session":  "E",
        }

    return splits


def create_loso_splits() -> dict:
    """36 folds: 9 test subjects × 4 random repetitions (5 train / 3 val / 1 test).

    Identical fold assignments to data_splits_TE.json — same seed=42 and same
    permutation logic — so LOSO results are directly comparable across pipelines.
    Train/val subjects use their T session; test subject uses the E session.

    Returns:
        dict mapping fold key (e.g. 'A01_rep0') to fold config dict.
    """
    splits = {}
    rng = np.random.RandomState(42)
    for test_subject in SUBJECTS:
        others = [s for s in SUBJECTS if s != test_subject]
        for rep in range(4):
            perm = rng.permutation(8)
            train_subjects = [others[i] for i in perm[:5]]
            val_subjects   = [others[i] for i in perm[5:]]
            fold_key = f"{test_subject}_rep{rep}"
            splits[fold_key] = {
                "train_subjects": train_subjects,
                "val_subjects":   val_subjects,
                "test_subject":   test_subject,
                "train_sessions": ["T"],
                "val_sessions":   ["T"],
                "test_sessions":  ["E"],
            }
    return splits


def main():
    config = {
        "subject_dependent": create_subject_dependent_splits(),
        "loso": create_loso_splits(),
    }

    os.makedirs(CONFIG_DIR, exist_ok=True)
    out_path = os.path.join(CONFIG_DIR, "data_splits_128.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    sd   = config["subject_dependent"]
    loso = config["loso"]
    print("Created 128 Hz data splits configuration")
    print(f"  Subject-dependent: {len(sd)} subjects (T train/val 80/20, E test)")
    for subj, split in sd.items():
        print(
            f"    {subj}: train={len(split['train'])}, "
            f"val={len(split['val'])}, test_session=E"
        )
    print(f"  LOSO: {len(loso)} folds (9 subjects × 4 reps, T train/val, E test)")
    print(f"  Saved to: {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
