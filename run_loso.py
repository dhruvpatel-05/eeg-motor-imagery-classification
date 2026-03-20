"""
Run LOSO training for 4 reps per subject (36 folds) using the 128 Hz / ERS pipeline.

Usage:
    python run_loso.py --model eegnet [--reps 4] [--epochs 300] \
        [--lr 0.001] [--weight_decay 5e-4] [--batch_size 8] [--dropout 0.4]
"""
import argparse
import json
import subprocess
import time
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="Model name (e.g. eegnet, cnn_lstm, tcn, lstm, cnn_gru, transformer)")
parser.add_argument("--mode", default="loso",
                    choices=["subject_dependent", "loso"],
                    help="subject_dependent or loso — controls architecture and training")
parser.add_argument("--reps",         type=int, default=4,    help="Number of repetitions per subject (1–4)")
parser.add_argument("--epochs",       default="300")
parser.add_argument("--lr",           default="0.001")
parser.add_argument("--weight_decay", default="5e-4")
parser.add_argument("--batch_size",   default="8")
parser.add_argument("--dropout",         default="0.4")
parser.add_argument("--min_epoch",       default="100")
parser.add_argument("--aug_shift",       default="64")
parser.add_argument("--label_smoothing", default="0.1")
parser.add_argument("--sign_flip_p",     default="0.0")
parser.add_argument("--num_workers",     default="0")
parser.add_argument("--euclidean_align", action="store_true", default=False,
                    help="Apply Euclidean alignment per subject before training")
args = parser.parse_args()

with open("configs/data_splits_128.json") as f:
    config = json.load(f)

fold_keys = [k for k in config["loso"] if int(k.split("_rep")[1]) < args.reps]
n = len(fold_keys)

subjects_order = []
folds_by_subject = defaultdict(list)
for k in fold_keys:
    subj = k.split("_rep")[0]
    if subj not in folds_by_subject:
        subjects_order.append(subj)
    folds_by_subject[subj].append(k)

n_subjects = len(subjects_order)


def fmt(s):
    h, m = divmod(int(s), 3600)
    m, sec = divmod(m, 60)
    return f"{h}h {m}m {sec}s"


print(
    f"LOSO-128 training: {args.model} | {n} folds ({args.reps} reps × {n_subjects} subjects) | "
    f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} bs={args.batch_size} "
    f"dropout={args.dropout} min_epoch={args.min_epoch} aug_shift={args.aug_shift} "
    f"label_smoothing={args.label_smoothing} sign_flip_p={args.sign_flip_p} "
    f"euclidean_align={args.euclidean_align}"
)

t0 = time.time()
fold_i = 0
subject_times = []

for s_idx, subj in enumerate(subjects_order):
    subj_start = time.time()
    subj_folds = folds_by_subject[subj]

    for rep_idx, fold_key in enumerate(subj_folds):
        fold_start = time.time()
        fold_i += 1
        print(
            f"\n[LOSO-128 {fold_i}/{n}] {args.model} — {fold_key}  "
            f"(subject {s_idx+1}/{n_subjects}, rep {rep_idx+1}/{args.reps})"
        )
        cmd = [
            "python", "-m", "src.train",
            "--model",        args.model,
            "--mode",         args.mode,
            "--fold",         fold_key,
            "--epochs",       args.epochs,
            "--lr",           args.lr,
            "--weight_decay", args.weight_decay,
            "--batch_size",      args.batch_size,
            "--dropout",         args.dropout,
            "--min_epoch",       args.min_epoch,
            "--aug_shift",       args.aug_shift,
            "--label_smoothing", args.label_smoothing,
            "--sign_flip_p",     args.sign_flip_p,
            "--num_workers",     args.num_workers,
        ]
        if args.euclidean_align:
            cmd.append("--euclidean_align")
        subprocess.run(cmd)
        fold_elapsed = time.time() - fold_start
        elapsed = time.time() - t0
        avg_fold = elapsed / fold_i
        eta = avg_fold * (n - fold_i)
        print(
            f"  rep done in {fmt(fold_elapsed)} | overall {fold_i}/{n} | "
            f"elapsed {fmt(elapsed)} | ETA {fmt(eta)}"
        )

    subj_elapsed = time.time() - subj_start
    subject_times.append(subj_elapsed)
    avg_subj = sum(subject_times) / len(subject_times)
    subj_eta = avg_subj * (n_subjects - s_idx - 1)
    print(
        f"\n[Subject {s_idx+1}/{n_subjects} — {subj}] All {args.reps} reps done in {fmt(subj_elapsed)} | "
        f"avg/subject {fmt(avg_subj)} | subject ETA {fmt(subj_eta)}"
    )

print(f"\nAll {n} LOSO-128 folds complete. Total time: {fmt(time.time()-t0)}")
