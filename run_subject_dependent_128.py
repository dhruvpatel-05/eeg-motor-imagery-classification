"""
Run subject-dependent training for all 9 subjects using the 128 Hz / ERS pipeline.

Usage:
    python run_subject_dependent_128.py --model eegnet [--epochs 300] [--lr 0.001] [--weight_decay 1e-4] [--batch_size 32]
"""
import argparse
import subprocess

SUBJECTS = [f"A0{i}" for i in range(1, 10)]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="Model name — use 'alternative_eegnet' for 128 Hz / 256-sample data")
parser.add_argument("--epochs",       default="300")
parser.add_argument("--lr",           default="0.001")
parser.add_argument("--weight_decay", default="5e-4")
parser.add_argument("--batch_size",   default="32")
parser.add_argument("--dropout",         default="0.6")
parser.add_argument("--min_epoch",       default="100")
parser.add_argument("--aug_shift",       default="64")
parser.add_argument("--label_smoothing", default="0.1")
parser.add_argument("--sign_flip_p",     default="0.0")
parser.add_argument("--band", default="full",
                    choices=["delta","theta","mu","beta","gamma","mu_beta","full"])
args = parser.parse_args()

print(f"Subject-dependent training (128 Hz): {args.model} | "
      f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} bs={args.batch_size} "
      f"dropout={args.dropout} min_epoch={args.min_epoch} "
      f"aug_shift={args.aug_shift} label_smoothing={args.label_smoothing} "
      f"sign_flip_p={args.sign_flip_p}")

for subj in SUBJECTS:
    print(f"\n[subject_dependent_128] {args.model} — {subj}")

    subprocess.run([
        "python", "-m", "src.train_128",
        "--model",        args.model,
        "--mode",         "subject_dependent",
        "--subject",      subj,
        "--band",         args.band,
        "--epochs",       args.epochs,
        "--lr",           args.lr,
        "--weight_decay", args.weight_decay,
        "--batch_size",   args.batch_size,
        "--dropout",         args.dropout,
        "--min_epoch",       args.min_epoch,
        "--aug_shift",       args.aug_shift,
        "--label_smoothing", args.label_smoothing,
        "--sign_flip_p",     args.sign_flip_p,
    ])
