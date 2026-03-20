#!/bin/bash
# Run all CNN+LSTM experiments: subject-dependent and LOSO
# Usage: bash experiments/run_cnn_lstm.sh
# Run from project root: /home/dhruvpatel/EEGNetC147A

set -e  # exit on error

MODEL="cnn_lstm"
LOG_DIR="experiments/logs"
mkdir -p $LOG_DIR

echo "========================================"
echo "  CNN+LSTM Experiments"
echo "========================================"

# -------------------------------------------------------------------
# Subject-dependent: train on each of the 9 subjects
# -------------------------------------------------------------------
echo ""
echo "--- Subject-Dependent ---"

for SUBJECT in A01 A02 A03 A04 A05 A06 A07 A08 A09; do
    echo ""
    echo "Training: $MODEL | subject_dependent | $SUBJECT"
    python -m src.train \
        --model $MODEL \
        --mode subject_dependent \
        --subject $SUBJECT \
        --epochs 100 \
        --lr 0.001 \
        --batch_size 32 \
        --weight_decay 0.0001 \
        2>&1 | tee $LOG_DIR/${MODEL}_${SUBJECT}_subject_dependent.log
done

# -------------------------------------------------------------------
# LOSO: train on each of the 9 folds
# -------------------------------------------------------------------
echo ""
echo "--- LOSO ---"

for FOLD in 0 1 2 3 4 5 6 7 8; do
    echo ""
    echo "Training: $MODEL | loso | fold $FOLD"
    python -m src.train \
        --model $MODEL \
        --mode loso \
        --fold $FOLD \
        --epochs 100 \
        --lr 0.001 \
        --batch_size 32 \
        --weight_decay 0.001 \
        2>&1 | tee $LOG_DIR/${MODEL}_fold${FOLD}_loso.log
done

# -------------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------------
echo ""
echo "--- Evaluation ---"

python -m src.evaluate --model $MODEL --mode subject_dependent
python -m src.evaluate --model $MODEL --mode loso

echo ""
echo "========================================"
echo "  All CNN+LSTM experiments complete"
echo "========================================"