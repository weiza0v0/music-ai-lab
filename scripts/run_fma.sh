#!/usr/bin/env bash
# Main experiment on FMA-small. Usage: bash scripts/run_fma.sh [seeds...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python}
export MUSIC_FEATURES=data/features/fma/features.npz
export MUSIC_CKPT=checkpoints/fma
export MUSIC_RESULTS=results/fma
export MUSIC_DATASET_NAME=FMA-small
SEEDS=${*:-"42 123 2024"}

mkdir -p "$MUSIC_CKPT" "$MUSIC_RESULTS/figures" "$MUSIC_RESULTS/predictions"
echo "== classical =="
$PY -m src.train_classical --models svm mlp --seeds $SEEDS
echo "== cnn =="
$PY -m src.train_cnn --model cnn --seeds $SEEDS --epochs 30
echo "== crnn =="
$PY -m src.train_cnn --model crnn --seeds $SEEDS --epochs 30
echo "== transformer =="
$PY -m src.train_cnn --model transformer --seeds $SEEDS --epochs 30
echo "== evaluate =="
$PY -m src.evaluate
