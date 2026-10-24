#!/usr/bin/env bash
# Phase 1 (token forecasting) end-to-end. Usage: bash scripts/run_token.sh [seeds...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export MUSIC_TOKENS=data/features/token_prediction
export MUSIC_CKPT=checkpoints/token_prediction
export MUSIC_RESULTS=results/token_prediction
export MUSIC_DATASET_NAME=FMA-small
SEEDS=${*:-"42 123 2024"}

echo "== 1/4 encode FMA windows with EnCodec 32 kHz =="
$PY -m src.forecasting.prepare_tokens --windows-per-track 2 --shard-size 512 \
    --encode-batch 16 --workers 10

echo "== 2/4 train + evaluate (baselines, LSTM, Transformer) =="
$PY -m src.forecasting.train_token --seeds $SEEDS --epochs 15

echo "== 3/4 decode predicted tokens back to audio =="
$PY -m src.forecasting.decode_tokens --model transformer --seed 42

echo "== 4/4 figures =="
$PY -m src.forecasting.plot_tokens
