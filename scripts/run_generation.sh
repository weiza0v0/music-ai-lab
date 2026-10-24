#!/usr/bin/env bash
# Phase 2 (MusicGen continuation + optional LoRA). Run AFTER phase 1 tokens exist.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python}
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export MUSIC_RESULTS=results/generation
export MUSIC_CKPT=checkpoints/generation
export MUSIC_TOKENS=data/features/token_prediction

echo "== 1/4 base model continuation (6 tracks, 5 s prompt -> 5 s) =="
$PY -m src.generation.run_musicgen --tag base --n 6 --seconds 3

echo "== 2/4 LoRA adaptation (reuses phase-1 windows, 5 s prompt -> 1 s target) =="
$PY -m src.generation.finetune_lora --steps 300 --max-minutes 40

echo "== 3/4 continuation with the adapted model =="
$PY -m src.generation.run_musicgen \
    --model checkpoints/generation/musicgen-small-lora --tag ft --n 6 --seconds 3

echo "== 4/4 figures =="
$PY -m src.generation.plot_generation
