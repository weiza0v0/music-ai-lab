#!/usr/bin/env bash
# Leakage comparison (track-level split) + AST transfer on FMA-small.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python}

echo "== leakage comparison: CNN on track-level split =="
MUSIC_FEATURES=data/fma/processed/features_tracklevel.npz \
MUSIC_CKPT=checkpoints_fma_leak \
MUSIC_RESULTS=results_fma_leak \
$PY -m src.train_cnn --model cnn --seeds 42 --epochs 30

echo "== AST transfer on FMA-small =="
MUSIC_SPLITS=data/splits_fma/splits.csv \
MUSIC_AST_CACHE=data/fma/processed/ast_embeddings.npz \
MUSIC_RESULTS=results_fma \
MUSIC_AST_DOC=docs/ast_fma.md \
$PY -m src.train_ast

echo "== merge AST =="
MUSIC_RESULTS=results_fma $PY scripts/integrate_ast.py
MUSIC_RESULTS=results_fma MUSIC_SUMMARY_DATA="FMA-small（7994 首，8 流派，艺术家分离划分）" $PY -m src.evaluate
