#!/usr/bin/env bash
# Leakage comparison (track-level split) + AST transfer on FMA-small.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}

echo "== leakage comparison: CNN on track-level split =="
MUSIC_FEATURES=data/features/fma/features_tracklevel.npz \
MUSIC_CKPT=checkpoints/fma_leak \
MUSIC_RESULTS=results/fma_leak \
$PY -m src.train_cnn --model cnn --seeds 42 --epochs 30

echo "== AST transfer on FMA-small =="
MUSIC_SPLITS=data/splits/fma/splits.csv \
MUSIC_AST_CACHE=data/features/fma/ast_embeddings.npz \
MUSIC_RESULTS=results/fma \
MUSIC_AST_DOC=docs/ast_fma.md \
$PY -m src.train_ast

echo "== merge AST =="
MUSIC_RESULTS=results/fma $PY scripts/integrate_ast.py
MUSIC_RESULTS=results/fma MUSIC_SUMMARY_DATA="FMA-small（7994 首，8 流派，艺术家分离划分）" $PY -m src.evaluate
