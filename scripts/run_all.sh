#!/usr/bin/env bash
# End-to-end MVP: data -> features -> baselines -> figures.
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuses the existing CUDA-enabled conda env (torch 2.12.1+cu130).
PY=${PYTHON:-python3}

echo "== 1/6 dataset =="
$PY -m src.data.prepare_dataset --download --extract --audit --split

echo "== 2/6 features =="
$PY scripts/extract_features.py --workers 8

echo "== 3/6 classical baselines (SVM, MLP) =="
$PY -m src.train_classical --models svm mlp --seeds 42 123 2024

echo "== 4/6 deep baselines (CNN, CRNN, Transformer) =="
$PY -m src.train_cnn --model cnn --seeds 42 123 2024 --epochs 30
$PY -m src.train_cnn --model crnn --seeds 42 123 2024 --epochs 30
$PY -m src.train_cnn --model transformer --seeds 42 123 2024 --epochs 30

echo "== 5/6 figures + summary =="
$PY -m src.evaluate

echo "== 6/6 tests =="
$PY -m pytest -q tests
