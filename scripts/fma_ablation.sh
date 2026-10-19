#!/usr/bin/env bash
# Validation-only optimisation study (seed 42). Test split is never evaluated here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python}
export MUSIC_FEATURES=data/features/fma/features.npz
mkdir -p checkpoints/fma_ablation results/fma_ablation

run_cfg() {
  local name=$1; shift
  echo "== $name =="
  MUSIC_RESULTS=results/fma_ablation/$name \
  MUSIC_CKPT=checkpoints/fma_ablation/$name \
  $PY -m src.train_cnn "$@"
}

run_cfg cnn_base       --model cnn --seeds 42 --epochs 20 --augment none --val-only
run_cfg cnn_specaug    --model cnn --seeds 42 --epochs 20 --augment specaug --val-only
run_cfg cnn_specaug_ls --model cnn --seeds 42 --epochs 20 --augment specaug --label-smoothing 0.05 --dropout 0.4 --val-only
run_cfg crnn_specaug   --model crnn --seeds 42 --epochs 20 --augment specaug --label-smoothing 0.05 --val-only
run_cfg tf_specaug     --model transformer --seeds 42 --epochs 20 --augment specaug --label-smoothing 0.05 --val-only
