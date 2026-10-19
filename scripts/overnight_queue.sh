#!/usr/bin/env bash
# 夜间训练队列：串行占用 GPU，逐条落日志，可重复执行（已完成的自动跳过）。
#
# 背景：复核发现两类"欠训练"，不是为了刷分，而是原有训练预算把模型停在了
# 还在变好的位置上：
#   1) 一期 Token 模型被早停砍在第 12 轮，但 val_nll 一路还在降
#      （LSTM 6.8146→6.8091→6.8060，Transformer 6.7249→6.7157→6.7127）；
#   2) 部分分类模型 best_epoch 正好撞在 30 轮上限（GTZAN CNN seed123、
#      GTZAN CRNN seed42、FMA CRNN seed42），即最后一轮仍在改善。
#
# 因此本队列统一加长训练轮数与 patience，并把结果写进 *_long 目录，
# **不覆盖**报告里已在用的结果，便于逐项对照后再决定是否采纳。
#
# 用法：
#   bash scripts/overnight_queue.sh              # 跑完整队列
#   ONLY=token_long bash scripts/overnight_queue.sh
#   nohup bash scripts/overnight_queue.sh >/dev/null 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python}
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export PYTHONUNBUFFERED=1          # 否则日志要等进程退出才落盘

SEEDS=${SEEDS:-"42 123 2024"}
ONLY=${ONLY:-}
LOGDIR=results/logs/overnight
STATUS="$LOGDIR/status.tsv"
mkdir -p "$LOGDIR" checkpoints
: > "$STATUS"

say() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }

want() {  # 只跑指定作业时支持子串匹配
  [ -z "$ONLY" ] && return 0
  case "$1" in *"$ONLY"*) return 0 ;; *) return 1 ;; esac
}

wait_gpu() {
  local used
  for _ in $(seq 1 960); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && { sleep 20; continue; }
    if [ "$used" -lt 800 ]; then return 0; fi
    sleep 20
  done
  say "GPU 持续占用中，仍尝试启动（可能排队）"
}

# run_job <名称> <哨兵文件> <命令...>
run_job() {
  local name="$1" sentinel="$2"; shift 2
  if ! want "$name"; then return 0; fi
  if [ -e "$sentinel" ]; then
    say "跳过 $name（已存在 $sentinel）"
    printf '%s\tSKIP\t%s\n' "$name" "$sentinel" >> "$STATUS"
    return 0
  fi
  wait_gpu
  # 排队等 GPU 期间外部进程可能已经产出结果，这里再判一次，避免重复训练
  if [ -e "$sentinel" ]; then
    say "跳过 $name（等待期间已产出 $sentinel）"
    printf '%s\tSKIP\t%s\n' "$name" "$sentinel" >> "$STATUS"
    return 0
  fi
  local t0 t1 rc
  t0=$(date +%s)
  say "开始 $name"
  "$@" >>"$LOGDIR/$name.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    say "完成 $name（$(( (t1 - t0) / 60 )) 分钟）"
    printf '%s\tOK\t%dmin\n' "$name" $(( (t1 - t0) / 60 )) >> "$STATUS"
  else
    say "失败 $name（rc=$rc，见 $LOGDIR/$name.log）"
    printf '%s\tFAIL(rc=%d)\t%dmin\n' "$name" "$rc" $(( (t1 - t0) / 60 )) >> "$STATUS"
  fi
}

# --------------------------------------------------------------------------- #
# 0) LoRA 修正版：训练 + 评估
#    （finetune_v2.json 若已由外部进程写好，这里会自动跳过，不会重复训练）
# --------------------------------------------------------------------------- #
if want lora_v2; then
  run_job lora_v2 results/generation/finetune_v2.json \
    "$PY" -m src.generation.finetune_lora \
        --steps 1200 --batch-size 16 --lr 1e-4 --rank 8 --alpha 16 \
        --mask-prefix --max-minutes 240 \
        --out musicgen-small-lora-v2 --result-name finetune_v2.json
fi

run_eval_v2() {
  if [ ! -d checkpoints/generation/musicgen-small-lora-v2 ]; then
    echo "缺少 checkpoints/generation/musicgen-small-lora-v2，跳过评估"
    return 1
  fi
  "$PY" -m src.generation.run_musicgen \
      --model checkpoints/generation/musicgen-small-lora-v2 \
      --tag ft --seconds 3 --n 8 --seed 42
}
if want eval_v2; then
  run_job eval_v2 results/generation/metrics_ft.json run_eval_v2
fi

# --------------------------------------------------------------------------- #
# 1) 一期 Token 模型：早停太早，加长轮数并放宽 patience
#    两种模型放同一次调用，保证 metrics.csv 汇总完整
# --------------------------------------------------------------------------- #
if want token_long; then
  run_job token_long results/token_prediction_long/metrics.csv \
    env MUSIC_RESULTS=results/token_prediction_long \
        MUSIC_CKPT=checkpoints/token_prediction_long \
        MUSIC_TOKENS=data/features/token_prediction \
    "$PY" -m src.forecasting.train_token \
        --kinds lstm transformer --epochs 45 --patience 10 --seeds $SEEDS
fi

# --------------------------------------------------------------------------- #
# 2) GTZAN 分类模型：加长到 60 轮
# --------------------------------------------------------------------------- #
for m in cnn crnn transformer; do
  if want "gtzan_long_$m"; then
    run_job "gtzan_long_$m" "checkpoints/gtzan_long/${m}_seed2024.pt" \
      env MUSIC_FEATURES=data/features/gtzan/features.npz \
          MUSIC_CKPT=checkpoints/gtzan_long \
          MUSIC_RESULTS=results/gtzan_long \
      "$PY" -m src.train_cnn --model "$m" --epochs 60 --patience 12 --seeds $SEEDS
  fi
done
if want gtzan_long_eval; then
  run_job gtzan_long_eval results/gtzan_long/metrics.csv \
    env MUSIC_RESULTS=results/gtzan_long \
        MUSIC_SUMMARY_DATA="GTZAN 清洗后 985 首（加长训练复核）" \
    "$PY" -m src.evaluate
fi

# --------------------------------------------------------------------------- #
# 3) FMA 主实验分类模型：加长到 60 轮
# --------------------------------------------------------------------------- #
for m in cnn crnn transformer; do
  if want "fma_long_$m"; then
    run_job "fma_long_$m" "checkpoints/fma_long/${m}_seed2024.pt" \
      env MUSIC_FEATURES=data/features/fma/features.npz \
          MUSIC_CKPT=checkpoints/fma_long \
          MUSIC_RESULTS=results/fma_long \
          MUSIC_DATASET_NAME=FMA-small \
      "$PY" -m src.train_cnn --model "$m" --epochs 60 --patience 12 --seeds $SEEDS
  fi
done
if want fma_long_eval; then
  run_job fma_long_eval results/fma_long/metrics.csv \
    env MUSIC_RESULTS=results/fma_long \
        MUSIC_DATASET_NAME=FMA-small \
        MUSIC_SUMMARY_DATA="FMA-small 7,994 首可解码（加长训练复核，artist-disjoint）" \
    "$PY" -m src.evaluate
fi

# --------------------------------------------------------------------------- #
# 4) LoRA 补充消融：换 rank / 去掉前缀掩码，用来定位增益来源
# --------------------------------------------------------------------------- #
if want lora_r16; then
  run_job lora_r16 results/generation/finetune_r16.json \
    "$PY" -m src.generation.finetune_lora \
        --steps 2000 --batch-size 16 --lr 1e-4 --rank 16 --alpha 32 \
        --mask-prefix --max-minutes 240 \
        --out musicgen-small-lora-r16 --result-name finetune_r16.json
fi

if want lora_full; then
  run_job lora_full results/generation/finetune_full.json \
    "$PY" -m src.generation.finetune_lora \
        --steps 1200 --batch-size 16 --lr 1e-4 --rank 8 --alpha 16 \
        --max-minutes 240 \
        --out musicgen-small-lora-full --result-name finetune_full.json
fi

# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
say "======== 队列结束 ========"
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"

{
  echo "# 夜间队列结果汇总"
  echo
  echo "生成时间：$(date '+%Y-%m-%d %H:%M')"
  echo
  echo "## 作业状态"
  echo '```'
  cat "$STATUS"
  echo '```'
  for f in results/token_prediction_long/metrics.csv results/gtzan_long/metrics.csv results/fma_long/metrics.csv; do
    [ -f "$f" ] || continue
    echo
    echo "## $f"
    echo '```'
    cat "$f"
    echo '```'
  done
  for f in results/generation/finetune_r16.json results/generation/finetune_full.json; do
    [ -f "$f" ] || continue
    echo
    echo "## $f（meta）"
    echo '```'
    "$PY" -c "import json,sys;print(json.dumps(json.load(open('$f'))['meta'],indent=2,ensure_ascii=False))"
    echo '```'
  done
} > "$LOGDIR/summary.md"
say "汇总已写入 $LOGDIR/summary.md"
