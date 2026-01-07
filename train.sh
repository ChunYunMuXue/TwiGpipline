#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 单卡训练启动脚本（只用 0 号卡）
# 训练数据由 TwiGpipline/twig_config.json 的 train_data 决定（无需传 data_glob）
# loss: lambda_pix * L1 + lambda_perc * VGG perceptual
# -----------------------------

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 建议把 HF/Torch 缓存放到 /nfs（避免写到 home，小机器/多机更稳）
export HF_HOME=/nfs/wenjie/wenjie_0104/data/.cache/huggingface
export TORCH_HOME=/nfs/wenjie/wenjie_0104/data/.cache/torch

mkdir -p /nfs/wenjie/wenjie_0104/logs
echo "[run] logging to: /nfs/wenjie/wenjie_0104/logs/train_single.log"

python -u /nfs/wenjie/wenjie_0104/TwiGpipline/train_twig_control_image_loss.py \
  --out_dir /nfs/wenjie/wenjie_0104/checkpoints_control_image_loss \
  --batch_size 1 \
  --num_workers 4 \
  --max_steps 2000 \
  --tbptt_window 64 \
  --log_every 10 \
  --save_every 200 \
  --l1_weight 1.0 \
  --mse_weight 0.1 \
  2>&1 | tee -a /nfs/wenjie/wenjie_0104/logs/train_single.log