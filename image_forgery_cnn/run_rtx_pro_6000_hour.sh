#!/usr/bin/env bash
set -euo pipefail

python train_forgery_cnn.py \
  --dataset-slug divg07/casia-20-image-tampering-detection-dataset \
  --output-dir runs/casia_rtx_pro_6000_hour \
  --max-per-class 5000 \
  --epochs 16 \
  --freeze-epochs 2 \
  --batch-size 96 \
  --image-size 384 \
  --num-workers 12 \
  --input-mode rgb
