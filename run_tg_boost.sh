#!/bin/bash

# TransPolymer Tg-Boost Training Script (Mac/MPS)
# Usage: bash run_tg_boost.sh

set -e  # Exit immediately if a command exits with a non-zero status.

echo "========================================"
echo "🚀 TransPolymer Tg-Boost Training"
echo "========================================"
echo "Step 1: Multi-task Pre-training..."
.venv/bin/python Downstream.py --config configs/config_finetune_Multi.yaml

echo ""
echo "Step 2: Boosting with Tg Specific Data..."
.venv/bin/python Downstream.py --config configs/config_finetune_Multi_Boost.yaml

echo "========================================"
echo "✅ Tg-Boost Training Complete!"
echo "Model saved to: ckpt/model_multi_boost_best.pt"
