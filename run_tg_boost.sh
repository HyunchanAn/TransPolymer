#!/bin/bash

# TransPolymer Tg-Boost Training Script (Mac/MPS)
# Usage: bash run_tg_boost.sh

set -e  # Exit immediately if a command exits with a non-zero status.

echo "========================================"
echo "🚀 TransPolymer Tg-Boost Training"
echo "========================================"
echo "Step 1: Multi-task Pre-training..."

# Check if data exists
if [ ! -f "data/train_Multi.csv" ]; then
    echo "⚠️  Data not found. Generatiing data from raw source..."
    .venv/bin/python utils/prepare_multi_data.py
fi

.venv/bin/python Downstream.py --config configs/config_finetune_Multi.yaml

echo ""
echo "Step 2: Boosting with Tg Specific Data..."

# Check if merged data exists
if [ ! -f "data/merged/train_Multi_POINT2.csv" ]; then
    echo "⚠️  Merged Data not found. Merging Multi-task and POINT2 data..."
    # Ensure dependencies are met (Multi data must exist, which is handled in Step 1)
    .venv/bin/python utils/merge_point2.py
fi

.venv/bin/python Downstream.py --config configs/config_finetune_Multi_Boost.yaml

echo "========================================"
echo "✅ Tg-Boost Training Complete!"
echo "Model saved to: ckpt/model_multi_boost_best.pt"
