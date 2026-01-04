#!/bin/bash

# TransPolymer Finetuning Script (Mac/MPS)
# Usage: bash run_finetune.sh

echo "========================================"
echo "🎯 TransPolymer Finetuning (General)"
echo "========================================"
echo "Config: configs/config_finetune.yaml"

source .venv/bin/activate
python Downstream.py --config configs/config_finetune.yaml
