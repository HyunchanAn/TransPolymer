#!/bin/bash

# TransPolymer Finetuning Script (Mac/MPS)
# Usage: bash run_finetune.sh

set -e  # Exit immediately if a command exits with a non-zero status.

echo "========================================"
echo "🎯 TransPolymer Finetuning (General)"
echo "========================================"
echo "Config: configs/config_finetune.yaml"

.venv/bin/python Downstream.py --config configs/config_finetune.yaml
