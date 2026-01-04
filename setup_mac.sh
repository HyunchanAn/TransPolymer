#!/bin/bash

# TransPolymer Mac Setup Script
# Run this script to set up the environment automatically.

echo "========================================"
echo "🧪 TransPolymer Setup for Mac (Apple Silicon)"
echo "========================================"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found."
    echo "Please install Python from https://www.python.org/downloads/macos/"
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

# 2. Create Virtual Environment
echo "📦 Creating virtual environment (.venv)..."
# Force creation (clean slate)
if [ -d ".venv" ]; then
    rm -rf .venv
fi
python3 -m venv .venv

if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Failed to create virtual environment. Please check your Python installation."
    python3 --version
    which python3
    exit 1
fi
echo "✅ Virtual environment created."

# 3. Activate Virtual Environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# 4. Install Dependencies
echo "⬇️  Installing dependencies (this may take a while)..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Success Message
echo "========================================"
echo "🎉 Setup Complete!"
echo "========================================"
echo ""
echo "To run the pretraining demo:"
echo "   source .venv/bin/activate"
echo "   python Pretrain.py --config configs/config.yaml"
echo ""
echo "To run the App demo:"
echo "   source .venv/bin/activate"
echo "   streamlit run app.py"
echo "========================================"
