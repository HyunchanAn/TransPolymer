@echo off
setlocal

:: Check if the virtual environment directory exists
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [INFO] Installing other dependencies...
pip install -r requirements.txt

echo.
echo [SUCCESS] Environment setup complete.
echo To activate the environment manually, run: venv\Scripts\activate
echo.
