@echo off
echo [INFO] Creating virtual environment 'venv'...
python -m venv venv

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo [SUCCESS] Environment setup complete!
echo To activate the environment manually, run: venv\Scripts\activate
echo.

