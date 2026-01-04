@echo off
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Starting training (Finely-tuning)...
python Downstream.py --config configs/config_finetune.yaml

echo.
echo [DONE] Execution finished.

