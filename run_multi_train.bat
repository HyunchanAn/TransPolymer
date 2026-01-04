@echo off
setlocal
cd /d "%~dp0"
call venv\Scripts\activate
python Downstream.py --config configs/config_finetune_Multi.yaml
pause
