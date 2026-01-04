@echo off
echo Starting POINT2 Tg-Boost Training...
venv\Scripts\python Downstream.py --config configs/config_finetune_Multi_Boost.yaml
pause
