@echo off
REM GPU環境チェックスクリプト (Windows)

REM 仮想環境をアクティベート
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM チェックスクリプトを実行
python check_gpu.py

pause

