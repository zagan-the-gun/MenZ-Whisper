#!/bin/bash
# GPU環境チェックスクリプト (Unix/Linux/macOS)

# 仮想環境をアクティベート
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# チェックスクリプトを実行
python3 check_gpu.py

