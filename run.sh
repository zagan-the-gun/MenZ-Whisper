#!/bin/bash
# MenZ-Whisper 実行スクリプト（Linux/macOS）

# 仮想環境をアクティベート
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "エラー: 仮想環境が見つかりません"
    echo "先に setup.sh を実行してください"
    exit 1
fi

# メイン処理を実行
echo "=========================================="
echo "MenZ-Whisper を起動します"
echo "=========================================="
echo ""

python -m app.main "$@"

