#!/bin/bash
# MenZ-Whisper セットアップスクリプト（Linux/macOS）

set -e

echo "=========================================="
echo "MenZ-Whisper セットアップ"
echo "=========================================="
echo ""

# Pythonバージョンチェック
echo "Pythonバージョンをチェック中..."
python3 --version

# 仮想環境の作成
if [ ! -d "venv" ]; then
    echo ""
    echo "仮想環境を作成中..."
    python3 -m venv venv
    echo "✅ 仮想環境を作成しました"
else
    echo ""
    echo "✅ 仮想環境が既に存在します"
fi

# 仮想環境をアクティベート
echo ""
echo "仮想環境をアクティベート中..."
source venv/bin/activate

# pipをアップグレード
echo ""
echo "pipをアップグレード中..."
pip install --upgrade pip setuptools wheel

# 依存関係のインストール
echo ""
echo "依存関係をインストール中..."
pip install -r requirements.txt

# 必要なディレクトリの作成
echo ""
echo "必要なディレクトリを作成中..."
mkdir -p logs
mkdir -p models
mkdir -p cache

# 設定ファイルの確認
if [ ! -f "config.ini" ]; then
    echo ""
    echo "⚠️  警告: config.ini が見つかりません"
    echo "デフォルトの設定ファイルが既に存在するはずですが、見つからない場合は手動で作成してください。"
else
    echo ""
    echo "✅ config.ini が存在します"
fi

echo ""
echo "=========================================="
echo "✅ セットアップ完了！"
echo "=========================================="
echo ""
echo "次のコマンドで起動できます:"
echo "  ./run.sh"
echo ""
echo "または:"
echo "  source venv/bin/activate"
echo "  python -m app.main"
echo ""

