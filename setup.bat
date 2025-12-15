@echo off
REM MenZ-Whisper セットアップスクリプト（Windows）

echo ==========================================
echo MenZ-Whisper セットアップ
echo ==========================================
echo.

REM Pythonバージョンチェック
echo Pythonバージョンをチェック中...
python --version
if errorlevel 1 (
    echo エラー: Pythonが見つかりません
    echo https://www.python.org/ からPythonをインストールしてください
    pause
    exit /b 1
)

REM 仮想環境の作成
if not exist "venv" (
    echo.
    echo 仮想環境を作成中...
    python -m venv venv
    echo 仮想環境を作成しました
) else (
    echo.
    echo 仮想環境が既に存在します
)

REM 仮想環境をアクティベート
echo.
echo 仮想環境をアクティベート中...
call venv\Scripts\activate.bat

REM pipをアップグレード
echo.
echo pipをアップグレード中...
python -m pip install --upgrade pip setuptools wheel

REM 依存関係のインストール
echo.
echo 依存関係をインストール中...
pip install -r requirements.txt

REM 必要なディレクトリの作成
echo.
echo 必要なディレクトリを作成中...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "cache" mkdir cache

REM 設定ファイルの確認
if not exist "config.ini" (
    echo.
    echo 警告: config.ini が見つかりません
    echo デフォルトの設定ファイルが既に存在するはずですが、見つからない場合は手動で作成してください。
) else (
    echo.
    echo config.ini が存在します
)

echo.
echo ==========================================
echo セットアップ完了！
echo ==========================================
echo.
echo 次のコマンドで起動できます:
echo   run.bat
echo.
echo または:
echo   venv\Scripts\activate
echo   python -m app.main
echo.

pause

