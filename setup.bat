@echo off
chcp 65001 > nul
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

REM Python UTF-8エンコーディング設定
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM pipをアップグレード
echo.
echo pipをアップグレード中...
python -m pip install --upgrade pip setuptools wheel

REM GPU選択
echo.
echo GPU support? (y/n)
set /p GPU="Choice: "

if /i "%GPU%"=="y" (
    echo.
    echo Installing PyTorch with CUDA ^(this may take several minutes^)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo.
        echo WARNING: CUDA installation failed, using CPU version...
        pip install torch torchvision torchaudio
    )
) else (
    echo.
    echo Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
)

REM PyTorchバージョン確認
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul

REM 依存関係のインストール（PyTorchは既にインストール済み）
echo.
echo 依存関係をインストール中...
echo ^(PyTorchは既にインストール済みのためスキップされます^)
pip install openai-whisper faster-whisper silero-vad
pip install numpy soundfile librosa sounddevice websockets tqdm

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

