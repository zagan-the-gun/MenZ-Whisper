@echo off
REM MenZ-Whisper 実行スクリプト（Windows）

REM 仮想環境をアクティベート
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo エラー: 仮想環境が見つかりません
    echo 先に setup.bat を実行してください
    pause
    exit /b 1
)

REM メイン処理を実行
echo ==========================================
echo MenZ-Whisper を起動します
echo ==========================================
echo.

python -m app.main %*

