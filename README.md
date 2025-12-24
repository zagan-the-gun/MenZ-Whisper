# MenZ-Whisper 🎤

OpenAI Whisperベースの音声認識システム（zagaroid連携専用）

## 概要

OpenAI Whisperモデルを使用した高精度音声認識サービスを提供するMCPクライアントです。zagaroidサーバーに接続し、JSON-RPC 2.0形式の音声認識リクエストを処理します。

**主な特徴:**

* OpenAI Whisper / Faster-Whisper モデルによる高精度音声認識
* **2つの動作モード（同時実行可能）**:
  - **ネットワークモード**: zagaroidサーバーに接続して音声認識リクエストを処理
  - **マイクモード**: マイクからの音声をリアルタイムで認識
  - **両方同時**: マイク入力とネットワーク依頼を同時に処理可能
* **JSON-RPC 2.0対応**（MCP準拠）
* 多言語対応（日本語に最適化）
* GPU/MPS/CPU 自動選択対応
* faster-whisper対応（標準版より高速）
* Silero VAD内蔵（音声区間自動検出）
* 自動再接続機能（エクスポネンシャルバックオフ）
* 詳細なログ機能

## OpenAI Whisperについて

[OpenAI Whisper](https://github.com/openai/whisper)は、OpenAIが開発したオープンソースの音声認識モデルです：

- 68万時間の多言語音声データで訓練
- 99種類の言語に対応
- 高い認識精度（特に日本語で優秀）
- MITライセンス（商用利用可能）

**faster-whisper**は、WhisperをCTranslate2で最適化した高速実装版で、標準版より最大4倍高速に動作します。

## システム要件

* **Python**: 3.8 以上
* **メモリ**: 4GB以上推奨（GPUメモリ含む）
* **GPU**: NVIDIA GPU (CUDA対応) またはApple Silicon推奨（CPUでも動作可能）
* **ストレージ**: 5GB以上の空き容量（モデルダウンロード用）
* **ネットワーク**: 初回実行時のモデルダウンロードに必要

## インストール

### 自動インストール（推奨）

#### macOS/Linux

```bash
# プロジェクトをダウンロード
git clone https://github.com/your-username/MenZ-Whisper.git
cd MenZ-Whisper

# 自動セットアップを実行
chmod +x setup.sh
./setup.sh

# クライアントを起動
./run.sh
```

#### Windows

```batch
# プロジェクトをダウンロード
git clone https://github.com/your-username/MenZ-Whisper.git
cd MenZ-Whisper

# 自動セットアップを実行
setup.bat

# クライアントを起動
run.bat
```

### 手動インストール

```bash
# 1. 仮想環境を作成（推奨）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows

# 2. 依存関係をインストール
pip install -r requirements.txt

# 3. クライアントを起動（初回起動時に自動的にモデルがダウンロードされます）
python -m app.main
```

## 設定

設定ファイル `config.ini` を編集して、システムの動作をカスタマイズできます。

### 主要設定

```ini
[mode]
# 動作モード（両方ともtrueにすることで同時実行可能）
# enable_network: ネットワーク経由のSTT依頼（zagaroid連携）
# enable_microphone: マイクからの直接入力（リアルタイム音声認識）
enable_network = true
enable_microphone = false

[microphone]
# マイク入力設定（mode=microphoneの時のみ有効）
device_id = 0  # 0=デフォルト, auto=起動時に選択
speaker = whisper_user  # 話者名（マイク入力時の話者識別子）
show_level = true  # 音声レベル表示
send_to_websocket = false  # 認識結果をzagaroidに送信

[model]
# 使用するモデルサイズ: tiny, base, small, medium, large, large-v2, large-v3
model_size = base
# faster-whisper使用（推奨）
use_faster_whisper = true
cache_dir = ./models

[inference]
device = auto  # auto, cpu, cuda, mps
gpu_id = auto
compute_type = float16  # float16, int8_float16, int8
beam_size = 5

[whisper]
language = ja  # ja, en, auto
task = transcribe  # transcribe, translate
vad_filter = true

[websocket]
host = 192.168.1.6
port = 50001

[debug]
show_debug = true
show_transcription = true
```

### モデルサイズの選択

| モデル | メモリ使用量 | 相対速度 | 精度 |
|-------|------------|---------|-----|
| tiny | ~1GB | 最速 | 低 |
| base | ~1GB | 高速 | 中 |
| small | ~2GB | 普通 | 中高 |
| medium | ~5GB | 遅い | 高 |
| large | ~10GB | 最遅 | 最高 |
| large-v2 | ~10GB | 最遅 | 最高 |
| large-v3 | ~10GB | 最遅 | 最高 |

**推奨:**
- リアルタイム用途: `base` または `small`
- 高精度が必要: `medium` または `large-v3`
- 低スペックPC: `tiny`

### デバイス設定

* `auto`: 自動選択（CUDA → MPS → CPU の順）
* `cuda`: NVIDIA GPU使用
* `mps`: Apple Silicon GPU使用（openai-whisperのみ）
* `cpu`: CPU使用

**注意:** faster-whisperはMPS（Apple Silicon GPU）に対応していません。Apple Siliconで高速化したい場合は、`use_faster_whisper = false`に設定してください。

## 使用方法

### 起動方法

1. **設定ファイルを編集**

`config.ini`:
```ini
[mode]
# 動作モード（両方ともtrueで同時実行可能）
enable_network = true     # ネットワーク経由のSTT依頼を処理
enable_microphone = false # マイクからの音声を認識

# ネットワークモードの場合
[websocket]
host = 192.168.1.6
port = 50001

# マイクモードの場合
[microphone]
device_id = auto  # 起動時に選択、または 0, 1, 2...
speaker = my_name
show_level = true

[model]
model_size = base
use_faster_whisper = true
```

**同時実行の例:**
```ini
[mode]
enable_network = true     # zagaroidからの音声認識依頼を処理
enable_microphone = true  # 自分のマイクからも音声入力
```

2. **起動**

```bash
./run.sh  # macOS/Linux
run.bat   # Windows
```

### GPU環境チェック

環境の準備状況を包括的に確認できます。

```bash
# Python直接実行
python check_gpu.py

# シェルスクリプト
./check_gpu.sh  # macOS/Linux
check_gpu.bat   # Windows
```

**チェック内容:**
- Python環境
- PyTorch環境（CUDA/MPS対応状況）
- Whisper環境（openai-whisper/faster-whisper）
- 音声処理ライブラリ
- システムリソース
- 設定ファイル
- 総合動作テスト

### マイクデバイスの選択

`config.ini` で `device_id = auto` に設定すると、起動時にマイクデバイスを選択できます：

```
利用可能なマイクデバイス:
  [0] MacBook Proのマイク (1ch, 48000Hz)
  [1] 外部USBマイク (2ch, 48000Hz)

マイクデバイスIDを入力してください (デフォルト: 0):
```

特定のデバイスを固定したい場合は、`device_id = 0` のように数値で指定します。

### 出力例

```
🔍 MenZ-Whisper 環境チェック
==================================================

🐍 Python環境:
  バージョン: 3.12.0
  プラットフォーム: macOS-14.0

🔥 PyTorch環境:
  PyTorch: 2.1.0
  CUDA利用可能: False
  MPS（Apple Silicon）: 利用可能

🎤 Whisper環境:
  openai-whisper: インストール済み
  faster-whisper: インストール済み
  ✅ 推奨: faster-whisperは標準版より高速です

🎵 音声処理ライブラリ:
  numpy: 1.24.3
  soundfile: 0.12.1
  librosa: 0.10.1
  silero-vad: インストール済み

📡 WebSocket環境:
  websockets: 12.0

💻 システムリソース:
  CPU: 8コア (使用率: 15.2%)
  メモリ: 16.0GB (使用率: 45.3%)
  利用可能ディスク容量: 256.5GB

==================================================
✅ 環境チェック完了
==================================================
```

## 動作モード

### ネットワークモード

zagaroidサーバーに接続し、JSON-RPC 2.0形式の音声認識リクエストを処理します。

**設定:**
```ini
[mode]
enable_network = true
enable_microphone = false
```

**用途:**
- zagaroidとの連携
- DiscordやTwitchからの音声認識
- サーバー・クライアント構成

**起動:** `python -m app.main` または `./run.sh`

### マイクモード

マイクからの音声をリアルタイムで認識します。

**設定:**
```ini
[mode]
enable_network = false
enable_microphone = true
```

**用途:**
- ローカルでの音声認識
- マイク音声のテスト
- 単体での音声入力

**制約:**
- **1つのマイクデバイスから入力**
- **話者は1人**（設定ファイルで指定）
- 複数人が同時に話しても全て同じ話者として扱われます

**起動:** `python -m app.main` または `./run.sh`

**機能:**
- Silero VADによる自動音声区間検出
- リアルタイム音声レベル表示
- 認識結果のコンソール出力（話者名付き）
- オプション: WebSocketでzagaroidに送信

### 同時実行モード（新機能）

**ネットワークモードとマイクモードを同時に実行**できます。

**設定:**
```ini
[mode]
enable_network = true
enable_microphone = true
```

**用途:**
- マイクから自分の音声を入力しつつ、Discord/Twitchからの音声もリアルタイムで認識
- ローカルマイクとリモート音声の同時処理
- 配信者が自分の声とリスナーの声を同時に認識

**動作:**
- マイクからの音声: 設定した話者名で認識
- ネットワークからの依頼: リクエストごとに話者名を指定
- 両方の音声を並行処理（互いに影響しません）
- 同じWhisperモデルを共有して効率的に処理

**起動:** `python -m app.main` または `./run.sh`

## JSON-RPC 2.0プロトコル（ネットワークモード）

### 音声認識リクエスト

**リクエスト:**
```json
{
  "jsonrpc": "2.0",
  "method": "recognize_audio",
  "params": {
    "speaker": "user_001",
    "audio_data": "BASE64_ENCODED_PCM16LE",
    "sample_rate": 16000,
    "channels": 1,
    "format": "pcm16le"
  }
}
```

**レスポンス（通知）:**
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subtitle",
  "params": {
    "text": "認識されたテキスト",
    "speaker": "user_001",
    "type": "subtitle",
    "language": "ja"
  }
}
```

### パラメータ

| パラメータ | 型 | 必須 | 説明 |
|-----------|----|----|------|
| `speaker` | string | ✅ | 話者識別子 |
| `audio_data` | string | ✅ | Base64エンコードされたPCM16LE音声データ |
| `sample_rate` | integer | | サンプリングレート（デフォルト: 16000） |
| `channels` | integer | | チャンネル数（デフォルト: 1） |
| `format` | string | | 音声フォーマット（デフォルト: pcm16le） |

## トラブルシューティング

### GPU が認識されない

```bash
# GPU環境をチェック
python check_gpu.py

# NVIDIA GPU
nvidia-smi

# PyTorchのCUDA対応版を再インストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### faster-whisperでエラーが発生

faster-whisperが動作しない場合は、標準のopenai-whisperにフォールバックします：

```ini
[model]
use_faster_whisper = false
```

### Apple Silicon (M1/M2/M3) で遅い

Apple SiliconでGPUアクセラレーションを使用する場合：

```ini
[inference]
device = mps

[model]
use_faster_whisper = false  # faster-whisperはMPS非対応
```

### メモリ不足エラー

モデルサイズを小さくするか、compute_typeを変更します：

```ini
[model]
model_size = base  # または tiny

[inference]
compute_type = int8  # メモリ使用量削減
```

### 認識精度が低い

以下の設定を調整してください：

```ini
[model]
model_size = medium  # より大きなモデルを使用

[inference]
beam_size = 5  # ビームサイズを増やす（精度向上、速度低下）

[whisper]
language = ja  # 言語を明示的に指定
initial_prompt = こんにちは。今日は良い天気ですね。  # 認識ヒント
```

### マイクが認識されない

`config.ini` でマイクデバイスを確認してください：

```ini
[microphone]
device_id = auto  # 起動時に選択画面が表示されます
```

### マイクモードで音声が検出されない

VAD設定を調整してください：

```ini
[silero_vad]
threshold = 0.3  # 閾値を上げる（より敏感に）
min_speech_duration_ms = 100  # 最小音声長を短く
min_silence_duration_ms = 300  # 無音判定を短く
```

## プロジェクト構造

```
MenZ-Whisper/
├── app/                          # メインパッケージ
│   ├── __init__.py
│   ├── config.py                 # 設定管理
│   ├── model.py                  # Whisperモデル
│   ├── jsonrpc_handler.py        # JSON-RPC処理
│   ├── mcp_client.py             # MCPクライアント
│   └── main.py                   # メイン実行ファイル
├── config.ini                    # 設定ファイル
├── requirements.txt              # 依存関係
├── setup.sh / setup.bat          # セットアップスクリプト
├── run.sh / run.bat              # 実行スクリプト
├── check_gpu.py                  # GPU環境チェック
├── check_gpu.sh / check_gpu.bat  # チェック実行スクリプト
├── logs/                         # ログファイル
├── models/                       # モデルキャッシュ
├── cache/                        # 一時ファイル
└── README.md                     # このファイル
```

## パフォーマンスチューニング

### GPU最適化（NVIDIA）

```ini
[inference]
device = cuda
compute_type = float16  # GPU高速化

[model]
use_faster_whisper = true  # faster-whisperを使用
```

### Apple Silicon最適化

```ini
[inference]
device = mps  # Apple Silicon GPU

[model]
use_faster_whisper = false  # openai-whisperを使用
```

### CPU最適化

```ini
[inference]
device = cpu
compute_type = int8  # CPU最適化
cpu_threads = 0  # 自動設定

[model]
use_faster_whisper = true  # faster-whisperはCPUでも高速
model_size = base  # 軽量モデル
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

使用するOpenAI WhisperモデルもMITライセンスで、研究・商用利用が可能です。

## 謝辞

- **OpenAI Whisper**: OpenAIによる優秀な音声認識モデルの提供に感謝いたします
- **faster-whisper**: Guillaume Kleinによる高速実装版に感謝いたします
- **参考プロジェクト**: 
  - [MenZ-ReazonSpeech](https://github.com/zagan-the-gun/MenZ-ReazonSpeech) - アーキテクチャを参考
  - [MenZ-FuguMT](https://github.com/zagan-the-gun/MenZ-FuguMT) - 設定スタイルを参考
  - [MenZ-GeminiCLI](https://github.com/zagan-the-gun/MenZ-GeminiCLI) - ディレクトリ構造を参考

## 貢献

プルリクエストやイシューの報告を歓迎します。

## サポート

問題が発生した場合は、GitHub Issues でお知らせください。

---

**MenZ-Whisper** - 高精度な音声認識をあなたのアプリケーションに 🎤

