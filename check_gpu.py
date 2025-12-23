#!/usr/bin/env python3
"""
GPU環境チェックスクリプト
MenZ-Whisperで利用可能なGPU環境を確認します。
"""

import sys
import platform
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_python_environment():
    """Python環境チェック"""
    print("🐍 Python環境:")
    print(f"  バージョン: {sys.version}")
    print(f"  プラットフォーム: {platform.platform()}")
    print()


def check_pytorch():
    """PyTorch環境チェック"""
    print("🔥 PyTorch環境:")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA利用可能: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDAバージョン: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"  GPUデバイス数: {gpu_count}")
            print()
            
            print("  利用可能なGPU:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_properties = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_properties.total_memory / (1024**3)
                compute_capability = f"{gpu_properties.major}.{gpu_properties.minor}"
                
                # メモリ使用状況
                try:
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    free = gpu_memory - cached
                    print(f"    GPU {i}: {gpu_name}")
                    print(f"      メモリ: {gpu_memory:.1f}GB (使用中: {cached:.1f}GB, 空き: {free:.1f}GB)")
                    print(f"      Compute Capability: {compute_capability}")
                except Exception:
                    print(f"    GPU {i}: {gpu_name}")
                    print(f"      メモリ: {gpu_memory:.1f}GB")
                    print(f"      Compute Capability: {compute_capability}")
                print()
            
            if gpu_count > 1:
                print("  🔍 複数GPUが検出されました！")
                print("  設定ファイルでGPU IDを指定することで特定のGPUを使用できます:")
                print("  config.ini の [inference] セクションで")
                print("    device = cuda")
                print("    gpu_id = 0 # 使用したいGPUのID（0から始まる）")
                print()
        
        # Apple Silicon (MPS) チェック
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  MPS（Apple Silicon）: 利用可能")
            print("  ⚠️ faster-whisperはMPS非対応のため、use_faster_whisper=trueの場合はCPUが使用されます")
            print()
            
    except ImportError:
        print("  ❌ PyTorchがインストールされていません")
        print("  pip install torch でインストールしてください")
        print()


def check_whisper():
    """Whisper環境チェック"""
    print("🎤 Whisper環境:")
    
    # OpenAI Whisper
    try:
        import whisper
        print(f"  openai-whisper: インストール済み")
        available_models = whisper.available_models()
        print(f"  利用可能なモデル: {', '.join(available_models)}")
    except ImportError:
        print("  ❌ openai-whisperがインストールされていません")
        print("  pip install openai-whisper でインストールしてください")
    
    # Faster Whisper
    try:
        import faster_whisper
        print(f"  faster-whisper: インストール済み")
        print(f"  バージョン: {faster_whisper.__version__}")
        print("  ✅ 推奨: faster-whisperは標準版より高速です")
    except ImportError:
        print("  ❌ faster-whisperがインストールされていません（推奨）")
        print("  pip install faster-whisper でインストールしてください")
    
    print()


def check_audio_libraries():
    """音声処理ライブラリチェック"""
    print("🎵 音声処理ライブラリ:")
    
    # numpy
    try:
        import numpy as np
        print(f"  numpy: {np.__version__}")
    except ImportError:
        print("  ❌ numpyがインストールされていません")
        print("  pip install numpy でインストールしてください")
    
    # soundfile
    try:
        import soundfile
        print(f"  soundfile: {soundfile.__version__}")
    except ImportError:
        print("  ❌ soundfileがインストールされていません")
        print("  pip install soundfile でインストールしてください")
    
    # librosa
    try:
        import librosa
        print(f"  librosa: {librosa.__version__}")
    except ImportError:
        print("  ❌ librosaがインストールされていません")
        print("  pip install librosa でインストールしてください")
    
    # silero-vad
    try:
        import silero_vad
        print(f"  silero-vad: インストール済み")
    except ImportError:
        print("  ❌ silero-vadがインストールされていません")
        print("  pip install silero-vad でインストールしてください")
    
    print()


def check_websocket():
    """WebSocket環境チェック"""
    print("📡 WebSocket環境:")
    try:
        import websockets
        print(f"  websockets: {websockets.__version__}")
    except ImportError:
        print("  ❌ websocketsがインストールされていません")
        print("  pip install websockets でインストールしてください")
    
    print()


def check_system_resources():
    """システムリソースチェック"""
    print("💻 システムリソース:")
    try:
        import psutil
        
        # CPU情報
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"  CPU: {cpu_count}コア (使用率: {cpu_percent}%)")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_percent = memory.percent
        print(f"  メモリ: {memory_gb:.1f}GB (使用率: {memory_percent}%)")
        
        if memory_gb < 4:
            print("  ⚠️ メモリが4GB未満です。動作が制限される可能性があります。")
        elif memory_gb < 8:
            print("  ⚠️ メモリが8GB未満です。大きなモデルで問題が生じる可能性があります。")
        
        # ディスク容量
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        print(f"  利用可能ディスク容量: {disk_gb:.1f}GB")
        
        if disk_gb < 5:
            print("  ⚠️ ディスク容量が不足しています。モデルダウンロードに必要です。")
        
        print()
        
    except ImportError:
        print("  psutilがインストールされていません（オプション）")
        print("  pip install psutil でより詳細な情報を確認できます")
        print()


def check_config():
    """設定ファイルチェック"""
    print("⚙️ 設定ファイルチェック:")
    
    config_path = project_root / "config.ini"
    if config_path.exists():
        print(f"  ✅ config.ini が存在します")
        
        try:
            from app.config import WhisperConfig
            config = WhisperConfig.from_ini(str(config_path))
            print(f"  モデルサイズ: {config.model_size}")
            print(f"  faster-whisper使用: {config.use_faster_whisper}")
            print(f"  デバイス設定: {config.device}")
            print(f"  GPU ID: {config.gpu_id}")
            print(f"  言語: {config.language or 'auto'}")
        except Exception as e:
            print(f"  ⚠️ 設定ファイルの読み込みエラー: {e}")
    else:
        print(f"  ⚠️ config.ini が見つかりません")
        print(f"  デフォルトの設定ファイルが同梱されているはずです")
    
    print()


def test_whisper_model():
    """Whisperモデル読み込みテスト"""
    print("🧪 総合動作テスト:")
    
    try:
        from app.config import WhisperConfig
        from app.model import WhisperModel
        import numpy as np
        
        print("  設定ファイル読み込み中...")
        config = WhisperConfig()
        
        # テスト用に小さいモデルを使用
        original_model_size = config.model_size
        if config.model_size not in ['tiny', 'base']:
            config.model_size = 'tiny'
            print(f"  テスト用に {config.model_size} モデルを使用します")
        
        print("  Whisperモデル初期化中...")
        print("  (初回実行時はモデルのダウンロードが行われるため時間がかかります...)")
        
        model = WhisperModel(config)
        print("  ✅ モデル初期化成功")
        
        # ダミー音声でテスト（1秒の無音）
        print("  音声認識テスト中...")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        result = model.transcribe_audio_segment(dummy_audio)
        print(f"  ✅ 音声認識テスト成功 (結果: '{result}')")
        
        # クリーンアップ
        model.cleanup()
        print("  ✅ 総合テスト成功")
        
    except Exception as e:
        print(f"  ❌ 総合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def main():
    """メイン関数"""
    print("=" * 50)
    print("🔍 MenZ-Whisper 環境チェック")
    print("=" * 50)
    print()
    
    check_python_environment()
    check_pytorch()
    check_whisper()
    check_audio_libraries()
    check_websocket()
    check_system_resources()
    check_config()
    test_whisper_model()
    
    print("=" * 50)
    print("✅ 環境チェック完了")
    print("=" * 50)
    print()
    print("推奨設定:")
    print("  - GPUが利用可能な場合: device = cuda, use_faster_whisper = true")
    print("  - Apple Siliconの場合: device = mps, use_faster_whisper = false")
    print("  - CPUのみの場合: device = cpu, use_faster_whisper = true")
    print()


if __name__ == "__main__":
    main()


