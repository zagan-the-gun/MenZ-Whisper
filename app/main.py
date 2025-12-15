#!/usr/bin/env python3
"""
MenZ-Whisper メイン実行ファイル
- network: MCPクライアントとしてzagaroidに接続し、音声認識リクエストを処理
- microphone: マイクからの音声をリアルタイムで認識
"""

import sys
import asyncio
import logging
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import WhisperConfig
from app.model import WhisperModel
from app.mcp_client import MCPClient
from app.jsonrpc_handler import JSONRPCHandler
from app.realtime import RealtimeTranscriber


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="MenZ-Whisper - OpenAI Whisperベースの音声認識システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本（config.iniの設定に従う）
  python -m app.main
  ./run.sh
  
  # 設定ファイルを一時的に上書き
  python -m app.main --mode microphone     # マイクモード
  python -m app.main --mode network        # ネットワークモード
  
  # マイクデバイス確認
  python -m app.main --list-devices
  
  # カスタム設定ファイル
  python -m app.main --config my_config.ini

推奨: config.iniで動作モードを設定し、python -m app.main で起動
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['network', 'microphone'],
        help='動作モード（config.iniの設定を上書き）'
    )
    
    parser.add_argument(
        '--microphone', '--mic',
        action='store_true',
        help='マイクモードで起動（--mode microphone の短縮形）'
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='利用可能なマイクデバイスを一覧表示'
    )
    
    parser.add_argument(
        '--device-id',
        type=int,
        help='マイクデバイスID'
    )
    
    parser.add_argument(
        '--config',
        default='config.ini',
        help='設定ファイルのパス（デフォルト: config.ini）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細なログを表示'
    )
    
    return parser.parse_args()


async def run_network_mode(config, model):
    """ネットワークモード（zagaroid連携）"""
    logger.info("ネットワークモード: zagaroidサーバーに接続します")
    
    # JSON-RPCハンドラーの初期化
    jsonrpc_handler = JSONRPCHandler(model, config)
    
    # MCPクライアントの初期化
    mcp_client = MCPClient(config, model, jsonrpc_handler)
    
    # シャットダウンイベントの設定
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("シャットダウンシグナルを受信しました")
        shutdown_event.set()
    
    # シグナルハンドラーの登録（WindowsとUnixで異なる実装）
    import signal
    import platform
    
    if platform.system() == 'Windows':
        # Windows環境ではsignal.signal()を使用
        def windows_signal_handler(signum, frame):
            logger.info("シャットダウンシグナルを受信しました")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, windows_signal_handler)
        signal.signal(signal.SIGTERM, windows_signal_handler)
        
        try:
            # クライアント開始
            await mcp_client.start_client(shutdown_event)
        finally:
            # シグナルハンドラーをデフォルトに戻す
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
    else:
        # Unix系環境ではloop.add_signal_handler()を使用
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
        
        try:
            # クライアント開始
            await mcp_client.start_client(shutdown_event)
        finally:
            # シグナルハンドラーの解除
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)


def run_microphone_mode(config, model, device_id=None):
    """マイクモード（リアルタイム音声認識）"""
    logger.info("マイクモード: リアルタイム音声認識を開始します")
    
    # RealtimeTranscriberの初期化
    transcriber = RealtimeTranscriber(
        config=config,
        model=model,
        show_level=config.microphone_show_level
    )
    
    # 音声認識開始
    transcriber.start(device_id=device_id)


def list_devices():
    """マイクデバイス一覧を表示"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        print("\n利用可能なマイクデバイス:")
        print("="*60)
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default_mark = "（デフォルト）" if i == sd.default.device[0] else ""
                print(f"[{i}] {device['name']} {default_mark}")
                print(f"    入力チャンネル: {device['max_input_channels']}")
                print(f"    サンプリングレート: {device['default_samplerate']:.0f}Hz")
                print()
        
        print("="*60)
        print("使用方法: python -m app.main --mode microphone --device-id <ID>")
        
    except ImportError:
        logger.error("sounddeviceがインストールされていません")
        logger.error("pip install sounddevice でインストールしてください")
        sys.exit(1)


async def main():
    """メイン関数"""
    args = parse_arguments()
    
    # デバイス一覧表示
    if args.list_devices:
        list_devices()
        return
    
    # 詳細ログ
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 設定の読み込み
        try:
            config = WhisperConfig.from_ini(args.config)
            logger.info(f"設定ファイルを読み込みました: {args.config}")
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗: {e}")
            logger.info("デフォルト設定を使用します")
            config = WhisperConfig()
        
        # 動作モードの決定
        if args.microphone:
            mode = 'microphone'
        elif args.mode:
            mode = args.mode
        else:
            mode = config.mode
        
        logger.info(f"動作モード: {mode}")
        
        # デバイスの自動選択
        if config.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    config.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # faster-whisperはMPS未対応なのでCPUを使用
                    if config.use_faster_whisper:
                        config.device = "cpu"
                        logger.info("faster-whisper使用時はMPS非対応のためCPUを使用します")
                    else:
                        config.device = "mps"
                else:
                    config.device = "cpu"
                logger.info(f"自動デバイス選択: {config.device}")
            except ImportError:
                config.device = "cpu"
                logger.info("PyTorchが見つからないため、CPUを使用します")
        
        # モデルのロード
        logger.info(f"Whisper音声認識モデルをロード中...")
        logger.info(f"モデルサイズ: {config.model_size}")
        logger.info(f"エンジン: {'faster-whisper' if config.use_faster_whisper else 'openai-whisper'}")
        logger.info(f"デバイス: {config.device}")
        
        model = WhisperModel(config)
        logger.info("モデルのロードが完了しました")
        
        try:
            # モード別の処理
            if mode == 'network':
                await run_network_mode(config, model)
            elif mode == 'microphone':
                run_microphone_mode(config, model, device_id=args.device_id)
            else:
                logger.error(f"未知の動作モード: {mode}")
                sys.exit(1)
        finally:
            # モデルのクリーンアップ
            model.cleanup()
        
    except KeyboardInterrupt:
        logger.info("終了しました")
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
