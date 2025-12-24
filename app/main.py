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
import json
import time
import numpy as np


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# 優先度定数
PRIORITY_MICROPHONE = 0  # マイク入力（最優先）
PRIORITY_NETWORK = 1     # ネットワーク入力（通常）


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
  
  # マイクデバイス確認
  python -m app.main --list-devices

推奨: config.iniで enable_network と enable_microphone を設定
      両方ともtrueにすると、マイク入力とネットワーク依頼を同時処理できます
        """
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='利用可能なマイクデバイスを一覧表示'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細なログを表示'
    )
    
    return parser.parse_args()


async def recognition_worker(queue, model, config, websocket, shutdown_event):
    """音声認識ワーカー
    
    優先度付きキューから音声認識リクエストを取り出して処理します。
    
    Args:
        queue: 優先度付きキュー
        model: Whisperモデル
        config: 設定オブジェクト
        websocket: WebSocket接続（結果送信用）
        shutdown_event: 停止イベント
    """
    logger.info("音声認識ワーカーを開始しました")
    
    while not shutdown_event.is_set():
        try:
            # キューからリクエストを取得（タイムアウト付き）
            try:
                priority, timestamp, request_data = await asyncio.wait_for(
                    queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            
            speaker = request_data['speaker']
            audio_data = request_data['audio_data']
            priority_name = "マイク" if priority == PRIORITY_MICROPHONE else "ネットワーク"
            
            # キューの待ち時間を計算
            wait_time = time.time() - timestamp
            queue_size = queue.qsize()
            
            logger.info(f"[{priority_name}] 音声認識開始: speaker={speaker}, "
                       f"待ち時間={wait_time:.2f}秒, キュー残={queue_size}")
            
            try:
                # 音声認識実行
                start_time = time.time()
                result = model.transcribe_audio_segment(audio_data)
                elapsed = time.time() - start_time
                
                if result and result.strip():
                    # フィルタリング
                    processed_result = result.strip()
                    
                    if config.exclude_whitespace_only and not processed_result:
                        logger.debug(f"空白のみのためスキップ: speaker={speaker}")
                    elif len(processed_result) < config.min_length:
                        logger.debug(f"最小文字数未満のためスキップ: {processed_result}")
                    else:
                        # 文章途中の「。」を「、」に変換（末尾以外）
                        if len(processed_result) > 1:
                            processed_result = processed_result[:-1].replace('。', '、') + processed_result[-1]
                        # 末尾の句読点を削除
                        processed_result = processed_result.rstrip('。')
                        
                        logger.info(f"[{priority_name}] 認識成功: speaker={speaker}, "
                                   f"text={processed_result}, 処理時間={elapsed:.2f}秒")
                        
                        # WebSocketで結果を送信
                        if websocket:
                            try:
                                notification = {
                                    "jsonrpc": "2.0",
                                    "method": "notifications/subtitle",
                                    "params": {
                                        "text": processed_result,
                                        "speaker": speaker,
                                        "type": "subtitle",
                                        "language": "ja"
                                    }
                                }
                                await websocket.send(json.dumps(notification, ensure_ascii=False))
                            except Exception as send_error:
                                logger.error(f"WebSocket送信エラー: {send_error}")
                        
                        # マイクモードの場合はコンソールにも表示
                        if priority == PRIORITY_MICROPHONE:
                            print(f"\n🎤 認識結果 [{speaker}] ({elapsed:.2f}秒): {processed_result}")
                else:
                    logger.debug(f"認識結果なし: speaker={speaker}")
                    
            except Exception as e:
                logger.error(f"音声認識エラー: speaker={speaker}, error={e}", exc_info=True)
            finally:
                queue.task_done()
                
        except Exception as e:
            logger.error(f"ワーカーエラー: {e}", exc_info=True)
    
    logger.info("音声認識ワーカーを停止しました")


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


async def run_microphone_mode_async(config, model, device_id, shutdown_event=None, recognition_queue=None):
    """マイクモード（リアルタイム音声認識）"""
    logger.info("マイクモード: リアルタイム音声認識を開始します")
    
    # RealtimeTranscriberの初期化
    transcriber = RealtimeTranscriber(
        config=config,
        model=model,
        show_level=config.microphone_show_level,
        recognition_queue=recognition_queue
    )
    
    # 音声認識開始
    await transcriber.start_async(device_id=device_id, stop_event=shutdown_event)


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
        config_file = 'config.ini'
        try:
            config = WhisperConfig.from_ini(config_file)
            logger.info(f"設定ファイルを読み込みました: {config_file}")
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗: {e}")
            logger.info("デフォルト設定を使用します")
            config = WhisperConfig()
        
        # 動作モードの確認
        enable_network = config.enable_network
        enable_microphone = config.enable_microphone
        
        # モード表示
        modes = []
        if enable_network:
            modes.append("ネットワーク")
        if enable_microphone:
            modes.append("マイク")
        
        if not modes:
            logger.error("有効なモードがありません。config.iniで enable_network または enable_microphone を true に設定してください。")
            sys.exit(1)
        
        mode_str = " + ".join(modes) + "モード"
        logger.info(f"動作モード: {mode_str}")
        
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
            # マイクデバイスの選択（マイクモードが有効な場合）
            mic_device_id = None
            if enable_microphone:
                device_str = config.microphone_device_id
                if device_str == 'select':
                    # 一覧から選択
                    from app.realtime import RealtimeTranscriber
                    temp_transcriber = RealtimeTranscriber(config, model)
                    mic_device_id = temp_transcriber.select_microphone()
                    if mic_device_id is None:
                        logger.error("マイクデバイスが選択されませんでした。終了します。")
                        return
                    logger.info(f"選択されたマイクデバイスID: {mic_device_id}")
                else:
                    try:
                        mic_device_id = int(device_str)
                        logger.info(f"設定ファイルで指定されたマイクデバイスID: {mic_device_id}")
                    except ValueError:
                        logger.error(f"無効なマイクデバイスID: {device_str}")
                        logger.error("数値または 'select' を指定してください")
                        return
            
            # モード別の処理
            if enable_network and enable_microphone:
                # 両方のモードを同時実行
                logger.info("ネットワークモードとマイクモードを同時に実行します")
                
                # 共通の停止イベント
                shutdown_event = asyncio.Event()
                
                # 優先度付きキューの作成
                queue_max_size = config.queue_max_size
                recognition_queue = asyncio.PriorityQueue(maxsize=queue_max_size)
                logger.info(f"音声認識キューを作成しました（最大サイズ: {queue_max_size}）")
                
                # シグナルハンドラーの設定
                def signal_handler():
                    logger.info("シャットダウンシグナルを受信しました")
                    shutdown_event.set()
                
                import signal
                import platform
                
                if platform.system() == 'Windows':
                    def windows_signal_handler(signum, frame):
                        signal_handler()
                    signal.signal(signal.SIGINT, windows_signal_handler)
                    signal.signal(signal.SIGTERM, windows_signal_handler)
                else:
                    loop = asyncio.get_running_loop()
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.add_signal_handler(sig, signal_handler)
                
                # 両方のタスクを並行実行
                try:
                    # JSON-RPCハンドラーとMCPクライアントの初期化
                    from app.jsonrpc_handler import JSONRPCHandler
                    from app.mcp_client import MCPClient
                    
                    jsonrpc_handler = JSONRPCHandler(model, config)
                    mcp_client = MCPClient(config, model, jsonrpc_handler, recognition_queue)
                    
                    # WebSocket接続が確立されるまで待機するためのタスクを作成
                    async def wait_for_websocket_and_start_worker():
                        # MCPクライアントが接続を確立するまで少し待つ
                        await asyncio.sleep(2)
                        
                        # WebSocket接続を取得
                        websocket = mcp_client.websocket
                        if websocket:
                            logger.info("ワーカーを開始します")
                            await recognition_worker(recognition_queue, model, config, websocket, shutdown_event)
                        else:
                            logger.warning("WebSocket接続がありません。ワーカーを開始できません")
                    
                    # すべてを並行実行
                    await asyncio.gather(
                        mcp_client.start_client(shutdown_event),
                        run_microphone_mode_async(config, model, mic_device_id, shutdown_event=shutdown_event, recognition_queue=recognition_queue),
                        wait_for_websocket_and_start_worker()
                    )
                finally:
                    # シグナルハンドラーの解除
                    if platform.system() == 'Windows':
                        signal.signal(signal.SIGINT, signal.SIG_DFL)
                        signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    else:
                        loop = asyncio.get_running_loop()
                        for sig in (signal.SIGINT, signal.SIGTERM):
                            loop.remove_signal_handler(sig)
                            
            elif enable_network:
                # ネットワークモードのみ
                logger.info("ネットワークモードで起動します")
                
                # 停止イベント
                shutdown_event = asyncio.Event()
                
                # 優先度付きキューの作成
                queue_max_size = config.queue_max_size
                recognition_queue = asyncio.PriorityQueue(maxsize=queue_max_size)
                logger.info(f"音声認識キューを作成しました（最大サイズ: {queue_max_size}）")
                
                # シグナルハンドラーの設定
                def signal_handler():
                    logger.info("シャットダウンシグナルを受信しました")
                    shutdown_event.set()
                
                import signal
                import platform
                
                if platform.system() == 'Windows':
                    def windows_signal_handler(signum, frame):
                        signal_handler()
                    signal.signal(signal.SIGINT, windows_signal_handler)
                    signal.signal(signal.SIGTERM, windows_signal_handler)
                else:
                    loop = asyncio.get_running_loop()
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.add_signal_handler(sig, signal_handler)
                
                try:
                    from app.jsonrpc_handler import JSONRPCHandler
                    from app.mcp_client import MCPClient
                    
                    jsonrpc_handler = JSONRPCHandler(model, config)
                    mcp_client = MCPClient(config, model, jsonrpc_handler, recognition_queue)
                    
                    # WebSocket接続確立後にワーカーを開始
                    async def wait_for_websocket_and_start_worker():
                        await asyncio.sleep(2)
                        websocket = mcp_client.websocket
                        if websocket:
                            logger.info("ワーカーを開始します")
                            await recognition_worker(recognition_queue, model, config, websocket, shutdown_event)
                    
                    await asyncio.gather(
                        mcp_client.start_client(shutdown_event),
                        wait_for_websocket_and_start_worker()
                    )
                finally:
                    if platform.system() == 'Windows':
                        signal.signal(signal.SIGINT, signal.SIG_DFL)
                        signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    else:
                        loop = asyncio.get_running_loop()
                        for sig in (signal.SIGINT, signal.SIGTERM):
                            loop.remove_signal_handler(sig)
                
            elif enable_microphone:
                # マイクモードのみ
                logger.info("マイクモードで起動します")
                
                shutdown_event = asyncio.Event()
                
                # 優先度付きキューの作成
                queue_max_size = config.queue_max_size
                recognition_queue = asyncio.PriorityQueue(maxsize=queue_max_size)
                logger.info(f"音声認識キューを作成しました（最大サイズ: {queue_max_size}）")
                
                # シグナルハンドラーの設定
                def signal_handler():
                    logger.info("シャットダウンシグナルを受信しました")
                    shutdown_event.set()
                
                import signal
                import platform
                
                if platform.system() == 'Windows':
                    def windows_signal_handler(signum, frame):
                        signal_handler()
                    signal.signal(signal.SIGINT, windows_signal_handler)
                    signal.signal(signal.SIGTERM, windows_signal_handler)
                else:
                    loop = asyncio.get_running_loop()
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.add_signal_handler(sig, signal_handler)
                
                try:
                    # マイクモードのみの場合、WebSocketはNone（コンソール出力のみ）
                    # ただし、send_to_websocketがtrueの場合は接続を確立する
                    websocket = None
                    if config.microphone_send_to_websocket:
                        logger.info("マイクの結果をWebSocketで送信します")
                        # WebSocket接続を確立（簡易版）
                        import websockets
                        uri = f"ws://{config.websocket_host}:{config.websocket_port}/"
                        try:
                            websocket = await websockets.connect(uri)
                            logger.info(f"WebSocketに接続しました: {uri}")
                        except Exception as e:
                            logger.warning(f"WebSocket接続失敗: {e}。コンソール出力のみになります")
                    
                    await asyncio.gather(
                        run_microphone_mode_async(config, model, mic_device_id, shutdown_event=shutdown_event, recognition_queue=recognition_queue),
                        recognition_worker(recognition_queue, model, config, websocket, shutdown_event)
                    )
                finally:
                    # WebSocket切断
                    if websocket:
                        try:
                            await websocket.close()
                        except Exception:
                            pass
                    
                    # シグナルハンドラーの解除
                    if platform.system() == 'Windows':
                        signal.signal(signal.SIGINT, signal.SIG_DFL)
                        signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    else:
                        loop = asyncio.get_running_loop()
                        for sig in (signal.SIGINT, signal.SIGTERM):
                            loop.remove_signal_handler(sig)
                
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
