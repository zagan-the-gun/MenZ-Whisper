"""
MCPクライアントモジュール

zagaroidサーバーに接続してJSON-RPC 2.0通信を行います。
- 送信: 音声認識結果の字幕通知
- 受信: 音声認識リクエストの処理
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Optional
from .config import WhisperConfig
from .jsonrpc_handler import JSONRPCHandler


class MCPClient:
    """MCPクライアントクラス（双方向通信対応）"""
    
    def __init__(self, config: WhisperConfig, model, jsonrpc_handler: JSONRPCHandler):
        """
        MCPクライアントの初期化
        
        Args:
            config: 設定オブジェクト
            model: Whisperモデル
            jsonrpc_handler: JSON-RPCハンドラー
        """
        self.config = config
        self.model = model
        self.jsonrpc_handler = jsonrpc_handler
        self.logger = logging.getLogger(__name__)
        
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self.reconnect_delay = 3  # 再接続待機時間（秒）
        self.max_reconnect_delay = 60  # 最大再接続待機時間（秒）
        
        # 統計情報
        self.stats = {
            'start_time': time.time(),
            'total_requests': 0,
            'total_notifications': 0,
            'total_errors': 0,
            'reconnect_count': 0,
            'last_connected': None
        }
    
    async def start_client(self, shutdown_event=None):
        """クライアント開始"""
        try:
            self.logger.info("MenZ-Whisper MCPクライアントを開始します...")
            uri = f"ws://{self.config.websocket_host}:{self.config.websocket_port}/"
            self.logger.info(f"接続先: {uri}")
            
            self.running = True
            current_delay = self.reconnect_delay
            
            while self.running:
                try:
                    # zagaroidサーバーに接続
                    self.logger.info(f"zagaroidサーバーに接続中: {uri}")
                    
                    async with websockets.connect(
                        uri,
                        max_size=None,
                        ping_interval=30,
                        ping_timeout=60
                    ) as websocket:
                        self.websocket = websocket
                        self.stats['last_connected'] = time.time()
                        current_delay = self.reconnect_delay  # 接続成功したら遅延をリセット
                        
                        self.logger.info("✅ zagaroidサーバーに接続しました")
                        
                        # 接続完了通知を送信
                        connect_msg = {
                            "jsonrpc": "2.0",
                            "method": "notifications/subtitle",
                            "params": {
                                "text": "MenZ-Whisper接続完了",
                                "speaker": "whisper",
                                "type": "system",
                                "language": "ja"
                            }
                        }
                        await websocket.send(json.dumps(connect_msg, ensure_ascii=False))
                        
                        # シャットダウンイベントとメッセージ処理を並行実行
                        if shutdown_event:
                            done, pending = await asyncio.wait(
                                [
                                    asyncio.create_task(self._message_loop(websocket)),
                                    asyncio.create_task(shutdown_event.wait())
                                ],
                                return_when=asyncio.FIRST_COMPLETED
                            )
                            
                            # 残りのタスクをキャンセル
                            for task in pending:
                                task.cancel()
                                try:
                                    await task
                                except asyncio.CancelledError:
                                    pass
                        else:
                            await self._message_loop(websocket)
                        
                        # 正常切断
                        if not self.running:
                            self.logger.info("クライアントを正常に停止しました")
                            break
                            
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("zagaroidサーバーとの接続が切断されました")
                    
                except Exception as e:
                    self.logger.error(f"接続エラー: {e}")
                    self.stats['reconnect_count'] += 1
                
                finally:
                    self.websocket = None
                
                # 再接続処理
                if self.running:
                    self.logger.info(f"{current_delay}秒後に再接続を試みます...")
                    
                    # sleepとshutdown_eventを並行して待機
                    if shutdown_event:
                        sleep_task = asyncio.create_task(asyncio.sleep(current_delay))
                        shutdown_task = asyncio.create_task(shutdown_event.wait())
                        
                        done, pending = await asyncio.wait(
                            [sleep_task, shutdown_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # 残りのタスクをキャンセル
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        
                        # シャットダウンが先に完了した場合
                        if shutdown_event.is_set():
                            break
                    else:
                        await asyncio.sleep(current_delay)
                    
                    # エクスポネンシャルバックオフ（指数バックオフ）
                    current_delay = min(current_delay * 2, self.max_reconnect_delay)
                    
        except Exception as e:
            self.logger.error(f"クライアント開始エラー: {e}", exc_info=True)
            raise
        finally:
            self.stop_client()
    
    async def _message_loop(self, websocket):
        """メッセージ受信ループ"""
        try:
            async for message in websocket:
                try:
                    await self._process_message(websocket, message)
                except Exception as e:
                    self.logger.error(f"メッセージ処理エラー: {e}", exc_info=True)
                    self.stats['total_errors'] += 1
                        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("接続が切断されました")
            raise
        except Exception as e:
            self.logger.error(f"メッセージループエラー: {e}", exc_info=True)
            raise
    
    async def _process_message(self, websocket, message: str):
        """メッセージ処理"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            self.logger.error(f"無効なJSONフォーマット: {e}")
            return
        
        # JSON-RPC 2.0リクエストの検証
        if not self.jsonrpc_handler.is_jsonrpc_request(data):
            self.logger.debug(f"非JSON-RPCメッセージを受信（無視）")
            return
        
        # リクエスト処理
        method = data.get('method')
        self.logger.info(f"JSON-RPCリクエスト受信: method={method}")
        self.stats['total_requests'] += 1
        
        # ハンドラーで処理（通知が返る）
        notification = await self.jsonrpc_handler.handle_request(data)
        
        # 通知を送信
        if notification:
            await websocket.send(json.dumps(notification, ensure_ascii=False))
            self.stats['total_notifications'] += 1
            self.logger.debug(f"通知送信完了: method={notification.get('method')}")
    
    def stop_client(self):
        """クライアント停止"""
        self.logger.info("クライアントを停止しています...")
        self.running = False
        
        # WebSocket接続を閉じる
        if self.websocket and not self.websocket.closed:
            try:
                asyncio.create_task(self.websocket.close())
            except RuntimeError:
                # イベントループが実行されていない場合はスキップ
                pass
        
        self.logger.info("クライアントが停止されました")
    
    def get_stats(self):
        """統計情報の取得"""
        stats = self.stats.copy()
        stats['uptime_seconds'] = time.time() - stats['start_time']
        stats['connected'] = self.websocket is not None and not self.websocket.closed
        return stats

