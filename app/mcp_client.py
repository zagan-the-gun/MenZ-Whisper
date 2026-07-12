"""
MZP（MenZ Protocol）クライアントモジュール

zagaroidサーバーに接続してJSON-RPC 2.0（MZP v1.0）通信を行います。
ワイヤ仕様の正本は zagaroid/docs/protocol.md。
- 送信: initialize（接続確立ごと）、音声認識結果の字幕通知 / レスポンス
- 受信: 音声認識リクエスト（recognize_audio）の処理

（モジュール名 mcp_client は歴史的経緯。旧称 "MCP" は Anthropic の
 Model Context Protocol とは無関係の独自規約で、現名称は MZP）
"""

import asyncio
import json
import logging
import time
import uuid
import websockets
from typing import Optional
from .config import WhisperConfig
from .jsonrpc_handler import JSONRPCHandler

# MZP v1.0（zagaroid/docs/protocol.md § 4.1）
PROTOCOL_VERSION = "mzp/1.0"
CLIENT_NAME = "MenZ-Whisper"
# initialize 応答の待機上限（protocol.md § 6.4）。超過時は再接続からやり直す
INITIALIZE_TIMEOUT_SECONDS = 10.0


class HandshakeError(Exception):
    """initialize が受理されなかった（エラー応答・タイムアウト）。"""


class MCPClient:
    """MCPクライアントクラス（双方向通信対応）"""
    
    def __init__(self, config: WhisperConfig, model, jsonrpc_handler: JSONRPCHandler, recognition_queue=None):
        """
        MCPクライアントの初期化
        
        Args:
            config: 設定オブジェクト
            model: Whisperモデル
            jsonrpc_handler: JSON-RPCハンドラー
            recognition_queue: 音声認識キュー（優先度付き）
        """
        self.config = config
        self.model = model
        self.jsonrpc_handler = jsonrpc_handler
        self.recognition_queue = recognition_queue
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

                        # MZP initialize（接続確立ごとに必須。再接続時も再送する）。
                        # 失敗時は例外 → 外側の except → バックオフ再接続に乗せる
                        await self._handshake(websocket)

                        # 接続確認用の挨拶を字幕として送信する。
                        # config.ini [microphone] speaker の名義で送ることで OBS の {speaker}_subtitle に
                        # 表示され、MenZ-Whisper → zagaroid → OBS の経路確認テストを兼ねる。
                        # （旧実装は speaker="whisper" 固定・type="system" だったが、zagaroid の MZP 移行後は
                        #   未登録話者として破棄されるため、登録済みの speaker 名義・type="subtitle" に変更）
                        connect_msg = {
                            "jsonrpc": "2.0",
                            "method": "notifications/subtitle",
                            "params": {
                                "text": "MenZ-Whisper接続完了",
                                "speaker": self.config.microphone_speaker,
                                "type": "subtitle",
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

                except HandshakeError as e:
                    self.logger.warning(f"initialize に失敗しました（再接続します）: {e}")
                    self.stats['reconnect_count'] += 1

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

    def _build_roles(self):
        """動作モードから MZP roles（protocol.md § 2.1）を組み立てる。

        - enable_network: hub からの recognize_audio を受けて返す → stt-pull
        - enable_microphone: マイク認識結果をこの接続で自発通知する → stt-push
          （mic+network 同時実行時、マイク結果は常にこの MZP 接続で送られるため）
        """
        roles = []
        if self.config.enable_network:
            roles.append("stt-pull")
        if self.config.enable_microphone:
            roles.append("stt-push")
        return roles

    async def _handshake(self, websocket):
        """接続確立ごとに initialize を送り、応答を検証する（protocol.md § 3.1 / § 4.1）。"""
        msg_id = str(uuid.uuid4())
        roles = self._build_roles()
        payload = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "initialize",
            "params": {
                "protocol": PROTOCOL_VERSION,
                "name": CLIENT_NAME,
                "roles": roles,
                # actors は wipe ロール専用フィールドのため送らない（protocol.md § 4.1）
            },
        }
        await websocket.send(json.dumps(payload, ensure_ascii=False))
        self.logger.info(f"initialize 送信（roles: {', '.join(roles)}）")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + INITIALIZE_TIMEOUT_SECONDS
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HandshakeError("initialize 応答がタイムアウトしました")
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                raise HandshakeError("initialize 応答がタイムアウトしました") from None

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
                continue
            # 応答（id 一致・method なし）だけを拾う。ハンドシェイク完了前に届いた
            # 字幕ブロードキャスト等は取りこぼしてよい（hub の応答は接続直後に返る）
            if "method" in data or str(data.get("id")) != msg_id:
                self.logger.debug(f"initialize 応答待ち中の受信を無視: {raw[:80]}")
                continue
            error = data.get("error")
            result = data.get("result") or {}
            if error is not None or not result.get("ok"):
                raise HandshakeError(f"initialize が拒否されました: {error or result}")
            self.logger.info(f"initialize 受理（{result.get('protocol', '?')}）")
            return

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
        
        # 音声認識リクエストの場合はキューに追加
        if method == 'recognize_audio' and self.recognition_queue:
            await self._enqueue_recognition_request(websocket, data)
        else:
            # その他のリクエスト（キュー無し時の recognize_audio 含む）を処理
            reply = await self.jsonrpc_handler.handle_request(data)

            # 応答（レスポンスまたは通知）を送信
            if reply:
                await websocket.send(json.dumps(reply, ensure_ascii=False))
                self.stats['total_notifications'] += 1
                self.logger.debug(f"応答送信完了: method={reply.get('method')}, id={reply.get('id')}")
    
    async def _enqueue_recognition_request(self, websocket, request: dict):
        """音声認識リクエストをキューに追加

        MZP v1.0（protocol.md § 4.4）では recognize_audio は id 付きリクエストで届き、
        結果は同じ id のレスポンスで返す。id 無し（旧 hub のブロードキャスト）で
        届いた場合は互換のため従来通り notifications/subtitle で返す。
        キューに乗せない場合（無音・エラー）も、id 付きなら必ず応答を返して
        hub 側のタイムアウト待ちを発生させない。

        Args:
            websocket: 応答送信用の WebSocket
            request: JSON-RPC 2.0リクエスト
        """
        import time
        import base64
        import numpy as np

        request_id = request.get('id')  # None なら旧ブロードキャスト（互換）
        params = request.get('params', {})

        # 必須パラメータの検証
        if 'speaker' not in params or 'audio_data' not in params:
            self.logger.warning("必須パラメータが不足しています")
            if request_id is not None:
                await self._send_error_response(websocket, request_id, -32602,
                                                "required parameter 'speaker' or 'audio_data' is missing")
            return
        
        try:
            speaker = params['speaker']
            audio_data_b64 = params['audio_data']
            sample_rate = params.get('sample_rate', 16000)
            
            # Base64デコード → PCM16LE → float32
            pcm_bytes = base64.b64decode(audio_data_b64)
            pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_f32 = pcm16.astype(np.float32) / 32767.0
            
            duration = len(audio_f32) / sample_rate
            
            # 短すぎる音声を無視（無音等は text:"" の正常レスポンス: protocol.md § 4.4）
            if duration < 0.5:
                self.logger.info(f"音声が短すぎるためキュー追加をスキップ: speaker={speaker}, duration={duration:.2f}s")
                if request_id is not None:
                    await self._send_recognition_response(websocket, request_id, "", speaker)
                return
            
            # キューに追加（優先度: 1=ネットワーク）
            priority = 1  # PRIORITY_NETWORK
            timestamp = time.time()
            request_data = {
                'speaker': speaker,
                'audio_data': audio_f32,
                'request_id': request_id,
            }
            
            # キューが満杯の場合の処理
            if self.recognition_queue.full():
                self.logger.warning(f"キューが満杯です。リクエストを破棄します: speaker={speaker}")
                if request_id is not None:
                    await self._send_error_response(websocket, request_id, -32000, "recognition queue is full")
                return
            
            await self.recognition_queue.put((priority, timestamp, request_data))
            queue_size = self.recognition_queue.qsize()
            self.logger.info(f"キューに追加: speaker={speaker}, duration={duration:.2f}s, キュー={queue_size}")
            
        except Exception as e:
            self.logger.error(f"キュー追加エラー: {e}", exc_info=True)
            if request_id is not None:
                try:
                    await self._send_error_response(websocket, request_id, -32000, str(e))
                except Exception:
                    pass

    async def _send_recognition_response(self, websocket, request_id, text: str, speaker: str):
        """recognize_audio の正常レスポンス（protocol.md § 4.4）を送信する。"""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"text": text, "speaker": speaker, "language": "ja"},
        }
        await websocket.send(json.dumps(response, ensure_ascii=False))

    async def _send_error_response(self, websocket, request_id, code: int, message: str):
        """JSON-RPC エラーレスポンスを送信する。"""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        await websocket.send(json.dumps(response, ensure_ascii=False))
    
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

