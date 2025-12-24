"""
JSON-RPC 2.0 ハンドラー

zagaroidからのJSON-RPC 2.0リクエストを処理します。
"""

import logging
import base64
import numpy as np
from typing import Dict, Any, Optional


class JSONRPCHandler:
    """JSON-RPC 2.0 リクエスト/レスポンスハンドラー"""
    
    def __init__(self, model, config):
        """
        初期化
        
        Args:
            model: Whisperモデル
            config: 設定オブジェクト
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def is_jsonrpc_request(self, data: Dict) -> bool:
        """
        JSON-RPC 2.0リクエストかどうかを判定
        
        Args:
            data: パース済みJSONデータ
            
        Returns:
            bool: JSON-RPC 2.0形式の場合True
        """
        return (
            isinstance(data, dict) and
            data.get('jsonrpc') == '2.0' and
            'method' in data
        )
    
    async def handle_request(self, request: Dict) -> Optional[Dict]:
        """
        JSON-RPC 2.0リクエストを処理
        
        Args:
            request: JSON-RPC 2.0リクエスト
            
        Returns:
            Dict: 通知を送信する場合はメッセージ、それ以外はNone
        """
        method = request.get('method')
        params = request.get('params', {})
        
        try:
            # メソッドの検証
            if method == 'recognize_audio':
                # 音声認識は非同期処理なので、通知を返す
                return await self._handle_recognize_audio(params)
            elif method == 'notifications/subtitle':
                # 他のAIからの字幕通知は無視（処理不要）
                self.logger.debug(f"字幕通知を受信（無視）: {params.get('text', '')[:50]}")
                return None
            else:
                self.logger.warning(f"未対応のメソッド: {method}")
                return None
                
        except ValueError as e:
            self.logger.error(f"パラメータエラー: {e}")
            return None
        except Exception as e:
            self.logger.error(f"JSON-RPC処理エラー: {e}", exc_info=True)
            return None
    
    async def _handle_recognize_audio(self, params: Dict) -> Optional[Dict]:
        """
        recognize_audio メソッドの処理
        
        Args:
            params: パラメータ
                - speaker (str): 話者名
                - audio_data (str): Base64エンコードされたPCM16LE
                - sample_rate (int): サンプリングレート
                - channels (int): チャンネル数
                - format (str): フォーマット（pcm16le）
                
        Returns:
            Dict: 通知メッセージ、または認識結果がない場合はNone
        """
        # 必須パラメータの検証
        if 'speaker' not in params:
            raise ValueError("Required parameter 'speaker' is missing")
        if 'audio_data' not in params:
            raise ValueError("Required parameter 'audio_data' is missing")
        
        speaker = params['speaker']
        audio_data_b64 = params['audio_data']
        sample_rate = params.get('sample_rate', 16000)
        channels = params.get('channels', 1)
        audio_format = params.get('format', 'pcm16le')
        
        # フォーマット検証
        if audio_format != 'pcm16le':
            raise ValueError(f"Unsupported audio format: {audio_format}")
        
        self.logger.info(f"音声認識リクエスト受信: speaker={speaker}, sample_rate={sample_rate}")
        
        # 処理時間計測開始
        import time
        start_time = time.time()
        processed_result = ""  # デフォルトは空文字列
        
        try:
            # Base64デコード
            pcm_bytes = base64.b64decode(audio_data_b64)
            
            # PCM16LE → float32 変換
            pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_f32 = pcm16.astype(np.float32) / 32767.0
            
            duration = len(audio_f32) / sample_rate
            self.logger.debug(f"音声データ変換完了: samples={len(audio_f32)}, duration={duration:.2f}s")
            
            # 短すぎる音声を無視（ハルシネーション対策）
            if duration < 0.5:
                self.logger.info(f"音声が短すぎるため空文字列として送信: speaker={speaker}, duration={duration:.2f}s")
            else:
                # 音声認識実行
                result = self.model.transcribe_audio_segment(audio_f32)
                
                if not result or not result.strip():
                    self.logger.info(f"認識結果なし（空文字列を返す）: speaker={speaker}")
                else:
                    # 文章整形（フィルタリングは呼び出し側で行う）
                    processed_result = result.strip()
                    
                    # 文章途中の「。」を「、」に変換（末尾以外）
                    if len(processed_result) > 1:
                        processed_result = processed_result[:-1].replace('。', '、') + processed_result[-1]
                    # 末尾の句読点を削除
                    processed_result = processed_result.rstrip('。')
                    
                    # 処理時間計測
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"認識成功: speaker={speaker}, text={processed_result}, 処理時間={elapsed_time:.2f}秒")
            
        except Exception as e:
            self.logger.error(f"音声認識エラー（空文字列として送信）: {e}", exc_info=True)
            processed_result = ""
        
        # 常に通知メッセージを構築（空文字列も含む）
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
        
        return notification

