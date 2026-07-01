"""
リアルタイム音声認識モジュール
マイクからの音声入力をリアルタイムで認識
"""

import numpy as np
import threading
import time
import queue
import collections
import logging
import asyncio
from typing import Optional, Callable, List
import sys


class RealtimeTranscriber:
    """リアルタイム音声認識クラス"""
    
    def __init__(self, config, model, show_level: bool = False, recognition_queue=None):
        """
        Args:
            config: WhisperConfig設定オブジェクト
            model: Whisperモデル
            show_level: 音声レベル表示の有効/無効
            recognition_queue: 音声認識キュー（優先度付き）
        """
        self.config = config
        self.model = model
        self.show_level = show_level
        self.recognition_queue = recognition_queue
        self.logger = logging.getLogger(__name__)
        
        # 音声入力設定
        self.chunk_size = self.config.chunk_size
        self.channels = 1
        self.rate = self.config.sample_rate
        
        # 状態管理
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Silero VAD設定
        try:
            import torch
            # torch.hubから直接ロード（日本語パス対応）
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True
            )
            self.vad_available = True
            self.logger.info(f"Silero VAD初期化完了")
        except Exception as e:
            self.logger.warning(f"Silero VADの初期化に失敗しました。VADなしで動作します。")
            self.vad_available = False
        
        self.frame_duration = self.config.frame_duration_ms  # ms
        self.frame_size = int(self.rate * self.frame_duration / 1000)
        
        # 先頭/末尾パディング用バッファ
        self.pre_speech_samples = int(self.rate * self.config.pre_speech_padding_ms / 1000)
        self.post_speech_samples = int(self.rate * self.config.post_speech_padding_ms / 1000)
        self.pre_speech_buffer = collections.deque(maxlen=max(1, self.pre_speech_samples))
        
        # 音声バッファ
        self.audio_buffer = collections.deque(maxlen=int(self.rate * self.config.max_duration))
        
        # 音声検出の状態
        self.in_speech = False
        self.silence_counter = 0
        self.silence_threshold = int(self.config.pause_threshold * self.rate / self.frame_size)
        
        # sounddeviceの遅延インポート
        self.sd = None
        
        # 停止イベント（非同期実行用）
        self.stop_event = None
        
    def _get_sounddevice(self):
        """sounddeviceの遅延インポート"""
        if self.sd is None:
            import sounddevice as sd
            self.sd = sd
        return self.sd
    
    def list_microphones(self) -> List[dict]:
        """利用可能なマイクデバイスをリスト"""
        sd = self._get_sounddevice()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        # Windowsでは複数のオーディオホストAPI（MME/DirectSound/WASAPI/WDM-KS）
        # 経由で同じ物理デバイスが重複して列挙されるため、WASAPIのみに絞り込む
        preferred_hostapi_index = None
        if sys.platform == 'win32':
            for idx, api in enumerate(hostapis):
                if 'WASAPI' in api['name']:
                    preferred_hostapi_index = idx
                    break

        microphones = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if preferred_hostapi_index is not None and device['hostapi'] != preferred_hostapi_index:
                    continue
                microphones.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })
        
        return microphones
    
    def select_microphone(self) -> Optional[int]:
        """マイクデバイスを選択"""
        microphones = self.list_microphones()
        
        if not microphones:
            self.logger.error("利用可能なマイクデバイスが見つかりません")
            return None
        
        print("\n利用可能なマイクデバイス:")
        for mic in microphones:
            print(f"  [{mic['id']}] {mic['name']} ({mic['channels']}ch, {mic['default_samplerate']:.0f}Hz)")
        
        while True:
            try:
                device_id = input(f"\nマイクデバイスIDを入力してください (デフォルト: 0): ").strip()
                if not device_id:
                    return 0
                device_id = int(device_id)
                if any(mic['id'] == device_id for mic in microphones):
                    return device_id
                print("無効なデバイスIDです。もう一度入力してください。")
            except ValueError:
                print("数値を入力してください。")
            except KeyboardInterrupt:
                print("\nキャンセルしました")
                return None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """sounddeviceのコールバック"""
        if status:
            self.logger.warning(f"オーディオステータス: {status}")
        
        # モノラルに変換
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        
        # キューに追加
        self.audio_queue.put(audio_data.copy())
    
    def _process_audio(self):
        """音声処理スレッド"""
        self.logger.info("音声処理スレッドを開始しました")
        
        while self.is_recording:
            try:
                # キューから音声データを取得（タイムアウト付き）
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # 音声レベル表示
                if self.show_level:
                    level = np.abs(audio_chunk).mean()
                    bar_length = int(level * 50)
                    bar = '█' * bar_length
                    print(f'\r音声レベル: [{bar:<50}] {level:.3f}', end='', flush=True)
                
                # 音声バッファに追加
                self.audio_buffer.extend(audio_chunk)
                self.pre_speech_buffer.extend(audio_chunk)
                
                # VADで音声検出
                # Silero VADは最低512サンプル（約32ms）必要
                min_vad_samples = 512
                if self.vad_available and len(self.audio_buffer) >= min_vad_samples:
                    # 現在のバッファから音声の有無を判定
                    # 最低サンプル数を確保
                    vad_window_size = max(self.frame_size, min_vad_samples)
                    buffer_array = np.array(list(self.audio_buffer)[-vad_window_size:], dtype=np.float32)
                    
                    # VADで音声検出
                    import torch
                    audio_tensor = torch.from_numpy(buffer_array).unsqueeze(0)
                    speech_prob = self.vad_model(audio_tensor, self.rate).item()
                    
                    is_speech = speech_prob > self.config.vad_threshold
                    
                    if is_speech:
                        if not self.in_speech:
                            # 音声開始
                            self.in_speech = True
                            self.silence_counter = 0
                            self.logger.debug("音声検出開始")
                    else:
                        if self.in_speech:
                            self.silence_counter += 1
                            
                            # 無音が一定時間続いたら認識実行
                            if self.silence_counter >= self.silence_threshold:
                                self._recognize_audio()
                                self.in_speech = False
                                self.silence_counter = 0
                                self.audio_buffer.clear()
                                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"音声処理エラー: {e}", exc_info=True)
        
        self.logger.info("音声処理スレッドを終了しました")
    
    def _recognize_audio(self):
        """音声認識リクエストをキューに追加"""
        if len(self.audio_buffer) < self.rate * self.config.phrase_threshold:
            self.logger.debug("音声が短すぎるためスキップ")
            return
        
        try:
            # バッファから音声データを取得
            audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
            
            # 音声レベルチェック
            audio_level = np.abs(audio_data).mean()
            if audio_level < self.config.min_audio_level:
                self.logger.debug(f"音声レベルが低すぎるためスキップ: {audio_level:.4f}")
                return
            
            if self.show_level:
                print()  # 改行
            
            duration = len(audio_data) / self.rate
            self.logger.info(f"マイク音声検出 ({duration:.1f}秒)")
            
            # キューに追加
            if self.recognition_queue:
                # 非同期でキューに追加（同期コンテキストから）
                import time
                priority = 0  # PRIORITY_MICROPHONE（最優先）
                timestamp = time.time()
                request_data = {
                    'speaker': self.config.microphone_speaker,
                    'audio_data': audio_data
                }
                
                # asyncio.run()ではなく、キューに直接追加
                # （スレッドセーフな方法で追加）
                try:
                    self.recognition_queue.put_nowait((priority, timestamp, request_data))
                    queue_size = self.recognition_queue.qsize()
                    self.logger.info(f"キューに追加: speaker={self.config.microphone_speaker}, "
                                   f"duration={duration:.1f}秒, キュー={queue_size}")
                except asyncio.QueueFull:
                    self.logger.warning("キューが満杯です。マイク入力を破棄します")
            else:
                # キューがない場合は従来通り処理（後方互換性）
                self.logger.warning("キューが設定されていません。認識をスキップします")
                
        except Exception as e:
            self.logger.error(f"キュー追加エラー: {e}", exc_info=True)
    
    
    async def start_async(self, device_id: int, stop_event: Optional[asyncio.Event] = None):
        """音声認識を開始
        
        Args:
            device_id: マイクデバイスID（必須）
            stop_event: 停止イベント（Noneの場合は内部で作成）
        """
        sd = self._get_sounddevice()
        
        # 停止イベントの設定
        if stop_event is None:
            self.stop_event = asyncio.Event()
        else:
            self.stop_event = stop_event
        
        self.logger.info(f"マイクデバイス: {device_id}")
        
        # デバイス情報表示
        device_info = sd.query_devices(device_id)
        self.logger.info(f"デバイス名: {device_info['name']}")
        self.logger.info(f"サンプリングレート: {self.rate}Hz")
        
        # 音声処理スレッド開始
        self.is_recording = True
        process_thread = threading.Thread(target=self._process_audio, daemon=True)
        process_thread.start()
        
        # 音声入力開始
        try:
            print("\n" + "="*60)
            print("🎤 リアルタイム音声認識を開始します")
            print("="*60)
            print()
            
            with sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                # 停止イベントを待機
                while self.is_recording and not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"音声入力エラー: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """音声認識を停止"""
        self.logger.info("音声認識を停止中...")
        self.is_recording = False
        time.sleep(0.5)  # スレッド終了を待つ
        self.logger.info("音声認識を停止しました")

