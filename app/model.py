"""
Whisper音声認識モデルモジュール
OpenAI WhisperとFaster-Whisperの両方をサポート
"""

import logging
import numpy as np
from typing import Optional, Union
from pathlib import Path

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class WhisperModel:
    """Whisper音声認識モデルのラッパークラス"""
    
    def __init__(self, config):
        """
        モデルを初期化
        
        Args:
            config: WhisperConfig設定オブジェクト
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.use_faster_whisper = config.use_faster_whisper and FASTER_WHISPER_AVAILABLE
        
        # faster-whisperが要求されているが利用不可の場合
        if config.use_faster_whisper and not FASTER_WHISPER_AVAILABLE:
            self.logger.warning(
                "faster-whisper が要求されていますが、インストールされていません。"
                "標準のwhisperを使用します。"
            )
            self.use_faster_whisper = False
        
        # どちらも利用不可の場合
        if not FASTER_WHISPER_AVAILABLE and not WHISPER_AVAILABLE:
            raise RuntimeError(
                "whisper または faster-whisper がインストールされていません。"
                "pip install openai-whisper または pip install faster-whisper を実行してください。"
            )
        
        self._load_model()
    
    def _load_model(self):
        """モデルをロード"""
        model_size = self.config.model_size
        device = self._get_device()
        
        self.logger.info(f"Whisperモデルをロード中: {model_size}")
        self.logger.info(f"使用エンジン: {'faster-whisper' if self.use_faster_whisper else 'openai-whisper'}")
        self.logger.info(f"デバイス: {device}")
        
        if self.use_faster_whisper:
            self._load_faster_whisper(model_size, device)
        else:
            self._load_openai_whisper(model_size, device)
        
        self.logger.info("モデルのロードが完了しました")
    
    def _get_device(self) -> str:
        """使用するデバイスを取得"""
        device = self.config.device
        
        if device == "auto":
            # デバイスの自動選択
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps" if not self.use_faster_whisper else "cpu"  # faster-whisperはMPS未対応
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
            
            self.logger.info(f"自動デバイス選択: {device}")
        
        return device
    
    def _load_faster_whisper(self, model_size: str, device: str):
        """faster-whisperモデルをロード"""
        # faster-whisperではMPSは未対応なのでCPUにフォールバック
        if device == "mps":
            self.logger.warning("faster-whisper は MPS をサポートしていません。CPUを使用します。")
            device = "cpu"
        
        # キャッシュディレクトリの作成
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # compute_typeの設定
        compute_type = self.config.compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"  # CPUではfloat16は使えない
            self.logger.info(f"CPU使用のため compute_type を {compute_type} に変更")
        
        # GPU IDの設定と検証
        device_index = 0
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA設定ですがGPUが利用できません")
            
            gpu_count = torch.cuda.device_count()
            gpu_id = self.config.gpu_id
            
            if gpu_id == "auto":
                device_index = 0
            else:
                try:
                    device_index = int(gpu_id)
                    if device_index >= gpu_count or device_index < 0:
                        raise ValueError(f"GPU ID {device_index} は存在しません（利用可能: 0-{gpu_count-1}）")
                except ValueError as e:
                    raise ValueError(f"無効なGPU ID設定: {gpu_id} - {e}")
        
        # モデルロード
        self.model = FasterWhisperModel(
            model_size,
            device=device,
            device_index=device_index if device == "cuda" else 0,
            compute_type=compute_type,
            download_root=str(cache_dir),
            cpu_threads=self.config.cpu_threads if device == "cpu" else 0
        )
        
        # 実際に使用されているデバイスを表示
        self._log_device_info(device, device_index)
    
    def _load_openai_whisper(self, model_size: str, device: str):
        """OpenAI Whisperモデルをロード"""
        # キャッシュディレクトリの作成
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルロード
        self.model = whisper.load_model(
            model_size,
            device=device,
            download_root=str(cache_dir)
        )
        
        # 実際に使用されているデバイスを表示
        self._log_device_info(device, 0)
    
    def _log_device_info(self, device: str, device_index: int):
        """実際に使用されているデバイス情報を表示"""
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    # 実際に使用されているGPUを取得
                    current_device = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(current_device)
                    gpu_props = torch.cuda.get_device_properties(current_device)
                    gpu_memory_gb = gpu_props.total_memory / (1024**3)
                    self.logger.info(f"使用GPU: GPU {current_device} - {gpu_name} ({gpu_memory_gb:.1f}GB)")
                else:
                    self.logger.warning("CUDA使用設定ですがGPUが利用できません")
            except Exception as e:
                self.logger.warning(f"GPU情報の取得に失敗: {e}")
        elif device == "mps":
            self.logger.info("使用デバイス: Apple Metal (MPS)")
        else:
            self.logger.info("使用デバイス: CPU")
    
    def transcribe_audio_segment(self, audio: np.ndarray) -> str:
        """
        音声セグメントを認識
        
        Args:
            audio: float32形式の音声データ（-1.0 〜 1.0）
            
        Returns:
            str: 認識結果テキスト
        """
        try:
            if self.use_faster_whisper:
                return self._transcribe_faster_whisper(audio)
            else:
                return self._transcribe_openai_whisper(audio)
        except Exception as e:
            self.logger.error(f"音声認識エラー: {e}", exc_info=True)
            return ""
    
    def _transcribe_faster_whisper(self, audio: np.ndarray) -> str:
        """faster-whisperで音声認識"""
        original_duration = len(audio) / 16000.0  # 元の音声長
        
        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            task=self.config.task,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            vad_parameters={
                "threshold": self.config.vad_threshold,
                "min_speech_duration_ms": self.config.min_speech_duration_ms,
                "min_silence_duration_ms": self.config.min_silence_duration_ms,
            },
            initial_prompt=self.config.initial_prompt,
            condition_on_previous_text=self.config.condition_on_previous_text,
            temperature=self.config.temperature if self.config.temperature > 0 else 0,
        )
        
        # セグメントをリスト化（検証のため）
        segment_list = list(segments)
        
        # VADで全削除された場合は空文字を返す（ハルシネーション対策）
        if len(segment_list) == 0:
            if self.config.show_debug:
                self.logger.debug("VADで全セグメント削除（無音と判定）")
            return ""
        
        # VADで大部分が削除された場合もスキップ（ハルシネーション対策）
        # info.duration_after_vad は faster-whisper 1.0+ で利用可能
        try:
            if hasattr(info, 'duration_after_vad') and info.duration_after_vad is not None:
                vad_duration = info.duration_after_vad
                if original_duration > 0.5 and vad_duration < 0.3:
                    # 元が0.5秒以上あるのにVAD後が0.3秒未満 → ほぼ無音
                    if self.config.show_debug:
                        self.logger.debug(f"VADで大部分削除（{original_duration:.2f}s → {vad_duration:.2f}s）")
                    return ""
        except Exception:
            pass
        
        # セグメントを結合
        text = " ".join([segment.text for segment in segment_list]).strip()
        
        # ループ検出（同じフレーズが3回以上繰り返されている）
        if text and len(text) > 50:
            # 最初の20文字を取得
            first_phrase = text[:20]
            # 全体に何回出現するか
            count = text.count(first_phrase)
            if count >= 3:
                if self.config.show_debug:
                    self.logger.warning(f"ハルシネーション検出（ループ）: {first_phrase}... が{count}回繰り返し")
                return ""
        
        if self.config.show_debug:
            self.logger.debug(f"認識言語: {info.language} (確率: {info.language_probability:.2f})")
            self.logger.debug(f"セグメント数: {len(segment_list)}")
        
        return text
    
    def _transcribe_openai_whisper(self, audio: np.ndarray) -> str:
        """OpenAI Whisperで音声認識"""
        result = self.model.transcribe(
            audio,
            language=self.config.language,
            task=self.config.task,
            beam_size=self.config.beam_size,
            initial_prompt=self.config.initial_prompt,
            condition_on_previous_text=self.config.condition_on_previous_text,
            temperature=self.config.temperature if self.config.temperature > 0 else 0,
        )
        
        text = result.get('text', '').strip()
        
        if self.config.show_debug and 'language' in result:
            self.logger.debug(f"認識言語: {result['language']}")
        
        return text
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        self.logger.info("モデルリソースをクリーンアップ中...")
        
        # メモリ解放
        del self.model
        self.model = None
        
        # ガベージコレクション
        import gc
        gc.collect()
        
        # CUDAキャッシュクリア
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.logger.info("クリーンアップ完了")

