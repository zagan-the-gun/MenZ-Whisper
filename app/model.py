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
        
        # モデルロード
        self.model = FasterWhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(cache_dir),
            cpu_threads=self.config.cpu_threads if device == "cpu" else 0
        )
    
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
        
        # セグメントを結合
        text = " ".join([segment.text for segment in segments]).strip()
        
        if self.config.show_debug:
            self.logger.debug(f"認識言語: {info.language} (確率: {info.language_probability:.2f})")
        
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

