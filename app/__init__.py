"""
MenZ-Whisper
OpenAI Whisperベースの音声認識システム
"""

__version__ = "1.0.0"
__author__ = "MenZ Project"

from .config import WhisperConfig
from .model import WhisperModel

__all__ = ['WhisperConfig', 'WhisperModel']

