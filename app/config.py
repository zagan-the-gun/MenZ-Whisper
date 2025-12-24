"""
設定管理モジュール
config.iniファイルから設定を読み込み、モデルやクライアントに提供します。
"""

import configparser
import os
from pathlib import Path
from typing import Optional


class WhisperConfig:
    """Whisper音声認識システムの設定を管理するクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        設定を初期化
        
        Args:
            config_path: 設定ファイルのパス（デフォルト: config.ini）
        """
        self.config_path = config_path or "config.ini"
        self.config = configparser.ConfigParser()
        
        # デフォルト値を設定
        self._set_defaults()
        
        # 設定ファイルが存在すれば読み込み
        if os.path.exists(self.config_path):
            self.config.read(self.config_path, encoding='utf-8')
    
    def _set_defaults(self):
        """デフォルト設定値を設定"""
        # [mode]
        self.config['mode'] = {
            'enable_network': 'true',
            'enable_microphone': 'false'
        }
        
        # [microphone]
        self.config['microphone'] = {
            'device_id': '0',
            'speaker': 'whisper_user',
            'show_level': 'true',
            'send_to_websocket': 'false'
        }
        
        # [model]
        self.config['model'] = {
            'model_size': 'base',
            'cache_dir': './models',
            'use_faster_whisper': 'true'
        }
        
        # [audio]
        self.config['audio'] = {
            'sample_rate': '16000',
            'chunk_size': '1024',
            'frame_duration_ms': '30'
        }
        
        # [silero_vad]
        self.config['silero_vad'] = {
            'threshold': '0.05',
            'min_speech_duration_ms': '10',
            'min_silence_duration_ms': '500'
        }
        
        # [inference]
        self.config['inference'] = {
            'device': 'auto',
            'gpu_id': 'auto',
            'compute_type': 'float16',
            'cpu_threads': '0',
            'beam_size': '5'
        }
        
        # [recognizer]
        self.config['recognizer'] = {
            'min_audio_level': '0.001',
            'pause_threshold': '0.8',
            'phrase_threshold': '0.3',
            'max_duration': '30.0',
            'pre_speech_padding_ms': '300',
            'post_speech_padding_ms': '150'
        }
        
        # [whisper]
        self.config['whisper'] = {
            'language': 'ja',
            'task': 'transcribe',
            'initial_prompt': '',
            'vad_filter': 'true',
            'condition_on_previous_text': 'true',
            'temperature': '0.0'
        }
        
        # [websocket]
        self.config['websocket'] = {
            'host': '127.0.0.1',
            'port': '50001'
        }
        
        # [filtering]
        self.config['filtering'] = {
            'min_length': '2',
            'exclude_whitespace_only': 'true'
        }
        
        # [debug]
        self.config['debug'] = {
            'show_debug': 'true',
            'show_transcription': 'true'
        }
        
        # [queue]
        self.config['queue'] = {
            'max_size': '100'
        }
    
    @classmethod
    def from_ini(cls, config_path: str) -> 'WhisperConfig':
        """
        INIファイルから設定を読み込み
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            WhisperConfig: 設定オブジェクト
        """
        return cls(config_path)
    
    # Mode settings
    @property
    def enable_network(self) -> bool:
        return self.config.getboolean('mode', 'enable_network')
    
    @property
    def enable_microphone(self) -> bool:
        return self.config.getboolean('mode', 'enable_microphone')
    
    # Microphone settings
    @property
    def microphone_device_id(self) -> str:
        return self.config.get('microphone', 'device_id')
    
    @property
    def microphone_speaker(self) -> str:
        return self.config.get('microphone', 'speaker')
    
    @property
    def microphone_show_level(self) -> bool:
        return self.config.getboolean('microphone', 'show_level')
    
    @property
    def microphone_send_to_websocket(self) -> bool:
        return self.config.getboolean('microphone', 'send_to_websocket')
    
    # Model settings
    @property
    def model_size(self) -> str:
        return self.config.get('model', 'model_size')
    
    @property
    def cache_dir(self) -> str:
        return self.config.get('model', 'cache_dir')
    
    @property
    def use_faster_whisper(self) -> bool:
        return self.config.getboolean('model', 'use_faster_whisper')
    
    # Audio settings
    @property
    def sample_rate(self) -> int:
        return self.config.getint('audio', 'sample_rate')
    
    @property
    def chunk_size(self) -> int:
        return self.config.getint('audio', 'chunk_size')
    
    @property
    def frame_duration_ms(self) -> int:
        return self.config.getint('audio', 'frame_duration_ms')
    
    # Silero VAD settings
    @property
    def vad_threshold(self) -> float:
        return self.config.getfloat('silero_vad', 'threshold')
    
    @property
    def min_speech_duration_ms(self) -> int:
        return self.config.getint('silero_vad', 'min_speech_duration_ms')
    
    @property
    def min_silence_duration_ms(self) -> int:
        return self.config.getint('silero_vad', 'min_silence_duration_ms')
    
    # Inference settings
    @property
    def device(self) -> str:
        return self.config.get('inference', 'device')
    
    @device.setter
    def device(self, value: str):
        self.config.set('inference', 'device', value)
    
    @property
    def gpu_id(self) -> str:
        return self.config.get('inference', 'gpu_id')
    
    @property
    def compute_type(self) -> str:
        return self.config.get('inference', 'compute_type')
    
    @property
    def cpu_threads(self) -> int:
        return self.config.getint('inference', 'cpu_threads')
    
    @property
    def beam_size(self) -> int:
        return self.config.getint('inference', 'beam_size')
    
    # Recognizer settings
    @property
    def min_audio_level(self) -> float:
        return self.config.getfloat('recognizer', 'min_audio_level')
    
    @property
    def pause_threshold(self) -> float:
        return self.config.getfloat('recognizer', 'pause_threshold')
    
    @property
    def phrase_threshold(self) -> float:
        return self.config.getfloat('recognizer', 'phrase_threshold')
    
    @property
    def max_duration(self) -> float:
        return self.config.getfloat('recognizer', 'max_duration')
    
    @property
    def pre_speech_padding_ms(self) -> int:
        return self.config.getint('recognizer', 'pre_speech_padding_ms')
    
    @property
    def post_speech_padding_ms(self) -> int:
        return self.config.getint('recognizer', 'post_speech_padding_ms')
    
    # Whisper settings
    @property
    def language(self) -> str:
        lang = self.config.get('whisper', 'language')
        return None if lang == 'auto' else lang
    
    @property
    def task(self) -> str:
        return self.config.get('whisper', 'task')
    
    @property
    def initial_prompt(self) -> Optional[str]:
        prompt = self.config.get('whisper', 'initial_prompt')
        return prompt if prompt else None
    
    @property
    def vad_filter(self) -> bool:
        return self.config.getboolean('whisper', 'vad_filter')
    
    @property
    def condition_on_previous_text(self) -> bool:
        return self.config.getboolean('whisper', 'condition_on_previous_text')
    
    @property
    def temperature(self) -> float:
        return self.config.getfloat('whisper', 'temperature')
    
    # WebSocket settings
    @property
    def websocket_host(self) -> str:
        return self.config.get('websocket', 'host')
    
    @property
    def websocket_port(self) -> int:
        return self.config.getint('websocket', 'port')
    
    # Filtering settings
    @property
    def min_length(self) -> int:
        return self.config.getint('filtering', 'min_length')
    
    @property
    def exclude_whitespace_only(self) -> bool:
        return self.config.getboolean('filtering', 'exclude_whitespace_only')
    
    # Debug settings
    @property
    def show_debug(self) -> bool:
        return self.config.getboolean('debug', 'show_debug')
    
    @property
    def show_transcription(self) -> bool:
        return self.config.getboolean('debug', 'show_transcription')
    
    # Queue settings
    @property
    def queue_max_size(self) -> int:
        return self.config.getint('queue', 'max_size')
    
    def save(self, config_path: Optional[str] = None):
        """
        設定をファイルに保存
        
        Args:
            config_path: 保存先パス（デフォルト: 読み込み元と同じ）
        """
        save_path = config_path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            self.config.write(f)

