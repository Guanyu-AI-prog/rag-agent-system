#!/usr/bin/env python3
"""
STT + TTS 语音交互模块 — 集成 dx_agent 套餐查询 Agent

架构：
  VoiceModule
    ├── STT (语音 → 文字): FunASR / Whisper / SiliconFlow API
    ├── TTS (文字 → 语音): Edge-TTS / CosyVoice / SiliconFlow API
    ├── 音频采集: PyAudio / sounddevice
    └── Agent 调用: dx_agent.run_single()

技术选型：
  - STT: FunASR（中文效果最佳）> Whisper > SiliconFlow STT API
  - TTS: Edge-TTS（免费、高质量中文语音）> SiliconFlow TTS API
  - 音频采集: sounddevice（跨平台，无需 PortAudio 手动编译）

用法:
  python voice_module.py              # 交互式语音模式
  python voice_module.py --text       # 文字模式（调试用）
  python voice_module.py --stt-file audio.wav  # 识别音频文件
"""

import asyncio
import io
import logging
import os
import struct
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from config import Config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 语音配置
# ═══════════════════════════════════════════════════

class STTBackend(str, Enum):
    """STT 引擎选择"""
    FUNASR = "funasr"           # 阿里 FunASR（中文最佳）
    WHISPER = "whisper"         # OpenAI Whisper
    SILICONFLOW = "siliconflow" # SiliconFlow STT API


class TTSBackend(str, Enum):
    """TTS 引擎选择"""
    EDGE_TTS = "edge-tts"       # Microsoft Edge-TTS（免费）
    SILICONFLOW = "siliconflow" # SiliconFlow TTS API
    COSYVOICE = "cosyvoice"     # 阿里 CosyVoice


@dataclass
class VoiceConfig:
    """语音模块配置，从环境变量读取"""

    # ── STT 配置 ──
    STT_BACKEND: STTBackend = STTBackend(
        os.getenv("STT_BACKEND", "funasr")
    )
    STT_MODEL: str = os.getenv("STT_MODEL", "paraformer-zh")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "zh")
    STT_SAMPLE_RATE: int = int(os.getenv("STT_SAMPLE_RATE", "16000"))
    STT_SILENCE_THRESHOLD: float = float(os.getenv("STT_SILENCE_THRESHOLD", "0.01"))
    STT_SILENCE_DURATION: float = float(os.getenv("STT_SILENCE_DURATION", "1.5"))
    STT_MAX_DURATION: int = int(os.getenv("STT_MAX_DURATION", "30"))

    # FunASR 配置
    FUNASR_MODEL_DIR: str = os.getenv("FUNASR_MODEL_DIR", "")
    FUNASR_VAD_MODEL: str = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
    FUNASR_PUNC_MODEL: str = os.getenv("FUNASR_PUNC_MODEL", "ct-punc-c")

    # Whisper 配置
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "medium")

    # SiliconFlow STT API
    STT_API_URL: str = os.getenv(
        "STT_API_URL",
        "https://api.siliconflow.cn/v1/audio/transcriptions"
    )
    STT_API_KEY: str = os.getenv("STT_API_KEY", "") or Config.SILICONFLOW_API_KEY

    # ── TTS 配置 ──
    TTS_BACKEND: TTSBackend = TTSBackend(
        os.getenv("TTS_BACKEND", "edge-tts")
    )
    TTS_VOICE: str = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    TTS_RATE: str = os.getenv("TTS_RATE", "+0%")
    TTS_VOLUME: str = os.getenv("TTS_VOLUME", "+0%")
    TTS_OUTPUT_FORMAT: str = os.getenv("TTS_OUTPUT_FORMAT", "mp3")

    # SiliconFlow TTS API
    TTS_API_URL: str = os.getenv(
        "TTS_API_URL",
        "https://api.siliconflow.cn/v1/audio/speech"
    )
    TTS_API_KEY: str = os.getenv("TTS_API_KEY", "") or Config.SILICONFLOW_API_KEY
    TTS_MODEL: str = os.getenv("TTS_MODEL", "FunAudioLLM/CosyVoice2-0.5B")

    # ── 音频采集配置 ──
    AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_SAMPLE_WIDTH: int = 2  # 16-bit
    AUDIO_BLOCK_SIZE: int = int(os.getenv("AUDIO_BLOCK_SIZE", "1024"))
    AUDIO_DEVICE_INDEX: Optional[int] = (
        int(os.getenv("AUDIO_DEVICE_INDEX", "-1"))
        if os.getenv("AUDIO_DEVICE_INDEX", "-1") != "-1"
        else None
    )

    # ── 交互配置 ──
    VOICE_LOOP_ENABLED: bool = os.getenv("VOICE_LOOP_ENABLED", "True").lower() == "true"
    AUTO_PLAY_TTS: bool = os.getenv("AUTO_PLAY_TTS", "True").lower() == "true"
    SHOW_STT_TEXT: bool = os.getenv("SHOW_STT_TEXT", "True").lower() == "true"
    WAKE_WORD: str = os.getenv("WAKE_WORD", "")  # 唤醒词，空则不需要


voice_config = VoiceConfig()


# ═══════════════════════════════════════════════════
# 异步工具函数
# ═══════════════════════════════════════════════════

def _run_async(coro):
    """在同步上下文中运行异步协程，兼容 FastAPI 已有事件循环"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已有事件循环（如 FastAPI），用 nest_asyncio 或线程池
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            # fallback: 在新线程中跑新事件循环
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
    else:
        # 无事件循环，直接跑
        return asyncio.run(coro)


# ═══════════════════════════════════════════════════
# STT 语音识别引擎
# ═══════════════════════════════════════════════════

class STTEngine:
    """STT 基类"""

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        raise NotImplementedError

    def transcribe_file(self, file_path: str) -> str:
        raise NotImplementedError


class FunASREngine(STTEngine):
    """阿里 FunASR 语音识别（中文效果最佳）

    支持两种模式：
    1. 本地模型（需要安装 funasr 包）
    2. FunASR WebSocket 服务端
    """

    def __init__(self):
        self._model = None
        self._init_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return
            try:
                from funasr import AutoModel

                model_kwargs = {
                    "model": voice_config.STT_MODEL,
                    "vad_model": voice_config.FUNASR_VAD_MODEL,
                    "punc_model": voice_config.FUNASR_PUNC_MODEL,
                    "vad_kwargs": {"max_single_segment_time": 30000},
                }
                if voice_config.FUNASR_MODEL_DIR:
                    model_kwargs["model"] = voice_config.FUNASR_MODEL_DIR

                self._model = AutoModel(**model_kwargs)
                logger.info("FunASR 模型加载完成: %s", voice_config.STT_MODEL)
            except ImportError:
                raise ImportError(
                    "请安装 FunASR: pip install funasr\n"
                    "或切换 STT_BACKEND=whisper / siliconflow"
                )

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        self._ensure_model()
        import numpy as np

        # bytes → float32 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        result = self._model.generate(
            input=audio_np,
            cache={},
            language=voice_config.STT_LANGUAGE,
            use_itn=True,
            batch_size_s=60,
        )

        if result and len(result) > 0:
            text = result[0].get("text", "")
            logger.info("FunASR 识别结果: %s", text[:50])
            return text
        return ""

    def transcribe_file(self, file_path: str) -> str:
        self._ensure_model()
        result = self._model.generate(
            input=file_path,
            cache={},
            language=voice_config.STT_LANGUAGE,
            use_itn=True,
        )
        if result and len(result) > 0:
            return result[0].get("text", "")
        return ""


class WhisperEngine(STTEngine):
    """OpenAI Whisper 语音识别"""

    def __init__(self):
        self._model = None
        self._init_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return
            try:
                import whisper

                self._model = whisper.load_model(voice_config.WHISPER_MODEL_SIZE)
                logger.info("Whisper 模型加载完成: %s", voice_config.WHISPER_MODEL_SIZE)
            except ImportError:
                raise ImportError(
                    "请安装 Whisper: pip install openai-whisper\n"
                    "或切换 STT_BACKEND=funasr / siliconflow"
                )

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        self._ensure_model()
        import numpy as np
        import tempfile

        # 写入临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f, "wb") as wf:
                wf.setnchannels(voice_config.AUDIO_CHANNELS)
                wf.setsampwidth(voice_config.AUDIO_SAMPLE_WIDTH)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            tmp_path = f.name

        try:
            result = self._model.transcribe(
                tmp_path,
                language=voice_config.STT_LANGUAGE,
                fp16=False,
            )
            text = result.get("text", "").strip()
            logger.info("Whisper 识别结果: %s", text[:50])
            return text
        finally:
            os.unlink(tmp_path)

    def transcribe_file(self, file_path: str) -> str:
        self._ensure_model()
        result = self._model.transcribe(
            file_path,
            language=voice_config.STT_LANGUAGE,
            fp16=False,
        )
        return result.get("text", "").strip()


class SiliconFlowSTTEngine(STTEngine):
    """SiliconFlow STT API 识别"""

    def __init__(self):
        self._api_key = voice_config.STT_API_KEY
        self._api_url = voice_config.STT_API_URL

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        import requests

        # 构建 WAV 文件
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(voice_config.AUDIO_CHANNELS)
            wf.setsampwidth(voice_config.AUDIO_SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        wav_buffer.seek(0)

        headers = {"Authorization": f"Bearer {self._api_key}"}
        files = {
            "file": ("audio.wav", wav_buffer, "audio/wav"),
        }
        data = {
            "model": voice_config.STT_MODEL,
            "language": voice_config.STT_LANGUAGE,
        }

        try:
            resp = requests.post(
                self._api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            text = result.get("text", "")
            logger.info("SiliconFlow STT 识别结果: %s", text[:50])
            return text
        except Exception as e:
            logger.error("SiliconFlow STT 调用失败: %s", e)
            raise

    def transcribe_file(self, file_path: str) -> str:
        import requests

        headers = {"Authorization": f"Bearer {self._api_key}"}
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/wav")}
            data = {
                "model": voice_config.STT_MODEL,
                "language": voice_config.STT_LANGUAGE,
            }
            resp = requests.post(
                self._api_url, headers=headers, files=files, data=data, timeout=30
            )
            resp.raise_for_status()
            return resp.json().get("text", "")


def create_stt_engine() -> STTEngine:
    """根据配置创建 STT 引擎"""
    backend = voice_config.STT_BACKEND
    if backend == STTBackend.FUNASR:
        return FunASREngine()
    elif backend == STTBackend.WHISPER:
        return WhisperEngine()
    elif backend == STTBackend.SILICONFLOW:
        return SiliconFlowSTTEngine()
    else:
        raise ValueError(f"不支持的 STT 后端: {backend}")


# ═══════════════════════════════════════════════════
# TTS 语音合成引擎
# ═══════════════════════════════════════════════════

class TTSEngine:
    """TTS 基类"""

    def synthesize(self, text: str) -> bytes:
        """合成语音，返回音频 bytes"""
        raise NotImplementedError

    async def synthesize_async(self, text: str) -> bytes:
        """异步合成语音"""
        raise NotImplementedError

    def synthesize_to_file(self, text: str, file_path: str) -> str:
        """合成并保存到文件，返回文件路径"""
        raise NotImplementedError


class EdgeTTSEngine(TTSEngine):
    """Microsoft Edge-TTS 语音合成（免费、高质量中文语音）

    可用中文语音：
    - zh-CN-XiaoxiaoNeural   (女声，温柔)
    - zh-CN-YunxiNeural      (男声，沉稳)
    - zh-CN-YunyangNeural    (男声，新闻播报)
    - zh-CN-XiaoyiNeural     (女声，活泼)
    - zh-CN-XiaochenNeural   (女声，甜美)
    - zh-CN-YunjianNeural    (男声，有力)
    """

    def __init__(self):
        self._voice = voice_config.TTS_VOICE
        self._rate = voice_config.TTS_RATE
        self._volume = voice_config.TTS_VOLUME

    async def synthesize_async(self, text: str) -> bytes:
        try:
            import edge_tts
        except ImportError:
            raise ImportError("请安装 edge-tts: pip install edge-tts")

        communicate = edge_tts.Communicate(
            text,
            voice=self._voice,
            rate=self._rate,
            volume=self._volume,
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_data = audio_buffer.getvalue()
        logger.info("Edge-TTS 合成完成: %d bytes, voice=%s", len(audio_data), self._voice)
        return audio_data

    def synthesize(self, text: str) -> bytes:
        return _run_async(self.synthesize_async(text))

    async def synthesize_to_file_async(self, text: str, file_path: str) -> str:
        try:
            import edge_tts
        except ImportError:
            raise ImportError("请安装 edge-tts: pip install edge-tts")

        communicate = edge_tts.Communicate(
            text,
            voice=self._voice,
            rate=self._rate,
            volume=self._volume,
        )
        await communicate.save(file_path)
        logger.info("Edge-TTS 已保存: %s", file_path)
        return file_path

    def synthesize_to_file(self, text: str, file_path: str) -> str:
        return _run_async(self.synthesize_to_file_async(text, file_path))


class SiliconFlowTTSEngine(TTSEngine):
    """SiliconFlow TTS API 语音合成"""

    def __init__(self):
        self._api_key = voice_config.TTS_API_KEY
        self._api_url = voice_config.TTS_API_URL
        self._model = voice_config.TTS_MODEL
        self._voice = voice_config.TTS_VOICE

    def synthesize(self, text: str) -> bytes:
        import requests

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": voice_config.TTS_OUTPUT_FORMAT,
        }

        try:
            resp = requests.post(
                self._api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            audio_data = resp.content
            logger.info(
                "SiliconFlow TTS 合成完成: %d bytes, model=%s",
                len(audio_data),
                self._model,
            )
            return audio_data
        except Exception as e:
            logger.error("SiliconFlow TTS 调用失败: %s", e)
            raise

    async def synthesize_async(self, text: str) -> bytes:
        try:
            import httpx
        except ImportError:
            # fallback to sync
            return self.synthesize(text)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": voice_config.TTS_OUTPUT_FORMAT,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self._api_url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.content

    def synthesize_to_file(self, text: str, file_path: str) -> str:
        audio_data = self.synthesize(text)
        with open(file_path, "wb") as f:
            f.write(audio_data)
        return file_path


def create_tts_engine() -> TTSEngine:
    """根据配置创建 TTS 引擎"""
    backend = voice_config.TTS_BACKEND
    if backend == TTSBackend.EDGE_TTS:
        return EdgeTTSEngine()
    elif backend == TTSBackend.SILICONFLOW:
        return SiliconFlowTTSEngine()
    else:
        raise ValueError(f"不支持的 TTS 后端: {backend}")


# ═══════════════════════════════════════════════════
# 音频采集与播放
# ═══════════════════════════════════════════════════

class AudioRecorder:
    """麦克风录音器，支持 VAD（语音活动检测）自动停止"""

    def __init__(self):
        self._sample_rate = voice_config.STT_SAMPLE_RATE
        self._channels = voice_config.AUDIO_CHANNELS
        self._block_size = voice_config.AUDIO_BLOCK_SIZE
        self._device_index = voice_config.AUDIO_DEVICE_INDEX
        self._silence_threshold = voice_config.STT_SILENCE_THRESHOLD
        self._silence_duration = voice_config.STT_SILENCE_DURATION
        self._max_duration = voice_config.STT_MAX_DURATION

    def record(self) -> bytes:
        """录音直到检测到静音，返回 PCM 音频数据"""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError(
                "请安装音频库: pip install sounddevice numpy\n"
                "Linux 额外需要: apt-get install libportaudio2"
            )

        logger.info(
            "开始录音 (采样率=%d, 静音阈值=%.3f, 静音时长=%.1fs)",
            self._sample_rate,
            self._silence_threshold,
            self._silence_duration,
        )

        frames = []
        silent_chunks = 0
        chunks_per_second = self._sample_rate / self._block_size
        silence_chunks_needed = int(self._silence_duration * chunks_per_second)
        max_chunks = int(self._max_duration * chunks_per_second)
        recording_started = False

        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._block_size,
            device=self._device_index,
        )

        with stream:
            while len(frames) < max_chunks:
                data, overflowed = stream.read(self._block_size)
                if overflowed:
                    logger.warning("音频缓冲区溢出")

                frames.append(data.copy())

                # 计算 RMS 能量
                audio_chunk = data.flatten().astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_chunk**2))

                if rms > self._silence_threshold:
                    recording_started = True
                    silent_chunks = 0
                elif recording_started:
                    silent_chunks += 1

                # 检测到足够静音 → 停止
                if recording_started and silent_chunks >= silence_chunks_needed:
                    logger.info("检测到静音，停止录音")
                    break

        # 合并所有帧
        audio_data = np.concatenate(frames, axis=0)
        pcm_bytes = audio_data.astype(np.int16).tobytes()

        duration = len(pcm_bytes) / (
            self._sample_rate * self._channels * voice_config.AUDIO_SAMPLE_WIDTH
        )
        logger.info("录音完成: %.1f 秒, %d bytes", duration, len(pcm_bytes))
        return pcm_bytes

    def record_to_wav(self, wav_path: str) -> str:
        """录音并保存为 WAV 文件"""
        pcm_data = self.record()
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(voice_config.AUDIO_SAMPLE_WIDTH)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm_data)
        return wav_path


class AudioPlayer:
    """音频播放器"""

    @staticmethod
    def play_bytes(audio_data: bytes, format: str = "mp3"):
        """播放音频 bytes"""
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            raise ImportError("请安装: pip install sounddevice soundfile")

        # 写入临时文件
        suffix = f".{format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        try:
            data, sr = sf.read(tmp_path)
            sd.play(data, sr)
            sd.wait()
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def play_file(file_path: str):
        """播放音频文件"""
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            raise ImportError("请安装: pip install sounddevice soundfile")

        data, sr = sf.read(file_path)
        sd.play(data, sr)
        sd.wait()

    @staticmethod
    def save_audio(audio_data: bytes, file_path: str, format: str = "mp3"):
        """保存音频到文件"""
        with open(file_path, "wb") as f:
            f.write(audio_data)
        logger.info("音频已保存: %s (%d bytes)", file_path, len(audio_data))


# ═══════════════════════════════════════════════════
# 语音交互主模块
# ═══════════════════════════════════════════════════

class VoiceModule:
    """STT + TTS 语音交互模块

    完整流程：
    1. 麦克风录音 → STT 识别 → 文字
    2. 文字 → dx_agent.run_single() → 回答
    3. 回答 → TTS 合成 → 播放语音

    集成 dx_agent 的 run_single() 和 Config。
    """

    def __init__(
        self,
        stt_engine: Optional[STTEngine] = None,
        tts_engine: Optional[TTSEngine] = None,
        recorder: Optional[AudioRecorder] = None,
        player: Optional[AudioPlayer] = None,
    ):
        self.stt = stt_engine or create_stt_engine()
        self.tts = tts_engine or create_tts_engine()
        self.recorder = recorder or AudioRecorder()
        self.player = player or AudioPlayer()
        self._running = False

        # 延迟导入 dx_agent 避免循环依赖
        self._agent_func = None

    def _get_agent(self) -> Callable:
        """延迟加载 dx_agent.run_single"""
        if self._agent_func is None:
            from dx_agent import run_single
            self._agent_func = run_single
        return self._agent_func

    def process_text(self, text: str) -> dict:
        """文字 → Agent 查询 → 返回结果

        Returns:
            {"text": str, "answer": str, "success": bool, "processing_time": float}
        """
        start = time.time()

        try:
            run_single = self._get_agent()
            result = run_single(text, verbose=False)

            answer = result.get("answer", "抱歉，查询失败，请重试。")
            success = result.get("success", False)
            processing_time = result.get("processing_time", time.time() - start)

            return {
                "text": text,
                "answer": answer,
                "success": success,
                "processing_time": processing_time,
            }
        except Exception as e:
            logger.error("Agent 调用失败: %s", e, exc_info=True)
            return {
                "text": text,
                "answer": "抱歉，系统出现错误，请稍后重试。",
                "success": False,
                "processing_time": time.time() - start,
            }

    def process_audio(self, audio_data: bytes) -> dict:
        """音频 → STT → Agent → TTS → 返回完整结果

        Returns:
            {
                "stt_text": str,
                "answer": str,
                "success": bool,
                "tts_audio": bytes,
                "processing_time": float,
            }
        """
        start = time.time()

        # 1. STT 识别
        stt_text = self.stt.transcribe(audio_data, voice_config.STT_SAMPLE_RATE)
        if not stt_text.strip():
            return {
                "stt_text": "",
                "answer": "没有检测到语音，请再说一次。",
                "success": False,
                "tts_audio": b"",
                "processing_time": time.time() - start,
            }

        if voice_config.SHOW_STT_TEXT:
            print(f"🎤 识别: {stt_text}")

        # 2. 唤醒词检测
        if voice_config.WAKE_WORD:
            if voice_config.WAKE_WORD not in stt_text:
                logger.debug("未检测到唤醒词 '%s'，跳过", voice_config.WAKE_WORD)
                return {
                    "stt_text": stt_text,
                    "answer": "",
                    "success": False,
                    "tts_audio": b"",
                    "processing_time": time.time() - start,
                }
            # 去除唤醒词
            stt_text = stt_text.replace(voice_config.WAKE_WORD, "").strip()

        # 3. Agent 查询
        query_result = self.process_text(stt_text)
        answer = query_result["answer"]

        # 4. TTS 合成
        tts_audio = b""
        try:
            tts_audio = self.tts.synthesize(answer)
        except Exception as e:
            logger.error("TTS 合成失败: %s", e)

        return {
            "stt_text": stt_text,
            "answer": answer,
            "success": query_result["success"],
            "tts_audio": tts_audio,
            "processing_time": time.time() - start,
        }

    def voice_loop(self):
        """交互式语音循环

        流程：录音 → 识别 → 查询 → 合成 → 播放 → 循环
        """
        self._running = True

        print("╔═══════════════════════════════════════════════════════╗")
        print("║       运营商套餐 Agent — 语音交互模式                  ║")
        print("║                                                       ║")
        print("║  🎤 对着麦克风说话，我会自动识别并回答                  ║")
        print("║  ⏸  说完后停顿 1.5 秒自动结束录音                      ║")
        print("║  🛑 按 Ctrl+C 退出                                    ║")
        print("║                                                       ║")
        print(f"║  STT: {voice_config.STT_BACKEND.value:<10}  TTS: {voice_config.TTS_BACKEND.value:<10}       ║")
        print(f"║  语音: {voice_config.TTS_VOICE:<43}   ║")
        print("╚═══════════════════════════════════════════════════════╝\n")

        # 预热 Agent
        try:
            self._get_agent()
            logger.info("Agent 预热完成")
        except Exception as e:
            logger.warning("Agent 预热失败: %s", e)

        round_num = 0
        while self._running:
            round_num += 1
            try:
                print(f"\n{'─' * 50}")
                print(f"🎤 第 {round_num} 轮 — 请说话...")
                print(f"{'─' * 50}")

                # 录音
                audio_data = self.recorder.record()

                # 处理
                result = self.process_audio(audio_data)

                # 显示结果
                print(f"\n💬 识别: {result['stt_text']}")
                print(f"🤖 回答: {result['answer']}")
                print(f"⏱  耗时: {result['processing_time']:.2f}s")

                # 播放 TTS
                if result["tts_audio"] and voice_config.AUTO_PLAY_TTS:
                    print("🔊 播放回答...")
                    try:
                        self.player.play_bytes(
                            result["tts_audio"],
                            format=voice_config.TTS_OUTPUT_FORMAT,
                        )
                    except Exception as e:
                        logger.warning("音频播放失败: %s", e)
                        print(f"⚠️  播放失败: {e}")

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                self._running = False
                break
            except Exception as e:
                logger.error("语音处理异常: %s", e, exc_info=True)
                print(f"❌ 错误: {e}")
                continue

    def stop(self):
        """停止语音循环"""
        self._running = False


# ═══════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════

def text_to_speech(text: str, output_path: str = None) -> bytes:
    """便捷函数：文字转语音

    Args:
        text: 要合成的文字
        output_path: 可选，保存到文件路径

    Returns:
        音频 bytes
    """
    engine = create_tts_engine()
    audio = engine.synthesize(text)
    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio)
    return audio


def speech_to_text(audio_path: str) -> str:
    """便捷函数：音频文件转文字

    Args:
        audio_path: 音频文件路径

    Returns:
        识别出的文字
    """
    engine = create_stt_engine()
    return engine.transcribe_file(audio_path)


def voice_query(text: str) -> str:
    """便捷函数：文字查询 Agent 并返回 TTS 音频路径

    Args:
        text: 用户问题

    Returns:
        (answer_text, tts_audio_path)
    """
    from dx_agent import run_single

    result = run_single(text, verbose=False)
    answer = result.get("answer", "查询失败")

    # 生成 TTS
    audio = text_to_speech(answer)
    audio_path = os.path.join(tempfile.gettempdir(), "voice_answer.mp3")
    with open(audio_path, "wb") as f:
        f.write(audio)

    return answer, audio_path


# ═══════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="运营商套餐 Agent 语音交互模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python voice_module.py                          # 语音交互模式
  python voice_module.py --text                   # 文字模式（调试）
  python voice_module.py --stt-file audio.wav     # 识别音频文件
  python voice_module.py --tts "59元套餐包含多少流量"  # 文字查询+语音回答
  python voice_module.py --tts-only "你好世界"    # 仅 TTS 合成

环境变量:
  STT_BACKEND=funasr|whisper|siliconflow    # STT 引擎
  TTS_BACKEND=edge-tts|siliconflow          # TTS 引擎
  TTS_VOICE=zh-CN-XiaoxiaoNeural           # TTS 语音角色
        """,
    )
    parser.add_argument("--text", action="store_true", help="文字交互模式（不使用麦克风）")
    parser.add_argument("--stt-file", type=str, help="识别指定音频文件")
    parser.add_argument("--tts", type=str, help="文字查询并生成语音回答")
    parser.add_argument("--tts-only", type=str, help="仅 TTS 合成（不查询 Agent）")
    parser.add_argument("--tts-output", type=str, help="TTS 输出文件路径")
    parser.add_argument("--list-voices", action="store_true", help="列出可用 TTS 语音")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备")

    args = parser.parse_args()

    # 列出音频设备
    if args.list_devices:
        try:
            import sounddevice as sd
            print("可用音频设备:")
            print(sd.query_devices())
        except ImportError:
            print("请安装 sounddevice: pip install sounddevice")
        return

    # 列出 TTS 语音
    if args.list_voices:
        _list_voices()
        return

    # STT 文件识别
    if args.stt_file:
        text = speech_to_text(args.stt_file)
        print(f"识别结果: {text}")
        return

    # 仅 TTS
    if args.tts_only:
        output = args.tts_output or "tts_output.mp3"
        audio = text_to_speech(args.tts_only, output)
        print(f"已合成: {output} ({len(audio)} bytes)")
        return

    # 文字查询 + TTS
    if args.tts:
        answer, audio_path = voice_query(args.tts)
        print(f"回答: {answer}")
        print(f"语音: {audio_path}")
        if args.tts_output:
            import shutil
            shutil.move(audio_path, args.tts_output)
            print(f"已保存: {args.tts_output}")
        return

    # 文字交互模式
    if args.text:
        _text_mode()
        return

    # 默认：语音交互模式
    try:
        module = VoiceModule()
        module.voice_loop()
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("\n安装依赖:")
        print("  pip install sounddevice numpy soundfile edge-tts")
        print("  # STT 三选一:")
        print("  pip install funasr            # FunASR（推荐，中文最佳）")
        print("  pip install openai-whisper     # Whisper")
        print("  # 或使用 SiliconFlow API（无需额外安装）")
        sys.exit(1)


def _text_mode():
    """文字交互模式（调试用）"""
    from dx_agent import run_single

    print("╔═══════════════════════════════════════════════════════╗")
    print("║       运营商套餐 Agent — 文字+语音模式                 ║")
    print("║  输入文字查询，回答将同时以文字和语音返回               ║")
    print("║  输入 quit 退出                                       ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    tts_engine = create_tts_engine()

    while True:
        try:
            query = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        # Agent 查询
        result = run_single(query, verbose=False)
        answer = result.get("answer", "查询失败")
        print(f"🤖 回答: {answer}")
        print(f"⏱  耗时: {result.get('processing_time', 0):.2f}s")

        # TTS 合成 + 播放
        try:
            audio = tts_engine.synthesize(answer)
            if voice_config.AUTO_PLAY_TTS:
                AudioPlayer.play_bytes(audio, format=voice_config.TTS_OUTPUT_FORMAT)
        except Exception as e:
            logger.debug("TTS 跳过: %s", e)


def _list_voices():
    """列出 Edge-TTS 可用的中文语音"""
    try:
        import edge_tts
    except ImportError:
        print("请安装 edge-tts: pip install edge-tts")
        return

    async def _list():
        voices = await edge_tts.list_voices()
        cn_voices = [v for v in voices if v["Locale"].startswith("zh-")]
        print(f"可用中文语音 ({len(cn_voices)} 个):\n")
        for v in cn_voices:
            gender = "♀" if v["Gender"] == "Female" else "♂"
            print(f"  {gender} {v['ShortName']:<35} {v['LocalName']}")
        print(f"\n当前使用: {voice_config.TTS_VOICE}")

    asyncio.run(_list())


if __name__ == "__main__":
    main()
