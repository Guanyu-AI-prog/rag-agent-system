#!/usr/bin/env python3
"""
语音交互 API 服务 — 运行在 8002 端口

接口：
  POST /voice/query      — 音频文件上传 → STT → Agent → TTS → 返回结果+音频
  POST /voice/text       — 文字查询 → Agent → TTS → 返回结果+音频
  POST /voice/stt        — 音频文件上传 → STT → 返回识别文字
  POST /voice/tts        — 文字 → TTS → 返回音频
  GET  /voice/voices     — 列出可用 TTS 语音
  GET  /voice/health     — 健康检查
  GET  /voice/config     — 当前语音配置

启动：
  python voice_api.py
  # 或
  uvicorn voice_api:app --host 0.0.0.0 --port 8002
"""

import logging
import os
import time
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import Config
from voice_module import (
    VoiceModule,
    VoiceConfig,
    voice_config,
    create_stt_engine,
    create_tts_engine,
    AudioPlayer,
    STTBackend,
    TTSBackend,
)

# ── 日志 ──
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
logger = logging.getLogger("voice_api")

# ── FastAPI 应用 ──
app = FastAPI(
    title="运营商套餐 Agent - 语音 API",
    description="STT + TTS 语音交互接口，集成 dx_agent 套餐查询",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局实例（延迟初始化） ──
_voice_module: Optional[VoiceModule] = None
_stt_engine = None
_tts_engine = None


def get_voice_module() -> VoiceModule:
    global _voice_module
    if _voice_module is None:
        _voice_module = VoiceModule()
        logger.info("VoiceModule 初始化完成")
    return _voice_module


def get_stt():
    global _stt_engine
    if _stt_engine is None:
        _stt_engine = create_stt_engine()
        logger.info("STT 引擎初始化: %s", voice_config.STT_BACKEND.value)
    return _stt_engine


def get_tts():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = create_tts_engine()
        logger.info("TTS 引擎初始化: %s", voice_config.TTS_BACKEND.value)
    return _tts_engine


# ═══════════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════════

class TextQueryRequest(BaseModel):
    """文字查询请求"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    tts_enabled: bool = Field(True, description="是否返回 TTS 音频")


class TTSRequest(BaseModel):
    """TTS 合成请求"""
    text: str = Field(..., min_length=1, max_length=5000, description="要合成的文字")
    voice: Optional[str] = Field(None, description="语音角色（覆盖默认配置）")
    rate: Optional[str] = Field(None, description="语速，如 +10%、-20%")
    volume: Optional[str] = Field(None, description="音量，如 +10%、-20%")


class VoiceQueryResponse(BaseModel):
    """语音查询响应"""
    stt_text: str = Field("", description="STT 识别文字")
    answer: str = Field("", description="Agent 回答")
    success: bool = Field(True, description="是否成功")
    processing_time: float = Field(0, description="总处理时间（秒）")
    has_audio: bool = Field(False, description="是否包含音频（需单独请求 /voice/tts）")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    stt_backend: str = ""
    tts_backend: str = ""
    uptime: float = 0


# ── 启动时间 ──
_start_time = time.time()


# ═══════════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════════

@app.get("/voice/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok",
        stt_backend=voice_config.STT_BACKEND.value,
        tts_backend=voice_config.TTS_BACKEND.value,
        uptime=round(time.time() - _start_time, 1),
    )


@app.get("/voice/config", tags=["系统"])
async def get_config():
    """查看当前语音配置"""
    return {
        "stt": {
            "backend": voice_config.STT_BACKEND.value,
            "model": voice_config.STT_MODEL,
            "language": voice_config.STT_LANGUAGE,
            "sample_rate": voice_config.STT_SAMPLE_RATE,
            "silence_threshold": voice_config.STT_SILENCE_THRESHOLD,
            "silence_duration": voice_config.STT_SILENCE_DURATION,
        },
        "tts": {
            "backend": voice_config.TTS_BACKEND.value,
            "voice": voice_config.TTS_VOICE,
            "rate": voice_config.TTS_RATE,
            "volume": voice_config.TTS_VOLUME,
            "output_format": voice_config.TTS_OUTPUT_FORMAT,
        },
    }


@app.post("/voice/stt", tags=["语音识别"])
async def speech_to_text(
    audio: UploadFile = File(..., description="音频文件（WAV/MP3/FLAC/OGG）"),
    language: Optional[str] = Form(None, description="语言代码，默认 zh"),
):
    """音频 → 文字识别

    支持上传 WAV、MP3、FLAC、OGG 等格式的音频文件。
    """
    request_id = uuid.uuid4().hex[:8]
    start = time.time()

    # 读取音频
    audio_bytes = await audio.read()
    logger.info("[%s] STT 请求: %s, %d bytes", request_id, audio.filename, len(audio_bytes))

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="音频文件为空")

    # 保存到临时文件
    suffix = _guess_audio_suffix(audio.filename)
    tmp_path = os.path.join(tempfile.gettempdir(), f"stt_{request_id}{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # STT 识别
        stt = get_stt()
        text = stt.transcribe_file(tmp_path)

        elapsed = time.time() - start
        logger.info("[%s] STT 完成: '%s' (%.2fs)", request_id, text[:50], elapsed)

        return {
            "text": text,
            "language": language or voice_config.STT_LANGUAGE,
            "processing_time": round(elapsed, 3),
        }
    except Exception as e:
        logger.error("[%s] STT 失败: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/voice/tts", tags=["语音合成"])
async def text_to_speech(request: TTSRequest):
    """文字 → 语音合成

    返回 MP3/WAV 音频流，可直接播放。
    """
    request_id = uuid.uuid4().hex[:8]
    start = time.time()

    logger.info("[%s] TTS 请求: '%s' (%d 字)", request_id, request.text[:30], len(request.text))

    try:
        tts = get_tts()

        # 临时覆盖配置（如果有自定义参数）
        original_voice = voice_config.TTS_VOICE
        original_rate = voice_config.TTS_RATE
        original_volume = voice_config.TTS_VOLUME

        if request.voice:
            voice_config.TTS_VOICE = request.voice
        if request.rate:
            voice_config.TTS_RATE = request.rate
        if request.volume:
            voice_config.TTS_VOLUME = request.volume

        # 如果参数有变化，重建引擎
        if request.voice or request.rate or request.volume:
            global _tts_engine
            _tts_engine = create_tts_engine()
            tts = _tts_engine

        audio_data = tts.synthesize(request.text)

        # 恢复配置
        voice_config.TTS_VOICE = original_voice
        voice_config.TTS_RATE = original_rate
        voice_config.TTS_VOLUME = original_volume
        if request.voice or request.rate or request.volume:
            _tts_engine = create_tts_engine()

        elapsed = time.time() - start
        content_type = "audio/mpeg" if voice_config.TTS_OUTPUT_FORMAT == "mp3" else "audio/wav"

        logger.info("[%s] TTS 完成: %d bytes (%.2fs)", request_id, len(audio_data), elapsed)

        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename=tts_{request_id}.{voice_config.TTS_OUTPUT_FORMAT}",
                "X-Processing-Time": str(round(elapsed, 3)),
            },
        )
    except Exception as e:
        logger.error("[%s] TTS 失败: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


@app.post("/voice/query", tags=["语音查询"])
async def voice_query(
    audio: UploadFile = File(..., description="用户语音音频文件"),
    tts_enabled: bool = Form(True, description="是否返回 TTS 音频"),
    session_id: Optional[str] = Form(None, description="会话ID"),
):
    """完整语音查询流程：音频 → STT → Agent → TTS

    返回 JSON 结果（包含识别文字和回答）。
    如需 TTS 音频，设置 tts_enabled=true，音频将作为二进制流返回（Content-Type: audio/mpeg）。
    """
    request_id = uuid.uuid4().hex[:8]
    start = time.time()

    audio_bytes = await audio.read()
    logger.info("[%s] 语音查询: %s, %d bytes", request_id, audio.filename, len(audio_bytes))

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="音频文件为空")

    # 保存临时文件用于 STT
    suffix = _guess_audio_suffix(audio.filename)
    tmp_path = os.path.join(tempfile.gettempdir(), f"vq_{request_id}{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # 1. STT
        stt = get_stt()
        stt_text = stt.transcribe_file(tmp_path)

        if not stt_text.strip():
            return JSONResponse(
                content={
                    "stt_text": "",
                    "answer": "没有检测到语音，请再说一次。",
                    "success": False,
                    "processing_time": round(time.time() - start, 3),
                }
            )

        # 2. Agent 查询
        module = get_voice_module()
        query_result = module.process_text(stt_text)

        elapsed = time.time() - start

        if not tts_enabled:
            return {
                "stt_text": stt_text,
                "answer": query_result["answer"],
                "success": query_result["success"],
                "processing_time": round(elapsed, 3),
            }

        # 3. TTS + 返回音频
        tts = get_tts()
        tts_audio = tts.synthesize(query_result["answer"])

        content_type = "audio/mpeg" if voice_config.TTS_OUTPUT_FORMAT == "mp3" else "audio/wav"

        logger.info(
            "[%s] 语音查询完成: stt='%s', answer='%s', %d bytes (%.2fs)",
            request_id,
            stt_text[:30],
            query_result["answer"][:30],
            len(tts_audio),
            elapsed,
        )

        return Response(
            content=tts_audio,
            media_type=content_type,
            headers={
                "X-STT-Text": stt_text[:200],
                "X-Answer": query_result["answer"][:200],
                "X-Success": str(query_result["success"]),
                "X-Processing-Time": str(round(elapsed, 3)),
                "Content-Disposition": f"inline; filename=answer_{request_id}.{voice_config.TTS_OUTPUT_FORMAT}",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[%s] 语音查询失败: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"语音查询失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/voice/text", tags=["语音查询"])
async def text_query_with_tts(request: TextQueryRequest):
    """文字查询 + 可选 TTS

    输入文字，Agent 回答。如 tts_enabled=true，回答同时返回音频 URL。
    """
    request_id = uuid.uuid4().hex[:8]
    start = time.time()

    logger.info("[%s] 文字查询: '%s'", request_id, request.question[:50])

    try:
        module = get_voice_module()
        result = module.process_text(request.question)

        elapsed = time.time() - start

        response = {
            "answer": result["answer"],
            "success": result["success"],
            "processing_time": round(elapsed, 3),
        }

        if request.tts_enabled:
            # 生成 TTS 并保存临时文件，返回访问路径
            try:
                tts = get_tts()
                audio_data = tts.synthesize(result["answer"])

                audio_filename = f"answer_{request_id}.{voice_config.TTS_OUTPUT_FORMAT}"
                audio_dir = os.path.join(tempfile.gettempdir(), "voice_api_audio")
                os.makedirs(audio_dir, exist_ok=True)
                audio_path = os.path.join(audio_dir, audio_filename)

                with open(audio_path, "wb") as f:
                    f.write(audio_data)

                response["audio_url"] = f"/voice/audio/{audio_filename}"
                response["audio_size"] = len(audio_data)
            except Exception as e:
                logger.warning("[%s] TTS 失败: %s", request_id, e)
                response["audio_error"] = str(e)

        return response

    except Exception as e:
        logger.error("[%s] 查询失败: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/voice/audio/{filename}", tags=["音频"])
async def get_audio_file(filename: str):
    """获取已生成的 TTS 音频文件"""
    audio_dir = os.path.join(tempfile.gettempdir(), "voice_api_audio")
    file_path = os.path.join(audio_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音频文件不存在或已过期")

    content_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    with open(file_path, "rb") as f:
        audio_data = f.read()

    return Response(
        content=audio_data,
        media_type=content_type,
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )


@app.get("/voice/voices", tags=["系统"])
async def list_voices():
    """列出可用 TTS 语音"""
    if voice_config.TTS_BACKEND == TTSBackend.EDGE_TTS:
        try:
            import asyncio
            import edge_tts

            voices = await edge_tts.list_voices()
            cn_voices = [
                {
                    "name": v["ShortName"],
                    "local_name": v["LocalName"],
                    "gender": v["Gender"],
                    "locale": v["Locale"],
                }
                for v in voices
                if v["Locale"].startswith("zh-")
            ]
            return {
                "backend": "edge-tts",
                "current_voice": voice_config.TTS_VOICE,
                "voices": cn_voices,
            }
        except ImportError:
            return {"backend": "edge-tts", "error": "edge-tts 未安装"}
    else:
        return {
            "backend": voice_config.TTS_BACKEND.value,
            "current_voice": voice_config.TTS_VOICE,
            "note": "当前后端不支持动态列出语音",
        }


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def _guess_audio_suffix(filename: Optional[str]) -> str:
    """从文件名推断音频格式后缀"""
    if not filename:
        return ".wav"
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
        return ext
    return ".wav"


# ═══════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("VOICE_API_PORT", "8002"))
    host = os.getenv("VOICE_API_HOST", "0.0.0.0")

    print(f"╔═══════════════════════════════════════════════════════╗")
    print(f"║       运营商套餐 Agent — 语音 API 服务                 ║")
    print(f"║                                                       ║")
    print(f"║  🌐 地址: http://{host}:{port}                        ║")
    print(f"║  📖 文档: http://{host}:{port}/docs                   ║")
    print(f"║  🎤 STT:  {voice_config.STT_BACKEND.value:<10}                              ║")
    print(f"║  🔊 TTS:  {voice_config.TTS_BACKEND.value:<10}                              ║")
    print(f"║  🗣  语音: {voice_config.TTS_VOICE:<35}    ║")
    print(f"╚═══════════════════════════════════════════════════════╝")

    uvicorn.run(
        "voice_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
