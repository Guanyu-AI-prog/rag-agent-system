"""
Agent API 服务 - 端口 8002
基于 dx_agent.py 的 PurePythonAgent，提供 RESTful + SSE 流式接口
"""

import os
import time
import asyncio
import json
import uuid
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn

from dx_agent import PurePythonAgent
from config import Config


logger = logging.getLogger("agent-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


app = FastAPI(title="Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（复用 8001 的 web_ui）
from fastapi.staticfiles import StaticFiles

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"静态文件目录已挂载: {static_dir}")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")


agent = None
executor = None


@app.on_event("startup")
async def startup_event():
    global agent, executor
    logger.info("启动 Agent API 服务...")
    agent = PurePythonAgent()
    executor = ThreadPoolExecutor(max_workers=4)
    logger.info("Agent API 服务就绪 (端口 8002)")


@app.get("/")
async def root():
    return {"message": "Agent API Service", "version": "1.0.0", "docs": "/docs", "health": "/health", "web_ui": "/web"}


@app.get("/web")
async def web_ui():
    return RedirectResponse(url="/static/web_ui.html")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "agent-api", "port": 8002}


@app.post("/query")
async def query_agent(request: QueryRequest):
    """同步查询 - 等待完整回答后返回"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent 未就绪")

    start = time.time()
    loop = asyncio.get_event_loop()
    try:
        answer = await loop.run_in_executor(
            executor,
            lambda: agent.run(request.question, verbose=False)
        )
        return {
            "answer": answer,
            "success": True,
            "processing_time": time.time() - start,
            "session_id": request.session_id or str(uuid.uuid4()),
        }
    except Exception as e:
        logger.error(f"Agent 查询异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """SSE 流式查询 - 逐字推送回答"""
    if not agent:
        async def err():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent 未就绪'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    start = time.time()

    async def generate():
        try:
            # 后台线程执行 Agent（可能耗时较长，涉及多轮 LLM 调用）
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                executor,
                lambda: agent.run(request.question, verbose=False)
            )
            elapsed = time.time() - start

            # 发送路由标识
            yield f"data: {json.dumps({'type': 'route', 'route': 'agent'})}\n\n"

            # 逐字流式推送
            for i in range(0, len(answer), 3):
                chunk = answer[i:i+3]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)

            # 完成信号
            yield f"data: {json.dumps({'type': 'done', 'processing_time': elapsed})}\n\n"

        except Exception as e:
            logger.error(f"Agent 流式查询异常: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("api_agent:app", host="0.0.0.0", port=8002, reload=False, workers=1)
