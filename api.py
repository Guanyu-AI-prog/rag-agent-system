"""
纯Python RAG API服务（不依赖LangChain）
基于FastAPI，支持RESTful接口
使用 simple_rag.py 作为后端
"""

import os
import time
import asyncio
import hashlib
import json
import uuid
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from simple_rag import SimpleRAG
from config import Config
from conversation_history import ConversationManager
from query_logger import QueryLogger


# ═══════════════════════════════════════════
# 管理接口 Token 鉴权
# ═══════════════════════════════════════════
async def verify_admin_token(authorization: Optional[str] = Header(None)):
    """校验 Authorization: Bearer <token>"""
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization 格式: Bearer <token>")
    if parts[1] != Config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token 无效")


def setup_logging():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(Config.LOG_FORMAT)

    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=Config.LOG_MAX_SIZE * 1024 * 1024,
        backupCount=Config.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID（可选）")
    user_id: Optional[str] = Field(None, description="用户ID（可选）")


class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1, max_length=Config.MAX_BATCH_SIZE, description="问题列表")
    session_id: Optional[str] = Field(None, description="会话ID（可选）")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="回答内容")
    sources: List[str] = Field(default_factory=list, description="来源文档")
    success: bool = Field(True, description="是否成功")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")
    token_usage: Optional[Dict[str, Any]] = Field(None, description="Token使用情况")
    session_id: Optional[str] = Field(None, description="会话ID")
    thinking_steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="思考过程步骤")


class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
    timestamp: float = Field(..., description="时间戳")


class StatsResponse(BaseModel):
    status: str = Field(..., description="状态")
    document_count: Optional[int] = Field(None, description="文档数量")
    llm_model: Optional[str] = Field(None, description="LLM模型")
    embedding_model: Optional[str] = Field(None, description="嵌入模型")


class EnhancedSimpleCache:
    """增强版内存缓存：支持统计和LRU淘汰"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000, enable_stats: bool = True):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.enable_stats = enable_stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lru_queue = deque()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    if key in self.lru_queue:
                        self.lru_queue.remove(key)
                    self.lru_queue.append(key)
                    if self.enable_stats:
                        self.hits += 1
                    return entry['value']
                else:
                    del self.cache[key]
                    if key in self.lru_queue:
                        self.lru_queue.remove(key)
            if self.enable_stats:
                self.misses += 1
            return None

    def set(self, key: str, value: Any):
        with self.lock:
            if key in self.lru_queue:
                self.lru_queue.remove(key)

            if len(self.cache) >= self.max_size and key not in self.cache:
                if self.lru_queue:
                    lru_key = self.lru_queue.popleft()
                    if lru_key in self.cache:
                        del self.cache[lru_key]
                        if self.enable_stats:
                            self.evictions += 1

            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self.lru_queue.append(key)

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.lru_queue.clear()
            if self.enable_stats:
                self.hits = 0
                self.misses = 0
                self.evictions = 0

    def size(self) -> int:
        with self.lock:
            return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "evictions": self.evictions,
                "ttl_seconds": self.ttl
            }


async def monitor_cache_stats():
    logger.info(f"缓存监控已启动，记录间隔: {Config.CACHE_MONITOR_INTERVAL}秒")
    while True:
        try:
            await asyncio.sleep(Config.CACHE_MONITOR_INTERVAL)
            if cache:
                stats = cache.get_stats()
                logger.info(
                    f"缓存监控 - "
                    f"命中率: {stats['hit_rate']} | "
                    f"大小: {stats['size']}/{stats['max_size']} | "
                    f"命中: {stats['hits']} | "
                    f"未命中: {stats['misses']} | "
                    f"淘汰: {stats['evictions']}"
                )
        except asyncio.CancelledError:
            logger.info("缓存监控任务已取消")
            break
        except Exception as e:
            logger.error(f"缓存监控错误: {e}")


def _postprocess_answer(answer: str, question: str = "") -> str:
    if "抱歉，暂未找到相关信息" not in answer:
        return answer
    if any(kw in question for kw in ["wifi", "WiFi", "无线"]):
        return "您是否想了解宽带融合套餐？例如99元套餐含100M宽带，129元套餐含300M宽带。"
    if any(kw in question for kw in ["兆", " M", "速率"]):
        return "各档位宽带速率参考：99元含100M，129元含300M，299元可选1000M宽带。您想了解哪个套餐？"
    if any(kw in question for kw in ["流量", "不够", "叠加", "通话"]):
        return "您是否想了解5G畅享29-199元系列套餐？不同档位包含不同流量和通话时长。"
    return "目前暂无该套餐的详细信息，建议您咨询人工客服或查看其他套餐。您是否想了解5G畅享29-199元系列套餐？"


def _make_cache_key(question: str, conversation_history: str = "") -> str:
    raw = f"{question}:{conversation_history}"
    return hashlib.md5(raw.encode()).hexdigest()


app = FastAPI(
    title="RAG API",
    description="纯Python RAG问答系统API（不依赖LangChain）",
    version="2.0.0",
    docs_url=None,  # 生产环境关闭 Swagger UI
    redoc_url=None,  # 生产环境关闭 ReDoc
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse

    static_dir = "static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"静态文件目录已挂载: {static_dir}")
    else:
        logger.warning(f"静态文件目录不存在: {static_dir}")
except ImportError:
    logger.warning("无法导入StaticFiles，静态文件服务不可用")
except Exception as e:
    logger.error(f"静态文件挂载失败: {e}")

rag = None
cache = None
executor = None
monitor_task = None
conversation_manager = None
query_logger = None
shortcut_cache = {}
SHORTCUT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "shortcut_cache.json")


# ═══════════════════════════════════════════
# 快捷问答（直达路由，跳过RAG全流程）
# ═══════════════════════════════════════════

SHORTCUT_QUERIES = {
    "1": "59套餐包含多少流量",
    "2": "对比99和129套餐",
    "3": "199套餐详情",
    "4": "各套餐转网数量对比",
    "5": "请展示几条常用的营销话术，每条只保留情景名称和一句话术要点，最后问我具体想了解哪方面的话术",
    "6": "请从上下文中提取59、99、129、199、299各套餐的补贴详情，优先展示橙分期方案。每个套餐必须列出：1.橙分期24个月和36个月的购机补贴金额；2.全额预存的原价月租、每月赠话费、实付金额。用表格展示，橙分期表格排在前面",
}


def _load_shortcut_cache_file():
    """从文件加载预计算缓存"""
    global shortcut_cache
    if not os.path.exists(SHORTCUT_CACHE_FILE):
        return False
    try:
        with open(SHORTCUT_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and len(data) >= len(SHORTCUT_QUERIES):
            shortcut_cache = data
            logger.info(f"从文件加载快捷问答缓存: {len(shortcut_cache)} 条")
            return True
    except Exception as e:
        logger.warning(f"加载快捷缓存文件失败: {e}")
    return False


def _save_shortcut_cache_file():
    """将预计算结果保存到文件"""
    try:
        with open(SHORTCUT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(shortcut_cache, f, ensure_ascii=False, indent=2)
        logger.info(f"快捷问答缓存已保存到文件: {len(shortcut_cache)} 条")
    except Exception as e:
        logger.warning(f"保存快捷缓存文件失败: {e}")


def _init_shortcuts():
    """初始化快捷问答：优先从文件加载，否则后台预计算"""
    import threading

    # 先尝试从文件加载
    if _load_shortcut_cache_file():
        return  # 文件加载成功，秒级可用

    # 文件不存在或不完整，后台预计算
    def _compute():
        global shortcut_cache
        if not rag:
            logger.warning("RAG未就绪，跳过快捷问答预计算")
            return

        logger.info("后台开始预计算快捷问答...")
        start = time.time()
        for sid, question in SHORTCUT_QUERIES.items():
            try:
                result = rag.query(question)
                shortcut_cache[sid] = {
                    "answer": result.answer,
                    "sources": result.sources[:3],
                }
                logger.info(f"  快捷[{sid}] 预计算完成")
            except Exception as e:
                logger.error(f"  快捷[{sid}] 预计算失败: {e}")

        elapsed = time.time() - start
        logger.info(f"快捷问答预计算完成: {len(shortcut_cache)} 条, 耗时 {elapsed:.1f}s")

        # 保存到文件
        _save_shortcut_cache_file()

    t = threading.Thread(target=_compute, daemon=True)
    t.start()


@app.on_event("startup")
async def startup_event():
    global rag, cache, executor, conversation_manager, monitor_task, query_logger

    logger.info("启动RAG API服务（纯Python模式）...")

    # 初始化 SimpleRAG
    rag = SimpleRAG()

    cache = EnhancedSimpleCache(
        ttl_seconds=Config.CACHE_TTL,
        max_size=Config.CACHE_MAX_SIZE,
        enable_stats=Config.CACHE_STATISTICS
    )

    conversation_manager = ConversationManager(
        max_history=Config.CONVERSATION_MAX_HISTORY,
        ttl_seconds=Config.CONVERSATION_TTL,
        max_sessions=Config.CONVERSATION_MAX_SESSIONS,
    )
    logger.info(f"会话管理器已初始化 | 最大历史: {Config.CONVERSATION_MAX_HISTORY}轮 | TTL: {Config.CONVERSATION_TTL}秒")

    query_logger = QueryLogger()
    logger.info("查询日志模块已初始化")

    executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

    monitor_task = asyncio.create_task(monitor_cache_stats())
    logger.info("缓存监控任务已创建")

    # 预计算快捷问答
    _init_shortcuts()

    logger.info(f"服务初始化完成（纯Python RAG模式），缓存TTL: {Config.CACHE_TTL}秒，最大工作线程: {Config.MAX_WORKERS}")


@app.on_event("shutdown")
async def shutdown_event():
    global executor, monitor_task

    logger.info("关闭RAG API服务...")

    if monitor_task:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("缓存监控任务已停止")

    if executor:
        executor.shutdown(wait=True)

    logger.info("清理完成")


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "message": "RAG API Service (纯Python，不依赖LangChain)",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "web_ui": "/web"
    }


@app.get("/web", include_in_schema=False)
async def web_ui():
    try:
        return RedirectResponse(url="/static/web_ui.html")
    except Exception as e:
        return {"error": "Web UI不可用", "detail": str(e)}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="rag-api-pure-python",
        version="2.0.0",
        timestamp=time.time()
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    if not rag:
        raise HTTPException(status_code=503, detail="服务未就绪")
    stats = rag.get_stats()
    return StatsResponse(**stats)


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"[{request_id}] 收到查询请求: {request.question[:50]}... | 用户: {request.user_id or '匿名'} | 会话: {session_id}")

    if not rag:
        logger.error(f"[{request_id}] RAG服务未就绪")
        raise HTTPException(status_code=503, detail="RAG服务未就绪")

    conversation_history_text = ""
    if conversation_manager:
        conversation_history_text = conversation_manager.format_history_for_prompt(session_id)
        if conversation_history_text:
            logger.info(f"[{request_id}] 会话历史: {len(conversation_manager.get_history(session_id)) // 2} 轮")

    cache_key = _make_cache_key(request.question, conversation_history_text)
    cached_result = cache.get(cache_key)

    if cached_result:
        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] 缓存命中 | 耗时: {processing_time:.3f}s")

        if conversation_manager:
            conversation_manager.add_exchange(session_id, request.question, cached_result["answer"])

        return QueryResponse(
            answer=cached_result["answer"],
            sources=cached_result.get("sources", []),
            success=True,
            processing_time=processing_time,
            token_usage={"cached": True},
            session_id=session_id
        )

    loop = asyncio.get_event_loop()
    try:
        logger.info(f"[{request_id}] RAG查询")
        result = await loop.run_in_executor(
            executor,
            lambda: rag.query(request.question, conversation_history_text)
        )

        answer = result.answer
        sources = result.sources
        processing_time = time.time() - start_time

        answer = _postprocess_answer(answer, request.question)

        MAX_SOURCE_LEN = 200
        trimmed_sources = []
        for src in sources:
            clean = src.replace("\n", " ").replace("\r", " ").strip()
            if len(clean) > MAX_SOURCE_LEN:
                trimmed_sources.append(clean[:MAX_SOURCE_LEN] + "...")
            else:
                trimmed_sources.append(clean)

        logger.info(f"[{request_id}] 查询成功 | 耗时: {processing_time:.3f}s")
        cache.set(cache_key, {"answer": answer, "sources": trimmed_sources})

        # 记录查询日志
        if query_logger:
            query_logger.log(
                question=request.question,
                answer=answer,
                success=True,
                session_id=session_id,
                user_id=request.user_id or "",
                processing_time=processing_time,
                tool_steps=0,
            )

        # 记录对话历史
        if conversation_manager:
            conversation_manager.add_exchange(session_id, request.question, answer)
            logger.debug(f"[{request_id}] 会话历史已更新 | 会话: {session_id}")

        return QueryResponse(
            answer=answer,
            sources=trimmed_sources,
            success=True,
            processing_time=processing_time,
            token_usage={"route": "rag"},
            session_id=session_id,
            thinking_steps=[]
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] RAG查询异常 | 耗时: {processing_time:.3f}s | 异常: {str(e)}", exc_info=True)

        # 记录错误日志
        if query_logger:
            query_logger.log(
                question=request.question,
                answer="",
                success=False,
                session_id=session_id,
                user_id=request.user_id or "",
                error_msg=str(e),
                processing_time=processing_time,
            )

        raise HTTPException(status_code=500, detail=f"RAG查询失败: {str(e)}")


@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """SSE流式查询接口 - 兼容 Web UI 前端的逐字显示"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"[{request_id}] 收到流式查询: {request.question[:50]}... | 用户: {request.user_id or '匿名'}")

    if not rag:
        logger.error(f"[{request_id}] RAG服务未就绪")
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'RAG服务未就绪'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 获取会话历史
    conversation_history_text = ""
    if conversation_manager:
        conversation_history_text = conversation_manager.format_history_for_prompt(session_id)

    # 检查缓存
    cache_key = _make_cache_key(request.question, conversation_history_text)
    cached_result = cache.get(cache_key)

    async def generate():
        try:
            if cached_result:
                answer = cached_result["answer"]
                sources = cached_result.get("sources", [])
                processing_time = time.time() - start_time
                # 发送路由信息
                yield f"data: {json.dumps({'type': 'route', 'route': 'cached'})}\n\n"
            else:
                # 在线程池中执行RAG查询（避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor,
                    lambda: rag.query(request.question, conversation_history_text)
                )
                answer = _postprocess_answer(result.answer, request.question)
                sources = result.sources

                MAX_SOURCE_LEN = 200
                trimmed_sources = []
                for src in sources:
                    clean = src.replace("\n", " ").replace("\r", " ").strip()
                    if len(clean) > MAX_SOURCE_LEN:
                        trimmed_sources.append(clean[:MAX_SOURCE_LEN] + "...")
                    else:
                        trimmed_sources.append(clean)
                sources = trimmed_sources
                processing_time = time.time() - start_time

                # 缓存结果
                cache.set(cache_key, {"answer": answer, "sources": sources})

                # 发送路由信息
                yield f"data: {json.dumps({'type': 'route', 'route': 'rag'})}\n\n"

            # 逐字流式推送回答（模拟打字机效果）
            for i in range(0, len(answer), 3):
                chunk = answer[i:i+3]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)

            # 发送来源信息
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources[:3]})}\n\n"

            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done', 'processing_time': processing_time})}\n\n"

            # 记录日志
            if query_logger:
                query_logger.log(
                    question=request.question,
                    answer=answer,
                    success=True,
                    session_id=session_id,
                    user_id=request.user_id or "",
                    processing_time=processing_time,
                    tool_steps=0,
                )

            # 更新会话历史
            if conversation_manager:
                conversation_manager.add_exchange(session_id, request.question, answer)

            logger.info(f"[{request_id}] 流式查询完成 | 耗时: {processing_time:.3f}s")

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[{request_id}] 流式查询异常: {str(e)}", exc_info=True)
            if query_logger:
                query_logger.log(
                    question=request.question,
                    answer="",
                    success=False,
                    session_id=session_id,
                    user_id=request.user_id or "",
                    error_msg=str(e),
                    processing_time=processing_time,
                )
            yield f"data: {json.dumps({'type': 'error', 'message': f'查询失败: {str(e)}'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/shortcut/{shortcut_id}")
async def shortcut_query(shortcut_id: str, session_id: Optional[str] = Query(None)):
    """快捷问答直达路由 - 跳过RAG全流程，直接返回预计算结果"""
    start_time = time.time()
    sid = session_id or str(uuid.uuid4())

    # 预计算命中 → 直接返回
    if shortcut_id in shortcut_cache:
        cached = shortcut_cache[shortcut_id]
        processing_time = time.time() - start_time
        if conversation_manager and shortcut_id in SHORTCUT_QUERIES:
            conversation_manager.add_exchange(sid, SHORTCUT_QUERIES[shortcut_id], cached["answer"])
        return QueryResponse(
            answer=cached["answer"],
            sources=cached.get("sources", []),
            success=True,
            processing_time=processing_time,
            token_usage={"route": "shortcut", "shortcut_id": shortcut_id},
            session_id=sid,
        )

    # 预计算未完成 → 降级到普通RAG查询
    if shortcut_id in SHORTCUT_QUERIES:
        question = SHORTCUT_QUERIES[shortcut_id]
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, lambda: rag.query(question))
        processing_time = time.time() - start_time
        if conversation_manager:
            conversation_manager.add_exchange(sid, question, result.answer)
        return QueryResponse(
            answer=result.answer,
            sources=result.sources[:3],
            success=True,
            processing_time=processing_time,
            token_usage={"route": "shortcut_fallback", "shortcut_id": shortcut_id},
            session_id=sid,
        )

    raise HTTPException(status_code=404, detail=f"快捷问答 {shortcut_id} 不存在")


@app.post("/batch_query")
async def batch_query(request: BatchQueryRequest):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 收到批量查询请求 | 问题数: {len(request.questions)}")

    if not rag:
        raise HTTPException(status_code=503, detail="RAG服务未就绪")

    session_id = request.session_id or str(uuid.uuid4())

    conversation_history_text = ""
    if conversation_manager:
        conversation_history_text = conversation_manager.format_history_for_prompt(session_id)
        if conversation_history_text:
            logger.info(f"[{request_id}] 会话历史: {len(conversation_manager.get_history(session_id)) // 2} 轮")

    results = []
    start_time = time.time()
    loop = asyncio.get_event_loop()

    for i, question in enumerate(request.questions):
        logger.debug(f"[{request_id}] 处理 {i+1}/{len(request.questions)}: {question[:30]}...")
        try:
            result = await loop.run_in_executor(
                executor,
                lambda q=question: rag.query(q, conversation_history_text)
            )
            answer = _postprocess_answer(result.answer, question)
            results.append({
                "question": question,
                "answer": answer,
                "success": True
            })
            # 记录对话历史
            if conversation_manager:
                conversation_manager.add_exchange(session_id, question, answer)
        except Exception as e:
            logger.error(f"[{request_id}] 批量查询第{i+1}题失败: {e}")
            results.append({
                "question": question,
                "answer": f"查询失败: {str(e)}",
                "success": False
            })

    total_time = time.time() - start_time
    logger.info(f"[{request_id}] 批量查询完成 | 总耗时: {total_time:.3f}s | 平均: {total_time / len(request.questions):.3f}s")

    return {"results": results}


@app.post("/cache/clear")
async def clear_cache():
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 清空缓存")

    if cache:
        cache.clear()
        logger.info(f"[{request_id}] 缓存已清空")
        return {"message": "缓存已清空", "status": "success"}

    return {"message": "缓存未初始化", "status": "error"}


@app.get("/cache/stats")
async def cache_stats():
    if cache:
        stats = cache.get_stats()
        stats["status"] = "active"
        return stats
    return {"status": "inactive"}


@app.post("/conversation/clear")
async def clear_conversation(session_id: str = Query(..., description="要清除历史的会话ID")):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 清除会话历史: {session_id}")

    if conversation_manager:
        success = conversation_manager.clear_history(session_id)
        if success:
            return {"message": "会话历史已清除", "session_id": session_id, "status": "success"}
        else:
            return {"message": "会话不存在", "session_id": session_id, "status": "not_found"}

    return {"message": "会话管理器未初始化", "status": "error"}


@app.get("/conversation/stats")
async def conversation_stats():
    if conversation_manager:
        stats = conversation_manager.get_stats()
        stats["status"] = "active"
        return stats
    return {"status": "inactive"}


@app.post("/reload", dependencies=[Depends(verify_admin_token)])
async def reload_workflow():
    global rag
    request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] 重新加载 RAG 系统")

    try:
        rag = SimpleRAG()
        return {"message": "RAG 系统重新加载成功", "status": "success"}
    except Exception as e:
        logger.error(f"[{request_id}] RAG 系统重新加载失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重新加载失败: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    request_id = str(uuid.uuid4())[:8]
    logger.error(f"[{request_id}] 未处理异常: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc),
            "type": type(exc).__name__,
            "request_id": request_id
        }
    )



# ═══════════════════════════════════════════
# 管理后台 API
# ═══════════════════════════════════════════

@app.get("/admin/dashboard", include_in_schema=False, dependencies=[Depends(verify_admin_token)])
async def admin_dashboard():
    """管理后台页面"""
    try:
        return RedirectResponse(url="/static/admin.html")
    except Exception as e:
        return {"error": "管理后台不可用", "detail": str(e)}


@app.get("/admin/stats", dependencies=[Depends(verify_admin_token)])
async def admin_stats(days: int = Query(7, description="统计天数")):
    """获取统计数据"""
    if not query_logger:
        raise HTTPException(status_code=503, detail="日志模块未初始化")
    return query_logger.get_stats(days=days)


@app.get("/admin/logs", dependencies=[Depends(verify_admin_token)])
async def admin_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    days: int = Query(7, description="查询天数"),
    success: Optional[bool] = Query(None, description="筛选成功/失败")
):
    """查询日志"""
    if not query_logger:
        raise HTTPException(status_code=503, detail="日志模块未初始化")
    start_time = time.time() - days * 86400
    return query_logger.get_logs(
        page=page, page_size=page_size,
        start_time=start_time, success_only=success
    )


@app.get("/admin/hot_questions", dependencies=[Depends(verify_admin_token)])
async def admin_hot_questions(
    days: int = Query(7, description="统计天数"),
    limit: int = Query(20, ge=1, le=100)
):
    """热门问题"""
    if not query_logger:
        raise HTTPException(status_code=503, detail="日志模块未初始化")
    return query_logger.get_hot_questions(days=days, limit=limit)


@app.get("/admin/errors", dependencies=[Depends(verify_admin_token)])
async def admin_errors(
    days: int = Query(7, description="统计天数"),
    limit: int = Query(50, ge=1, le=200)
):
    """错误日志"""
    if not query_logger:
        raise HTTPException(status_code=503, detail="日志模块未初始化")
    return query_logger.get_error_logs(days=days, limit=limit)


@app.post("/admin/clear_old_logs", dependencies=[Depends(verify_admin_token)])
async def admin_clear_old_logs(days: int = Query(30, description="保留天数")):
    """清理旧日志"""
    if not query_logger:
        raise HTTPException(status_code=503, detail="日志模块未初始化")
    deleted = query_logger.clear_old_logs(days=days)
    return {"deleted": deleted, "message": f"已清理 {deleted} 条超过 {days} 天的日志"}


def main():
    print("=" * 50)
    print("RAG API Service (纯Python，不依赖LangChain)")
    print("=" * 50)

    Config.print_config()

    print(f"\n启动服务: http://{Config.API_HOST}:{Config.API_PORT}")
    print(f"API文档: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    print(f"健康检查: http://{Config.API_HOST}:{Config.API_PORT}/health")
    print(f"日志文件: {Config.LOG_FILE}")
    print(f"Web UI: http://{Config.API_HOST}:{Config.API_PORT}/web")
    print(f"管理后台: http://{Config.API_HOST}:{Config.API_PORT}/admin/dashboard")
    print(f"语音接口: http://{Config.API_HOST}:{Config.API_PORT}/voice/tts")
    print("=" * 50)

    logger.info("API服务启动")


# ═══════════════════════════════════════════════════
# 语音 API（STT + TTS）
# ═══════════════════════════════════════════════════

_tts_engine_cache = None
_stt_engine_cache = None


def _get_tts_engine():
    global _tts_engine_cache
    if _tts_engine_cache is None:
        from voice_module import create_tts_engine
        _tts_engine_cache = create_tts_engine()
    return _tts_engine_cache


def _get_stt_engine():
    global _stt_engine_cache
    if _stt_engine_cache is None:
        from voice_module import create_stt_engine
        _stt_engine_cache = create_stt_engine()
    return _stt_engine_cache


@app.post("/voice/tts")
async def voice_tts(request: Request):
    """文字 → 语音合成，返回音频"""
    import tempfile
    from fastapi.responses import Response

    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    request_id = uuid.uuid4().hex[:8]
    start = time.time()

    try:
        tts = _get_tts_engine()
        audio_data = tts.synthesize(text)
        elapsed = time.time() - start
        logger.info("[%s] TTS: %d bytes, %.2fs", request_id, len(audio_data), elapsed)
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"X-Processing-Time": str(round(elapsed, 3))},
        )
    except Exception as e:
        logger.error("[%s] TTS 失败: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS 失败: {str(e)}")


@app.post("/voice/stt")
async def voice_stt(audio: UploadFile = File(...)):
    """音频 → 文字识别"""
    import tempfile

    request_id = uuid.uuid4().hex[:8]
    start = time.time()
    audio_bytes = await audio.read()

    suffix = ".wav"
    if audio.filename:
        ext = os.path.splitext(audio.filename)[1].lower()
        if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
            suffix = ext

    tmp_path = os.path.join(tempfile.gettempdir(), f"stt_{request_id}{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        stt = _get_stt_engine()
        text = stt.transcribe_file(tmp_path)
        elapsed = time.time() - start
        logger.info("[%s] STT: '%s', %.2fs", request_id, text[:50], elapsed)
        return {"text": text, "processing_time": round(elapsed, 3)}
    except Exception as e:
        logger.error("[%s] STT 失败: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT 失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    uvicorn.run(
        "api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
