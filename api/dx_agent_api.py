"""
dx_agent API 服务 (v4 - P0 生产级增强)
FastAPI 包装 dx_agent.py，复用 web_ui.html 前端
端口: 8002

P0 增强:
- 结构化日志 + PII 脱敏
- 健康检查 /healthz /readyz
- 优雅关闭 (SIGTERM/SIGINT)
- Prometheus 指标 /metrics
- 限流配额
- 熔断器集成
- trace_id 贯穿请求链
"""

import os
import signal
import sys
import time
import uuid
import json
import asyncio
import logging
import re
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# ── 路径引导：无论从哪个工作目录启动都能找到 core/infra/api ──
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_BASE_DIR, "core"), os.path.join(_BASE_DIR, "infra"), os.path.join(_BASE_DIR, "api")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import extract_plan_tiers, get_config_reloader, get_key_rotator
from dx_agent import (
    run_single, _classify_query, _session_ctx, _get_workflow,
    _get_llm_client, _get_history_text, _rewrite_query,
    _fast_path_rag, _fast_path_comparison, _deduplicate_answer,
    _record_exchange, SYSTEM_PROMPT, _get_openai_tools, _execute_tool,
    PurePythonAgent, _COMPARISON_PROMPT
)
from config import Config
from conversation_history import ConversationManager

# P0: 结构化日志
from structured_logging import setup_structured_logging, set_trace_id, set_log_session_id
from metrics import (
    record_request, record_tool_call, record_cache_hit, record_cache_miss,
    record_llm_tokens, update_circuit_state, update_active_sessions,
    set_service_info, get_metrics_bytes, get_metrics_content_type,
)
from rate_limiter import get_rate_limiter
from circuit_breaker import get_llm_breaker, CircuitBreakerOpenError

setup_structured_logging(
    level=Config.LOG_LEVEL,
    json_output=Config.LOG_JSON,
    log_file=Config.LOG_FILE,
)
logger = logging.getLogger("dx_agent_api")

app = FastAPI(title="运营商套餐 Agent API", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

# P0: Redis 会话存储（可选）
conversation_manager = ConversationManager(
    max_history=Config.CONVERSATION_MAX_HISTORY,
    ttl_seconds=Config.CONVERSATION_TTL,
    max_sessions=Config.CONVERSATION_MAX_SESSIONS,
    redis_url=Config.REDIS_URL,
)

# P0: 服务状态追踪
_service_ready = False
_service_start_time = 0.0


# P0: 中间件 — trace_id + 限流 + 请求指标
@app.middleware("http")
async def p0_middleware(request: Request, call_next):
    global _service_start_time
    # 生成 trace_id
    trace_id = request.headers.get("X-Trace-Id", "") or uuid.uuid4().hex[:16]
    set_trace_id(trace_id)

    # P0: 限流检查
    user_key = request.headers.get("X-User-Id", "") or request.client.host if request.client else "unknown"
    rate_limiter = get_rate_limiter()
    allowed, retry_after = rate_limiter.check(user_key)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Too Many Requests", "retry_after": round(retry_after, 1)},
            headers={"Retry-After": str(int(retry_after) + 1), "X-Trace-Id": trace_id},
        )

    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        # 记录指标
        route = request.url.path
        status = "success" if response.status_code < 400 else "error"
        record_request(route=route, status=status, duration=duration)
        response.headers["X-Trace-Id"] = trace_id
        return response
    except Exception as e:
        duration = time.time() - start
        record_request(route=request.url.path, status="error", duration=duration)
        raise

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    success: bool = True
    processing_time: Optional[float] = None
    session_id: Optional[str] = None
    route: Optional[str] = None

@app.on_event("startup")
async def startup():
    global _service_ready, _service_start_time
    _service_start_time = time.time()
    logger.info("正在预热 dx_agent ...")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(executor, _get_workflow)
        logger.info("dx_agent 预热完成")
    except Exception as e:
        logger.error("dx_agent 预热失败: %s", e)

    # P0: 配置热加载
    try:
        reloader = get_config_reloader()
        reloader.start()
    except Exception as e:
        logger.warning("配置热加载启动失败: %s", e)

    # P0: 设置服务信息
    set_service_info(version="4.0", model=Config.LLM_MODEL)
    _service_ready = True
    logger.info("dx_agent 服务就绪")


# P0: 优雅关闭
@app.on_event("shutdown")
async def shutdown():
    logger.info("dx_agent 正在关闭...")
    # 停止配置热加载
    try:
        reloader = get_config_reloader()
        reloader.stop()
    except Exception:
        pass
    # 关闭线程池
    executor.shutdown(wait=True, cancel_futures=False)
    logger.info("dx_agent 已关闭")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/web_ui.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "dx_agent", "model": Config.LLM_MODEL}


# P0: 健康检查端点
@app.get("/healthz")
async def healthz():
    """存活检查：LLM 连通性、缓存可达性"""
    checks = {}
    # 检查 LLM 客户端
    try:
        client = _get_llm_client()
        checks["llm"] = "ok"
    except Exception as e:
        checks["llm"] = f"error: {e}"

    # 检查熔断器状态
    breaker = get_llm_breaker()
    from circuit_breaker import CircuitState
    breaker_state = breaker.state
    checks["circuit_breaker"] = breaker_state.value
    update_circuit_state("llm_api", {"closed": 0, "open": 1, "half_open": 2}[breaker_state.value])

    healthy = checks["llm"] == "ok" and breaker_state != CircuitState.OPEN
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "unhealthy", "checks": checks},
    )


@app.get("/readyz")
async def readyz():
    """就绪检查：数据是否加载完成"""
    if not _service_ready:
        return JSONResponse(status_code=503, content={"status": "not_ready", "message": "服务预热中"})
    return {"status": "ready"}


# P0: Prometheus 指标端点
@app.get("/metrics")
async def metrics():
    data = get_metrics_bytes()
    if data:
        return Response(content=data, media_type=get_metrics_content_type())
    return Response(content=b"# prometheus_client not installed\n", media_type="text/plain")

@app.get("/stats")
async def stats():
    try:
        wf = _get_workflow()
        s = wf.get_stats()
        return {
            "status": s.get("status", "unknown"),
            "document_count": s.get("document_count", 0),
            "model": Config.LLM_MODEL,
            "embedding_model": Config.EMBED_MODEL,
            "service": "dx_agent",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/conversation/clear")
async def clear_conversation(session_id: str = Query(...)):
    conversation_manager.clear_history(session_id)
    return {"success": True, "message": "会话已清除"}

# ── 非流式接口（保留兼容）──
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())[:12]
    token = _session_ctx.set(session_id)
    set_log_session_id(session_id)
    start = time.time()

    # P0: 熔断器检查
    breaker = get_llm_breaker()
    if not breaker.allow_request():
        return JSONResponse(
            status_code=503,
            content={"answer": "服务暂时不可用，请稍后重试", "success": False,
                      "processing_time": 0, "session_id": session_id, "route": "circuit_open"},
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor, lambda: run_single(request.question, verbose=False)
        )
        elapsed = time.time() - start
        route = _classify_query(request.question)
        breaker.record_success()
        return QueryResponse(
            answer=result["answer"], success=result["success"],
            processing_time=round(elapsed, 2), session_id=session_id, route=route,
        )
    except Exception as e:
        elapsed = time.time() - start
        breaker.record_failure()
        logger.error("查询异常: %s", e, exc_info=True)
        return QueryResponse(
            answer=f"查询出错: {str(e)}", success=False,
            processing_time=round(elapsed, 2), session_id=session_id,
        )
    finally:
        _session_ctx.reset(token)

# ── 流式接口 ──
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())[:12]
    original_query = request.question
    set_log_session_id(session_id)

    # P0: 熔断器检查
    breaker = get_llm_breaker()
    if not breaker.allow_request():
        async def _circuit_open():
            yield f"data: {json.dumps({'type': 'error', 'message': '服务暂时不可用（熔断中），请稍后重试'})}\n\n"
        return StreamingResponse(_circuit_open(), media_type="text/event-stream")

    async def event_generator():
        token = _session_ctx.set(session_id)
        start = time.time()
        try:
            # 查询重写 + 路由
            query = _rewrite_query(original_query)
            route = _classify_query(query)
            yield f"data: {json.dumps({'type': 'route', 'route': route})}\n\n"

            if route == "simple":
                # 简单查询：RAG 检索后流式分段返回
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    executor, lambda: _fast_path_rag(query)
                )
                answer = _deduplicate_answer(answer)
                _record_exchange(original_query, answer)
                # 按句子分块流式返回，模拟打字机效果
                import re as _re
                chunks = _re.split(r'(?<=[。！？])', answer)
                for chunk in chunks:
                    if chunk.strip():
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.08)

            elif route == "comparison":
                # 对比查询：使用 SimpleRAG 检索（无档位过滤）+ 流式生成
                from dx_agent import _get_simple_rag, _resolve_shifu_to_tiers
                rag = _get_simple_rag()
                shifu_tiers = _resolve_shifu_to_tiers(query)
                direct_tiers = extract_plan_tiers(query)
                tiers = shifu_tiers if shifu_tiers else direct_tiers

                loop = asyncio.get_event_loop()

                if len(tiers) >= 2:
                    seen = set()
                    docs = []
                    for tier in tiers:
                        tier_query = f"{tier}元套餐 流量 通话 宽带 副卡 月租"
                        tier_docs = await loop.run_in_executor(
                            executor,
                            lambda tq=tier_query: rag.hybrid_retriever.search(tq, Config.RETRIEVAL_K)
                        )
                        for d in tier_docs:
                            key = d.page_content.strip()
                            if key not in seen:
                                seen.add(key)
                                docs.append(d)
                else:
                    docs = await loop.run_in_executor(
                        executor,
                        lambda: rag.hybrid_retriever.search(query, Config.RETRIEVAL_K)
                    )

                context = "\n\n".join(doc.page_content for doc in docs[:8])
                prompt = _COMPARISON_PROMPT.format(context=context, question=query)

                client = _get_llm_client()
                response = client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=Config.AGENT_TEMPERATURE,
                    max_tokens=Config.AGENT_MAX_TOKENS,
                    stream=True,
                )
                full_answer = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_answer += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                if full_answer:
                    _record_exchange(original_query, full_answer)

            else:
                # Agent 复杂查询：流式 + 工具调用
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in conversation_manager.get_history(session_id):
                    if msg["role"] in ("user", "assistant"):
                        messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": query})

                tools = _get_openai_tools()
                client = _get_llm_client()

                for iteration in range(Config.AGENT_MAX_ITERATIONS):
                    response = client.chat.completions.create(
                        model=Config.LLM_MODEL,
                        messages=messages,
                        tools=tools if tools else None,
                        tool_choice="auto" if tools else None,
                        temperature=Config.AGENT_TEMPERATURE,
                        max_tokens=Config.AGENT_MAX_TOKENS,
                        stream=True,
                    )

                    full_content = ""
                    tool_calls_data = {}
                    has_tool_calls = False

                    for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta

                        if delta.tool_calls:
                            has_tool_calls = True
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_data:
                                    tool_calls_data[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                                if tc.function:
                                    if tc.function.name:
                                        tool_calls_data[idx]["name"] = tc.function.name
                                    if tc.function.arguments:
                                        tool_calls_data[idx]["arguments"] += tc.function.arguments

                        if delta.content:
                            full_content += delta.content
                            yield f"data: {json.dumps({'type': 'token', 'content': delta.content})}\n\n"

                    if has_tool_calls and tool_calls_data:
                        for idx in sorted(tool_calls_data.keys()):
                            tc = tool_calls_data[idx]
                            tool_name = tc["name"]
                            try:
                                tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except json.JSONDecodeError:
                                tool_args = {}

                            yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'args': tool_args})}\n\n"

                            result = await asyncio.get_event_loop().run_in_executor(
                                executor, lambda n=tool_name, a=tool_args: _execute_tool(n, a)
                            )

                            yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_name, 'result': result[:300]})}\n\n"

                            messages.append({
                                "role": "assistant", "content": None,
                                "tool_calls": [{
                                    "id": tc["id"] or f"call_{idx}",
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": json.dumps(tool_args, ensure_ascii=False)}
                                }]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"] or f"call_{idx}",
                                "content": result
                            })
                        continue
                    else:
                        if full_content:
                            _record_exchange(original_query, full_content)
                        break

            elapsed = time.time() - start
            yield f"data: {json.dumps({'type': 'done', 'processing_time': round(elapsed, 2)})}\n\n"

        except Exception as e:
            logger.error("流式查询异常: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            _session_ctx.reset(token)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ── 挂载静态文件（统一使用项目根目录 static/）──
static_dir = os.path.join(_BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    # P0: 优雅关闭 — 注册信号处理器
    def _signal_handler(signum, frame):
        logger.info("收到信号 %s，准备优雅关闭...", signum)
        # uvicorn 内部会处理 shutdown 事件

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    uvicorn.run(app, host="0.0.0.0", port=8002)
