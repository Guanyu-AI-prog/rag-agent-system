"""
Prometheus 指标模块 (P0)
采集: requests_total, latency_seconds, tool_calls_total, cache_hit_ratio, llm_token_total

prometheus_client 为可选依赖，未安装时降级为空操作。
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client 未安装，指标采集降级为空操作。安装: pip install prometheus-client")

# ── 指标定义 ──

if PROMETHEUS_AVAILABLE:
    # 请求计数
    REQUESTS_TOTAL = Counter(
        "dx_agent_requests_total",
        "总请求数",
        ["route", "status"]
    )

    # 请求延迟
    REQUEST_LATENCY = Histogram(
        "dx_agent_request_latency_seconds",
        "请求延迟（秒）",
        ["route"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )

    # 工具调用计数
    TOOL_CALLS_TOTAL = Counter(
        "dx_agent_tool_calls_total",
        "工具调用总数",
        ["tool_name", "status"]
    )

    # 缓存命中
    CACHE_HITS = Counter("dx_agent_cache_hits_total", "缓存命中次数", ["cache_name"])
    CACHE_MISSES = Counter("dx_agent_cache_misses_total", "缓存未命中次数", ["cache_name"])

    # LLM Token 消耗
    LLM_TOKENS = Counter(
        "dx_agent_llm_tokens_total",
        "LLM Token 消耗",
        ["type"]  # prompt / completion
    )

    # 熔断器状态
    CIRCUIT_STATE = Gauge(
        "dx_agent_circuit_breaker_state",
        "熔断器状态 (0=closed, 1=open, 2=half_open)",
        ["name"]
    )

    # 活跃会话数
    ACTIVE_SESSIONS = Gauge("dx_agent_active_sessions", "活跃会话数")

    # 服务信息
    SERVICE_INFO = Info("dx_agent", "服务信息")
else:
    # 降级空操作
    class _NoopMetric:
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

    REQUESTS_TOTAL = _NoopMetric()
    REQUEST_LATENCY = _NoopMetric()
    TOOL_CALLS_TOTAL = _NoopMetric()
    CACHE_HITS = _NoopMetric()
    CACHE_MISSES = _NoopMetric()
    LLM_TOKENS = _NoopMetric()
    CIRCUIT_STATE = _NoopMetric()
    ACTIVE_SESSIONS = _NoopMetric()
    SERVICE_INFO = _NoopMetric()


def record_request(route: str, status: str, duration: float):
    """记录一次请求"""
    REQUESTS_TOTAL.labels(route=route, status=status).inc()
    REQUEST_LATENCY.labels(route=route).observe(duration)


def record_tool_call(tool_name: str, success: bool):
    """记录一次工具调用"""
    status = "success" if success else "error"
    TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status=status).inc()


def record_cache_hit(cache_name: str):
    CACHE_HITS.labels(cache_name=cache_name).inc()


def record_cache_miss(cache_name: str):
    CACHE_MISSES.labels(cache_name=cache_name).inc()


def record_llm_tokens(prompt_tokens: int = 0, completion_tokens: int = 0):
    if prompt_tokens:
        LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(type="completion").inc(completion_tokens)


def update_circuit_state(name: str, state_value: int):
    """更新熔断器状态指标 (0=closed, 1=open, 2=half_open)"""
    CIRCUIT_STATE.labels(name=name).set(state_value)


def update_active_sessions(count: int):
    ACTIVE_SESSIONS.set(count)


def set_service_info(version: str, model: str, env: str = "production"):
    SERVICE_INFO.info({"version": version, "model": model, "env": env})


@contextmanager
def track_latency(route: str):
    """上下文管理器：跟踪代码块延迟并自动记录"""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        REQUEST_LATENCY.labels(route=route).observe(duration)


def get_metrics_bytes() -> Optional[bytes]:
    """获取 Prometheus 格式的指标数据"""
    if PROMETHEUS_AVAILABLE:
        return generate_latest()
    return None


def get_metrics_content_type() -> str:
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain"
