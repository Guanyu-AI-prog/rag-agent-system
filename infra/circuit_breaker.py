"""
熔断器模块 (P0)
防止 LLM API 级联雪崩：连续失败 N 次后熔断 M 秒，期间快速失败。

状态机: CLOSED → OPEN → HALF_OPEN → CLOSED
"""

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"        # 正常：请求通过
    OPEN = "open"            # 熔断：快速失败
    HALF_OPEN = "half_open"  # 半开：放一个请求试探


class CircuitBreaker:
    """
    线程安全的熔断器

    Args:
        failure_threshold: 连续失败多少次后触发熔断
        recovery_timeout: 熔断后多少秒进入半开状态
        half_open_max_calls: 半开状态允许的最大试探调用数
        name: 熔断器名称（用于日志）
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        name: str = "default",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否到了恢复时间
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("熔断器 [%s] OPEN → HALF_OPEN（恢复超时 %.1fs）", self.name, self.recovery_timeout)
            return self._state

    def record_success(self):
        """记录一次成功调用"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("熔断器 [%s] HALF_OPEN → CLOSED（恢复正常）", self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self):
        """记录一次失败调用"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态失败 → 重新熔断
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning("熔断器 [%s] HALF_OPEN → OPEN（试探失败）", self.name)
            elif self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                # 连续失败达阈值 → 熔断
                self._state = CircuitState.OPEN
                logger.error(
                    "熔断器 [%s] CLOSED → OPEN（连续失败 %d 次，熔断 %.1fs）",
                    self.name, self._failure_count, self.recovery_timeout
                )

    def allow_request(self) -> bool:
        """判断是否允许本次请求通过"""
        state = self.state  # 触发状态检查
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        # OPEN
        return False

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


def circuit_protected(breaker: CircuitBreaker):
    """装饰器：用熔断器保护函数调用"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not breaker.allow_request():
                raise CircuitBreakerOpenError(
                    f"熔断器 [{breaker.name}] 已打开，请求被拒绝"
                )
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        return wrapper
    return decorator


class CircuitBreakerOpenError(Exception):
    """熔断器打开时抛出的异常"""
    pass


# ── 全局熔断器实例 ──

_llm_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=1,
    name="llm_api",
)

_rag_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=20.0,
    half_open_max_calls=1,
    name="rag_query",
)


def get_llm_breaker() -> CircuitBreaker:
    return _llm_breaker


def get_rag_breaker() -> CircuitBreaker:
    return _rag_breaker
