"""
限流配额模块 (P0)
令牌桶算法：按用户/IP 分级配额，超配额返回 429 + Retry-After
"""

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    令牌桶限流器

    Args:
        rate: 每秒补充的令牌数
        capacity: 桶容量（突发上限）
        name: 限流器名称
    """

    def __init__(self, rate: float, capacity: float, name: str = "default"):
        self.name = name
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def try_acquire(self, tokens: int = 1) -> tuple[bool, float]:
        """
        尝试获取令牌

        Returns:
            (allowed, retry_after_seconds)
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True, 0.0
            # 计算需要等待多久
            wait = (tokens - self._tokens) / self.rate
            return False, wait

    def get_stats(self) -> dict:
        with self._lock:
            self._refill()
            return {
                "name": self.name,
                "tokens": round(self._tokens, 2),
                "capacity": self.capacity,
                "rate": self.rate,
            }


class RateLimiter:
    """
    分级限流管理器

    支持：
    - 全局限流（所有请求共享）
    - 按 key 限流（用户/IP 级别）
    """

    def __init__(
        self,
        global_rate: float = 100.0,     # 全局每秒请求数
        global_capacity: float = 200.0,  # 全局突发上限
        per_user_rate: float = 10.0,     # 单用户每秒请求数
        per_user_capacity: float = 20.0, # 单用户突发上限
        per_user_ttl: int = 3600,        # 用户桶过期时间
    ):
        self._global = TokenBucket(global_rate, global_capacity, name="global")
        self._per_user_rate = per_user_rate
        self._per_user_capacity = per_user_capacity
        self._per_user_ttl = per_user_ttl
        self._user_buckets: Dict[str, tuple[TokenBucket, float]] = {}
        self._lock = threading.Lock()

    def _get_user_bucket(self, key: str) -> TokenBucket:
        """获取或创建用户级令牌桶"""
        now = time.time()
        with self._lock:
            if key in self._user_buckets:
                bucket, last_access = self._user_buckets[key]
                self._user_buckets[key] = (bucket, now)
                return bucket
            # 过期清理
            expired = [k for k, (_, t) in self._user_buckets.items() if now - t > self._per_user_ttl]
            for k in expired:
                del self._user_buckets[k]
            bucket = TokenBucket(self._per_user_rate, self._per_user_capacity, name=f"user:{key}")
            self._user_buckets[key] = (bucket, now)
            return bucket

    def check(self, user_key: str = "default") -> tuple[bool, Optional[float]]:
        """
        检查是否允许请求

        Args:
            user_key: 用户标识（user_id 或 IP）

        Returns:
            (allowed, retry_after_seconds) — retry_after 为 None 表示无限制
        """
        # 全局限流
        allowed, retry_after = self._global.try_acquire()
        if not allowed:
            logger.warning("全局限流触发，retry_after=%.1fs", retry_after)
            return False, retry_after

        # 用户级限流
        user_bucket = self._get_user_bucket(user_key)
        allowed, retry_after = user_bucket.try_acquire()
        if not allowed:
            logger.warning("用户 [%s] 限流触发，retry_after=%.1fs", user_key, retry_after)
            return False, retry_after

        return True, None

    def get_stats(self) -> dict:
        stats = {"global": self._global.get_stats()}
        with self._lock:
            stats["active_users"] = len(self._user_buckets)
        return stats


# ── 全局限流器实例 ──

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        from config import Config
        _rate_limiter = RateLimiter(
            global_rate=getattr(Config, "RATE_LIMIT_GLOBAL_RATE", 100.0),
            global_capacity=getattr(Config, "RATE_LIMIT_GLOBAL_CAPACITY", 200.0),
            per_user_rate=getattr(Config, "RATE_LIMIT_PER_USER_RATE", 10.0),
            per_user_capacity=getattr(Config, "RATE_LIMIT_PER_USER_CAPACITY", 20.0),
        )
    return _rate_limiter
