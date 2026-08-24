import json
import time
import logging
from threading import Lock
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# Redis 会话后端 (P0)
# ═══════════════════════════════════════════════════

class RedisSessionBackend:
    """Redis 会话存储后端，支持 TTL 自动过期"""

    def __init__(self, redis_url: str, ttl_seconds: int = 1800, key_prefix: str = "dx_session:"):
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()  # 验证连接
            self._available = True
            logger.info("Redis 会话后端已连接: %s", redis_url)
        except Exception as e:
            logger.warning("Redis 连接失败，降级到内存存储: %s", e)
            self._available = False
            self._redis = None
        self.ttl = ttl_seconds
        self.prefix = key_prefix

    @property
    def available(self) -> bool:
        return self._available

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        if not self._available:
            return []
        try:
            data = self._redis.get(self._key(session_id))
            if data:
                self._redis.expire(self._key(session_id), self.ttl)  # 续期
                return json.loads(data)
            return []
        except Exception as e:
            logger.error("Redis get 失败: %s", e)
            return []

    def save_messages(self, session_id: str, messages: List[Dict[str, str]]):
        if not self._available:
            return
        try:
            self._redis.setex(self._key(session_id), self.ttl, json.dumps(messages, ensure_ascii=False))
        except Exception as e:
            logger.error("Redis set 失败: %s", e)

    def delete_session(self, session_id: str) -> bool:
        if not self._available:
            return False
        try:
            return bool(self._redis.delete(self._key(session_id)))
        except Exception as e:
            logger.error("Redis delete 失败: %s", e)
            return False

    def get_session_count(self) -> int:
        if not self._available:
            return 0
        try:
            return len(self._redis.keys(f"{self.prefix}*"))
        except Exception:
            return 0


# ═══════════════════════════════════════════════════
# 会话管理器（支持 Redis / 内存双后端）
# ═══════════════════════════════════════════════════

class ConversationManager:
    """会话历史管理器，支持多会话、自动过期和清理
    P0: 支持 Redis 后端（可选，降级到内存）
    """

    def __init__(self, max_history: int = 5, ttl_seconds: int = 1800, max_sessions: int = 1000,
                 redis_url: str = ""):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self.max_history = max_history
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions

        # P0: Redis 后端（可选）
        self._redis_backend: Optional[RedisSessionBackend] = None
        if redis_url:
            self._redis_backend = RedisSessionBackend(redis_url, ttl_seconds)
            if self._redis_backend.available:
                logger.info("会话存储: Redis 后端已启用")
            else:
                logger.info("会话存储: Redis 不可用，使用内存存储")
                self._redis_backend = None
        else:
            logger.info("会话存储: 内存存储 (max_sessions=%d, ttl=%ds)", max_sessions, ttl_seconds)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        # P0: 优先从 Redis 读取
        if self._redis_backend:
            messages = self._redis_backend.get_messages(session_id)
            if messages:
                return messages

        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session['last_access'] = time.time()
                return list(session['messages'])
            return []

    def add_exchange(self, session_id: str, question: str, answer: str):
        # P0: Redis 后端
        if self._redis_backend:
            messages = self._redis_backend.get_messages(session_id)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
            if len(messages) > self.max_history * 2:
                messages = messages[-(self.max_history * 2):]
            self._redis_backend.save_messages(session_id, messages)
            return

        with self.lock:
            if session_id not in self.sessions:
                self._evict_if_needed_locked()
                self.sessions[session_id] = {
                    'messages': [],
                    'created': time.time(),
                    'last_access': time.time()
                }
            session = self.sessions[session_id]
            session['messages'].append({"role": "user", "content": question})
            session['messages'].append({"role": "assistant", "content": answer})
            session['last_access'] = time.time()

            if len(session['messages']) > self.max_history * 2:
                session['messages'] = session['messages'][-(self.max_history * 2):]

    def clear_history(self, session_id: str) -> bool:
        if self._redis_backend:
            return self._redis_backend.delete_session(session_id)

        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

    def format_history_for_prompt(self, session_id: str) -> str:
        messages = self.get_history(session_id)
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _evict_if_needed_locked(self):
        """调用前必须已持有 self.lock"""
        if len(self.sessions) >= self.max_sessions:
            now = time.time()
            expired = [
                sid for sid, s in self.sessions.items()
                if now - s['last_access'] > self.ttl
            ]
            if expired:
                for sid in expired:
                    del self.sessions[sid]
                logger.info(f"清理了 {len(expired)} 个过期会话")
            while len(self.sessions) >= self.max_sessions:
                oldest = min(self.sessions.items(), key=lambda x: x[1]['last_access'])
                del self.sessions[oldest[0]]
                logger.info(f"会话数已达上限，淘汰最旧会话: {oldest[0]}")

    def cleanup_expired(self):
        with self.lock:
            self._evict_if_needed_locked()

    def get_stats(self) -> Dict[str, Any]:
        # P0: 包含 Redis 状态
        if self._redis_backend:
            session_count = self._redis_backend.get_session_count()
            return {
                "active_sessions": session_count,
                "backend": "redis",
                "max_history_per_session": self.max_history,
                "ttl_seconds": self.ttl,
                "max_sessions": self.max_sessions
            }

        with self.lock:
            total_messages = sum(len(s['messages']) for s in self.sessions.values())
            return {
                "active_sessions": len(self.sessions),
                "total_messages": total_messages,
                "backend": "memory",
                "max_history_per_session": self.max_history,
                "ttl_seconds": self.ttl,
                "max_sessions": self.max_sessions
            }
