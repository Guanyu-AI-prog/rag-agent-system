"""
缓存管理模块
提供统一的缓存接口，支持查询结果缓存、嵌入缓存、BM25/Rerank 缓存
"""

import hashlib
import logging
import re
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EnhancedSimpleCache:
    """增强版内存缓存：支持统计和O(1) LRU淘汰（基于OrderedDict）"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000, enable_stats: bool = True):
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = Lock()
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.enable_stats = enable_stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    # O(1): 移到末尾表示最近使用
                    self.cache.move_to_end(key)
                    if self.enable_stats:
                        self.hits += 1
                    return entry['value']
                else:
                    # 过期删除
                    del self.cache[key]
            if self.enable_stats:
                self.misses += 1
            return None

    def set(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                # 更新已有 key，移到末尾
                self.cache.move_to_end(key)
            else:
                # LRU 淘汰：删除最旧的条目
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                    if self.enable_stats:
                        self.evictions += 1

            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }

    def clear(self):
        with self.lock:
            self.cache.clear()
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


def make_query_cache_key(query: str) -> str:
    """
    生成查询缓存键

    规则：
    - 标准化查询（去空格、转小写）
    - 去除标点差异
    - 不同会话共享相同问题的缓存（套餐信息是公共的）
    """
    # 标准化：去空格、转小写、去标点
    normalized = query.strip().lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[？！。，、；：""''（）\[\]【】]', '', normalized)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def make_rerank_cache_key(query: str, doc_contents: list) -> str:
    """
    生成 Rerank 缓存键

    基于查询和文档内容前50字符的哈希
    """
    query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
    doc_hash = hashlib.md5(
        "".join(d[:50] for d in doc_contents).encode('utf-8')
    ).hexdigest()[:8]
    return f"rerank:{query_hash}:{doc_hash}"


# 全局缓存实例（延迟初始化）
_query_cache: Optional[EnhancedSimpleCache] = None
_rerank_cache: Optional[EnhancedSimpleCache] = None
_embedding_cache: Optional[EnhancedSimpleCache] = None


def get_query_cache() -> EnhancedSimpleCache:
    """获取查询结果缓存实例"""
    global _query_cache
    if _query_cache is None:
        from config import Config
        _query_cache = EnhancedSimpleCache(
            ttl_seconds=Config.CACHE_TTL,
            max_size=Config.CACHE_MAX_SIZE,
            enable_stats=Config.CACHE_STATISTICS
        )
        logger.info("查询缓存已初始化 (TTL=%ds, MaxSize=%d)", Config.CACHE_TTL, Config.CACHE_MAX_SIZE)
    return _query_cache


def get_rerank_cache() -> EnhancedSimpleCache:
    """获取 Rerank 结果缓存实例"""
    global _rerank_cache
    if _rerank_cache is None:
        from config import Config
        _rerank_cache = EnhancedSimpleCache(
            ttl_seconds=Config.CACHE_TTL * 2,  # Rerank 结果缓存更久
            max_size=500,
            enable_stats=Config.CACHE_STATISTICS
        )
        logger.info("Rerank 缓存已初始化")
    return _rerank_cache


def get_embedding_cache() -> EnhancedSimpleCache:
    """获取嵌入向量缓存实例"""
    global _embedding_cache
    if _embedding_cache is None:
        from config import Config
        _embedding_cache = EnhancedSimpleCache(
            ttl_seconds=Config.CACHE_TTL * 3,  # 嵌入缓存更久
            max_size=2000,
            enable_stats=Config.CACHE_STATISTICS
        )
        logger.info("嵌入缓存已初始化")
    return _embedding_cache


def clear_all_caches():
    """清除所有缓存"""
    if _query_cache:
        _query_cache.clear()
    if _rerank_cache:
        _rerank_cache.clear()
    if _embedding_cache:
        _embedding_cache.clear()
    logger.info("所有缓存已清除")


def get_all_cache_stats() -> Dict[str, Any]:
    """获取所有缓存统计信息"""
    stats = {}
    if _query_cache:
        stats["query_cache"] = _query_cache.get_stats()
    if _rerank_cache:
        stats["rerank_cache"] = _rerank_cache.get_stats()
    if _embedding_cache:
        stats["embedding_cache"] = _embedding_cache.get_stats()
    return stats
