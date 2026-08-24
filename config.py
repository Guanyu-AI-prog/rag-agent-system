"""
RAG系统配置管理
基于LangChain + SiliconFlow API

P0 增强:
- 配置热加载: 监听 .env 文件变更，自动 reload
- API Key 轮转: 支持多 key 轮转（逗号分隔）
"""

import logging
import os
import re
import threading
import time
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """系统配置类"""

    # ============== API配置 ==============
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")

    LLM_API_KEY = os.getenv("LLM_API_KEY", "") or SILICONFLOW_API_KEY
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.siliconflow.cn/v1")

    DINGTALK_APP_KEY = os.getenv("DINGTALK_APP_KEY", "")
    DINGTALK_APP_SECRET = os.getenv("DINGTALK_APP_SECRET", "")

    # ============== 管理接口鉴权 ==============
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "sObaXSN0NFKzFbPDuBP_dLwZYeZ6ZXqyFKycPJ29NOY")

    # ============== 模型配置 ==============
    LLM_MODEL = os.getenv("LLM_MODEL", "step-3.5-flash")
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 1024

    EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    USE_RERANK = os.getenv("USE_RERANK", "True").lower() == "true"
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))  # 降低：减少 rerank 文档数
    RERANK_API_URL = os.getenv("RERANK_API_URL", "https://api.siliconflow.cn/v1/rerank")
    RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "10"))  # 10s，兼顾速度和成功率
    RERANK_MAX_RETRIES = int(os.getenv("RERANK_MAX_RETRIES", "1"))  # 降低：减少重试次数

    # ============== 向量库配置 ==============
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain_collection")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
    CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "100"))  # 低于此长度的 chunk 与相邻 chunk 合并
    CHUNK_SEPARATORS = ["\n\n", "###", "####", "#####", "\n", "。", "！", "？", "；", "，", " "]

    # 分类型切分配置：小chunk(Q&A) / 大chunk(详情) / 默认(流程)
    CHUNK_PROFILES = {
        "small": {"chunk_size": 250, "chunk_overlap": 50},
        "large": {"chunk_size": 500, "chunk_overlap": 100},
        "default": {"chunk_size": 300, "chunk_overlap": 60},
    }
    # 文件名关键词 → chunk profile 映射
    CHUNK_FILE_RULES = {
        "small": ["Q&A", "qa_plans", "transfer_faq", "搭配，Q&A"],
        "large": ["套餐详情", "搭配表_重构", "plan_details", "套餐搭配表.csv"],
    }

    # 套餐级别切分：用"一、销售品内容"作为分隔符，确保每个套餐的完整信息在一个chunk里
    PLAN_LEVEL_CHUNK_FILES = ["套餐详情"]  # 文件名包含这些关键词的文件使用套餐级别切分
    PLAN_LEVEL_CHUNK_SEPARATOR = r"(?=一、\s*销售品内容)"  # 正则分隔符
    PLAN_LEVEL_MAX_CHUNK_SIZE = 800  # 单个套餐最大chunk size，超过则按子标题二次切分

    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
    COMPARISON_RETRIEVAL_K = int(os.getenv("COMPARISON_RETRIEVAL_K", "50"))  # 降低：100 → 50
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "3"))  # 降低：5 → 3
    VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))  # 降低：15 → 8，减少 rerank 输入

    # ============== 查询配置 ==============
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10"))

    # ============== 查询重写配置 ==============
    QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "True").lower() == "true"
    QUERY_REWRITE_MAX_HISTORY = int(os.getenv("QUERY_REWRITE_MAX_HISTORY", "3"))
    QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "step-3.5-flash")
    QUERY_REWRITE_TEMPERATURE = float(os.getenv("QUERY_REWRITE_TEMPERATURE", "0.0"))
    QUERY_REWRITE_MAX_TOKENS = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "256"))

    INTENT_EXPANSIONS = {
        "wifi": "宽带 无线",
        "WiFi": "宽带 无线",
        "兆": "宽带 速率 M",
        "宽带": "宽带 安装 融合 兆",
        "无线": "宽带 wifi",
        "流量": "流量 基础 通用 赠送 GB 套餐",
        "流量不足": "流量 叠加包 升级",
        "不够用": "升级 叠加 套餐变更",
        "便宜": "优惠 低价 折扣 划算",
        "多少钱": "月费 月基本费 价格 资费",
        "送什么": "赠送 优惠 礼品 活动",
        "橙分期": "直降金额 月权益金 分期 合约",
        "补贴": "直降 月权益金 合约 预存",
        "直降": "橙分期 权益金",
        "副卡": "副卡 数量 费用 共享 张",
        "最多": "上限 最大 最多 可办理",
        "套外": "套外 超出 额外 收费 标准",
        "转网": "携号转网 转入 流程 授权码",
        "投诉": "投诉 渠道 工信部 10086",
        "定向流量": "定向流量 应用 APP 免流",
        "全家": "家庭 副卡 共享 全家享",
        "购机": "购机 合约 直降 补贴 优惠",
        "降档": "降档 降低 变更 套餐变更",
        "星卡": "星卡 定向流量 39元 学生",
        "方案": "全额预存 橙分期 互斥 优惠方案",
        "全额预存": "预存 话费赠送 合约 月返",
        "学生": "星卡 39元 学生 低月租",
    }

    QUERY_SYNONYMS = {
        "流量达人": "流量达人升级包",
        "合约促销": "合约促销体验包",
    }

    COMPARISON_KEYWORDS = [
        "月基本费", "月费", "流量", "语音", "通话", "分钟", "宽带",
        "国内通用流量", "国内通话", "套餐方案", "套餐内包含",
        "融合", "安装", "副卡", "共享", "合约", "承诺", "转网", "携号"
    ]

    # ============== Agent配置 ==============
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1024"))
    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    ANSWER_MAX_LENGTH = int(os.getenv("ANSWER_MAX_LENGTH", "3000"))

    # ============== 会话历史配置 ==============
    CONVERSATION_MAX_HISTORY = int(os.getenv("CONVERSATION_MAX_HISTORY", "5"))
    CONVERSATION_TTL = int(os.getenv("CONVERSATION_TTL", "1800"))
    CONVERSATION_MAX_SESSIONS = int(os.getenv("CONVERSATION_MAX_SESSIONS", "1000"))

    # ============== 性能配置 ==============
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))  # 增加：支持并行检索
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "15"))  # 15s，对比查询需要更多时间

    CACHE_TTL = int(os.getenv("CACHE_TTL", "600"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    CACHE_STATISTICS = os.getenv("CACHE_STATISTICS", "True").lower() == "true"

    # ============== 重试配置 ==============
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
    LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "2.0"))
    LLM_RETRY_MAX_DELAY = float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0"))

    # ============== 服务配置 ==============
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8001"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

    # ============== 路径配置 ==============
    DATA_DIR = os.getenv("DATA_DIR", "./data")

    # ============== 日志配置 ==============
    LOG_FILE = os.getenv("LOG_FILE", "./logs/api.log")
    LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", "10"))  # MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    LOG_DIR = os.getenv("LOG_DIR", "./logs")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s"
    LOG_JSON = os.getenv("LOG_JSON", "False").lower() == "true"  # P0: JSON 结构化日志

    # ============== 限流配置 (P0) ==============
    RATE_LIMIT_GLOBAL_RATE = float(os.getenv("RATE_LIMIT_GLOBAL_RATE", "100"))
    RATE_LIMIT_GLOBAL_CAPACITY = float(os.getenv("RATE_LIMIT_GLOBAL_CAPACITY", "200"))
    RATE_LIMIT_PER_USER_RATE = float(os.getenv("RATE_LIMIT_PER_USER_RATE", "10"))
    RATE_LIMIT_PER_USER_CAPACITY = float(os.getenv("RATE_LIMIT_PER_USER_CAPACITY", "20"))

    # ============== 熔断器配置 (P0) ==============
    CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    CIRCUIT_BREAKER_RECOVERY = float(os.getenv("CIRCUIT_BREAKER_RECOVERY", "30"))

    # ============== Redis 配置 (P0，可选) ==============
    REDIS_URL = os.getenv("REDIS_URL", "")  # e.g. redis://localhost:6379/0

    # ============== API Key 轮转 (P0) ==============
    # 支持逗号分隔的多 key，自动轮转
    LLM_API_KEYS = [k.strip() for k in os.getenv("LLM_API_KEYS", "").split(",") if k.strip()]
    LLM_API_BASES = [b.strip() for b in os.getenv("LLM_API_BASES", "").split(",") if b.strip()]

    @classmethod
    def validate(cls):
        if not cls.SILICONFLOW_API_KEY and not cls.LLM_API_KEY:
            raise ValueError("请配置 SILICONFLOW_API_KEY 或 LLM_API_KEY")

        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        return True

    # ============== 套餐档位常量（单一数据源） ==============
    PLAN_TIERS = ['29', '39', '59', '79', '99', '129', '169', '199', '229', '299']
    PLAN_PRICE_MIN = 19
    PLAN_PRICE_MAX = 599

    @classmethod
    def print_config(cls):
        print("=== RAG系统配置 ===")
        print(f"LLM模型: {cls.LLM_MODEL}")
        print(f"嵌入模型: {cls.EMBED_MODEL}")
        print(f"向量库路径: {cls.VECTOR_DB_PATH}")
        print(f"API服务: {cls.API_HOST}:{cls.API_PORT}")
        print(f"最大工作线程: {cls.MAX_WORKERS}")
        print(f"缓存时间: {cls.CACHE_TTL}秒")
        print(f"缓存最大条目: {cls.CACHE_MAX_SIZE}")
        print(f"缓存统计: {'启用' if cls.CACHE_STATISTICS else '禁用'}")
        print(f"Agent最大迭代: {cls.AGENT_MAX_ITERATIONS}")
        print(f"Agent详细日志: {'启用' if cls.AGENT_VERBOSE else '禁用'}")
        print(f"会话历史轮次: {cls.CONVERSATION_MAX_HISTORY}轮")
        print(f"会话过期时间: {cls.CONVERSATION_TTL}秒")
        print(f"最大会话数: {cls.CONVERSATION_MAX_SESSIONS}")
        print(f"CORS Origins: {cls.CORS_ORIGINS}")
        print("=" * 30)


config = Config()


def extract_plan_tier(query: str) -> str | None:
    """从查询中提取单个套餐档位数字（如 '59套餐包含多少流量' → '59'）。

    优先匹配已知档位；若不在已知列表中，只要在合理价格范围内也接受。
    返回档位字符串或 None。
    """
    patterns = [
        r'(\d+)\s*元?\s*套餐',
        r'套餐\s*(\d+)',
        r'畅享\s*(\d+)',
        r'(\d+)\s*元',
    ]
    for p in patterns:
        m = re.search(p, query)
        if m:
            num = m.group(1)
            if num in Config.PLAN_TIERS:
                return num
            try:
                price = int(num)
                if Config.PLAN_PRICE_MIN <= price <= Config.PLAN_PRICE_MAX:
                    return num
            except ValueError:
                continue
    return None


def extract_plan_tiers(query: str) -> list[str]:
    """从查询中提取所有套餐档位数字列表（用于对比场景）。"""
    tiers = []
    for num in re.findall(r'(\d+)', query):
        if num in Config.PLAN_TIERS and num not in tiers:
            tiers.append(num)
    return tiers


if __name__ == "__main__":
    try:
        Config.validate()
        Config.print_config()
        print("配置验证通过")
    except Exception as e:
        print(f"配置错误: {e}")


# ═══════════════════════════════════════════════════
# API Key 轮转器 (P0)
# ═══════════════════════════════════════════════════

class APIKeyRotator:
    """
    API Key 轮转管理器

    支持多个 API Key 轮转使用，遇到 429 时自动切换到下一个 key。
    使用 round-robin + 429 触发切换策略。
    """

    def __init__(self, keys: List[str], bases: Optional[List[str]] = None):
        if not keys:
            raise ValueError("至少需要一个 API Key")
        self._keys = list(keys)
        self._bases = list(bases) if bases else [""] * len(keys)
        # 确保 bases 和 keys 长度一致
        while len(self._bases) < len(self._keys):
            self._bases.append(self._bases[-1] if self._bases else "")
        self._index = 0
        self._lock = threading.Lock()

    @property
    def current_key(self) -> str:
        with self._lock:
            return self._keys[self._index]

    @property
    def current_base(self) -> str:
        with self._lock:
            return self._bases[self._index]

    def rotate(self) -> tuple[str, str]:
        """轮转到下一个 key，返回 (key, base_url)"""
        with self._lock:
            self._index = (self._index + 1) % len(self._keys)
            logger.info("API Key 轮转到索引 %d/%d", self._index, len(self._keys))
            return self._keys[self._index], self._bases[self._index]

    def on_rate_limit(self) -> tuple[str, str]:
        """遇到 429 时调用，自动轮转"""
        logger.warning("遇到 429 限流，触发 API Key 轮转")
        return self.rotate()

    @property
    def key_count(self) -> int:
        return len(self._keys)


# 全局轮转器实例（延迟初始化）
_key_rotator: Optional[APIKeyRotator] = None


def get_key_rotator() -> Optional[APIKeyRotator]:
    """获取 API Key 轮转器（如果配置了多 key）"""
    global _key_rotator
    if _key_rotator is None:
        keys = Config.LLM_API_KEYS
        if keys and len(keys) > 1:
            _key_rotator = APIKeyRotator(keys, Config.LLM_API_BASES or None)
            logger.info("API Key 轮转器已初始化，共 %d 个 key", len(keys))
    return _key_rotator


# ═══════════════════════════════════════════════════
# 配置热加载 (P0)
# ═══════════════════════════════════════════════════

class ConfigReloader:
    """
    配置热加载器：监听 .env 文件变更，自动重载配置

    使用 os.stat 轮询方式，无需 watchdog 依赖。
    """

    def __init__(self, env_path: str = ".env", check_interval: float = 5.0):
        self.env_path = os.path.abspath(env_path)
        self.check_interval = check_interval
        self._last_mtime = self._get_mtime()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list = []

    def _get_mtime(self) -> float:
        try:
            return os.stat(self.env_path).st_mtime
        except OSError:
            return 0.0

    def on_reload(self, callback):
        """注册配置变更回调"""
        self._callbacks.append(callback)

    def _check_loop(self):
        while self._running:
            time.sleep(self.check_interval)
            mtime = self._get_mtime()
            if mtime > self._last_mtime:
                logger.info("检测到 .env 文件变更，重新加载配置...")
                self._last_mtime = mtime
                try:
                    load_dotenv(override=True)
                    for cb in self._callbacks:
                        try:
                            cb()
                        except Exception as e:
                            logger.error("配置变更回调失败: %s", e)
                    logger.info("配置热加载完成")
                except Exception as e:
                    logger.error("配置热加载失败: %s", e)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True, name="config-reloader")
        self._thread.start()
        logger.info("配置热加载已启动 (检查间隔: %.1fs, 文件: %s)", self.check_interval, self.env_path)

    def stop(self):
        self._running = False


_config_reloader: Optional[ConfigReloader] = None


def get_config_reloader() -> ConfigReloader:
    global _config_reloader
    if _config_reloader is None:
        _config_reloader = ConfigReloader()
    return _config_reloader
