"""
结构化日志模块 (P0)
- JSON 格式输出，可被 ELK/Loki/LPG 消费
- trace_id / session_id 贯穿请求链
- PII 脱敏过滤器（手机号、身份证号）
"""

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

# ── trace_id 上下文变量 ──
_trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")
_session_id_ctx: ContextVar[str] = ContextVar("session_id", default="")


def set_trace_id(trace_id: str = "") -> str:
    """设置当前请求的 trace_id，返回实际值"""
    tid = trace_id or uuid.uuid4().hex[:16]
    _trace_id_ctx.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_ctx.get("")


def set_log_session_id(session_id: str):
    _session_id_ctx.set(session_id)


def get_log_session_id() -> str:
    return _session_id_ctx.get("")


# ── PII 脱敏 ──

class PIIFilter(logging.Filter):
    """日志过滤器：脱敏手机号、身份证号、银行卡号"""

    # 手机号: 138****0000
    _PHONE = re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')
    # 身份证号: 18位或15位
    _ID_CARD = re.compile(r'(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)')
    # 银行卡号: 16-19位数字
    _BANK_CARD = re.compile(r'(?<!\d)(\d{4})(\d{8,12})(\d{4})(?!\d)')

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._mask(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._mask(str(v)) if isinstance(v, str) else v
                               for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._mask(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True

    @classmethod
    def _mask(cls, text: str) -> str:
        text = cls._PHONE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[7:], text)
        text = cls._ID_CARD.sub(lambda m: m.group(1) + "********" + m.group(3), text)
        text = cls._BANK_CARD.sub(lambda m: m.group(1) + "****" + m.group(3), text)
        return text


# ── JSON 格式化器 ──

class JSONFormatter(logging.Formatter):
    """输出 JSON 格式日志，包含 timestamp、level、trace_id、session_id、logger、message"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
            "session_id": get_log_session_id(),
        }

        # 附加额外字段
        if hasattr(record, "route"):
            log_entry["route"] = record.route
        if hasattr(record, "cost_ms"):
            log_entry["cost_ms"] = record.cost_ms
        if hasattr(record, "tool_name"):
            log_entry["tool_name"] = record.tool_name

        # 异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


# ── 初始化 ──

def setup_structured_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
):
    """
    配置全局日志

    Args:
        level: 日志级别
        json_output: 是否输出 JSON 格式（生产环境建议 True）
        log_file: 日志文件路径（可选）
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清除已有 handler
    root.handlers.clear()

    # PII 脱敏过滤器
    pii_filter = PIIFilter()

    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        )

    # 控制台输出
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(pii_filter)
    root.addHandler(console)

    # 文件输出（可选）
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(JSONFormatter())  # 文件始终用 JSON
        file_handler.addFilter(pii_filter)
        root.addHandler(file_handler)

    return root
