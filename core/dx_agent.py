#!/usr/bin/env python3
"""
运营商套餐 Agent — 纯 Python 实现（无 LangChain 依赖）
使用 OpenAI 兼容 API + 正则匹配工具调用

架构：
  Agent (纯 Python 自主规划)
    ├── Tool: 套餐知识查询 → RAGWorkflow.query()
    ├── Tool: 费用计算器   → simpleeval 安全计算
    └── Tool: 套餐统计     → RAGWorkflow.get_stats()

用法:
  python taocan_agent_pure.py              # 进入交互模式
  python taocan_agent_pure.py "59套餐多少流量"  # 单次查询
"""

import contextvars
import csv
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import uuid
import warnings
from functools import wraps
from typing import Any, Callable, TypedDict

from openai import OpenAI
from simpleeval import simple_eval

from config import Config, extract_plan_tier, extract_plan_tiers, get_key_rotator
from conversation_history import ConversationManager
from simple_rag import SimpleRAG as RAGWorkflow
from simple_rag import SimpleRAG
from cache_manager import get_query_cache, make_query_cache_key, clear_all_caches, get_all_cache_stats

# P0: 结构化日志 + 熔断器 + 指标
from structured_logging import setup_structured_logging, set_trace_id, set_log_session_id
from circuit_breaker import get_llm_breaker, get_rag_breaker, CircuitBreakerOpenError
from metrics import record_tool_call, record_llm_tokens

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ──────────── 日志 (P0: 结构化 + PII 脱敏) ────────────
setup_structured_logging(
    level=Config.LOG_LEVEL,
    json_output=Config.LOG_JSON,
    log_file=Config.LOG_FILE,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 工具系统（简单字典实现，替代 OOP 注册表）
# ═══════════════════════════════════════════════════

# 工具字典：{name: {"func": callable, "openai": dict}}
_TOOLS: dict[str, dict] = {}


def _get_openai_tools() -> list[dict]:
    """获取 OpenAI 格式的工具列表"""
    return [t["openai"] for t in _TOOLS.values()]


def _execute_tool(name: str, arguments: dict) -> str:
    """执行指定工具（P0: 增加指标采集）"""
    tool = _TOOLS.get(name)
    if not tool:
        record_tool_call(name, success=False)
        return f"错误: 未知工具 '{name}'"
    try:
        result = tool["func"](**arguments)
        record_tool_call(name, success=True)
        return result
    except Exception as e:
        record_tool_call(name, success=False)
        logger.error("工具 %s 执行失败: %s", name, e, exc_info=True)
        return f"工具执行错误: {e}"


# ═══════════════════════════════════════════════════
# 结构化套餐数据加载（启动时加载一次）
# ═══════════════════════════════════════════════════

_struct_plans: dict[str, dict] = {}       # 基础套餐: {月租_str: {字段...}}
_struct_csv_plans: dict[str, dict] = {}   # CSV搭配表: {套餐_方案: {字段...}}
_struct_loaded = False
_struct_lock = threading.Lock()


def _load_struct_data():
    """启动时加载 plan_details.jsonl 和 套餐搭配表.csv 到内存。"""
    global _struct_loaded
    if _struct_loaded:
        return
    with _struct_lock:
        if _struct_loaded:
            return
        data_dir = Config.DATA_DIR

        # 加载 plan_details.jsonl（基础套餐）
        jsonl_path = os.path.join(data_dir, "plan_details.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("_type") == "基础套餐":
                            tier = str(obj.get("月租", ""))
                            if tier:
                                _struct_plans[tier] = obj
                    except json.JSONDecodeError:
                        pass
            logger.info("结构化数据加载: %d 个基础套餐", len(_struct_plans))

        # 加载 套餐搭配表.csv
        csv_path = os.path.join(data_dir, "套餐搭配表.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = f"{row.get('套餐', '')}_{row.get('方案', '')}"
                    if key:
                        _struct_csv_plans[key] = dict(row)
            logger.info("结构化数据加载: %d 条CSV搭配记录", len(_struct_csv_plans))

        _struct_loaded = True


# 对比维度字段映射（JSONL字段 → 中文标签）
_COMPARE_FIELDS = [
    ("套餐名称", "套餐"),
    ("月租", "月租"),
    ("国内通用流量", "通用流量"),
    ("国内流量", "国内流量"),
    ("定向流量", "定向流量"),
    ("语音", "通话"),
    ("国内语音拨打", "通话"),
    ("国内通话", "通话"),
    ("副卡", "副卡数"),
    ("副卡费用", "副卡费用"),
    ("会员档次", "会员档次"),
    ("宽带", "宽带"),
    ("套外流量", "套外流量"),
    ("套外语音", "套外通话"),
    ("转网", "转网规则"),
]


def _format_plan_comparison(tiers: list[str]) -> str:
    """根据档位列表，从结构化数据中拼出对比表。返回纯文本。"""
    _load_struct_data()
    parts = []
    found_any = False

    for tier in tiers:
        plan = _struct_plans.get(tier)
        if not plan:
            parts.append(f"【{tier}元套餐】暂无结构化数据")
            continue
        found_any = True
        lines = [f"【{plan.get('套餐名称', tier + '元套餐')}】"]
        for field, label in _COMPARE_FIELDS:
            val = plan.get(field)
            if val and field != "套餐名称":
                lines.append(f"  {label}: {val}")
        # 搭配方案信息
        for scheme in ["全额预存", "橙分期"]:
            csv_key = f"{tier}_{scheme}"
            csv_row = _struct_csv_plans.get(csv_key)
            if csv_row:
                lines.append(f"  --- {scheme}方案 ---")
                lines.append(f"    实付: {csv_row.get('实付', '?')}元/月")
                lines.append(f"    赠送流量: {csv_row.get('赠送流量', '?')}GB")
                lines.append(f"    最大流量: {csv_row.get('最大流量', '?')}GB")
        parts.append("\n".join(lines))

    if not found_any:
        return ""

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════
# 重试工具：指数退避 + 429 限流检测
# ═══════════════════════════════════════════════════

def _is_rate_limit_error(exc: Exception) -> bool:
    """判断是否为 429 限流错误"""
    err_str = str(exc).lower()
    return "429" in err_str or "rate limit" in err_str or "too many requests" in err_str


def _retry_on_rate_limit(func):
    """装饰器：对 429 限流错误进行指数退避重试（P0: 加随机抖动 + API Key 轮转）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = Config.LLM_MAX_RETRIES
        delay = Config.LLM_RETRY_DELAY
        backoff = Config.LLM_RETRY_BACKOFF
        max_delay = Config.LLM_RETRY_MAX_DELAY

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e) and attempt < max_retries:
                    # P0: 加随机抖动，防止重试风暴
                    wait = min(delay * (backoff ** attempt), max_delay)
                    jittered_wait = wait * (0.5 + random.random())
                    logger.warning(
                        "遇到 429 限流，第 %d/%d 次重试，等待 %.1fs...",
                        attempt + 1, max_retries, jittered_wait
                    )
                    # P0: API Key 轮转
                    rotator = get_key_rotator()
                    if rotator:
                        try:
                            rotator.on_rate_limit()
                        except Exception:
                            pass
                    time.sleep(jittered_wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
    return wrapper


# ═══════════════════════════════════════════════════
# 全局实例（延迟初始化，double-checked locking 保证线程安全）
# ═══════════════════════════════════════════════════

# 使用 contextvars 替代全局变量，保证并发安全
_session_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="cli_default")


def _lazy_init(lock, holder: dict, key: str, factory):
    """通用延迟初始化：double-checked locking 模板"""
    if holder[key] is None:
        with lock:
            if holder[key] is None:
                holder[key] = factory()
    return holder[key]


_workflow_lock = threading.Lock()
_workflow_holder = {"inst": None}

_simple_rag_lock = threading.Lock()
_simple_rag_holder = {"inst": None}

_conversation_mgr_lock = threading.Lock()
_conversation_mgr_holder = {"inst": None}

_llm_client_lock = threading.Lock()
_llm_client_holder = {"inst": None}


def _get_workflow() -> RAGWorkflow:
    """延迟初始化 RAGWorkflow 单例（线程安全）"""
    def _create():
        logger.info("初始化 RAGWorkflow ...")
        return RAGWorkflow()
    return _lazy_init(_workflow_lock, _workflow_holder, "inst", _create)


def _get_simple_rag() -> SimpleRAG:
    """延迟初始化 SimpleRAG 单例（线程安全）"""
    def _create():
        logger.info("初始化 SimpleRAG ...")
        return SimpleRAG()
    return _lazy_init(_simple_rag_lock, _simple_rag_holder, "inst", _create)


def _get_conversation_mgr() -> ConversationManager:
    """延迟初始化 ConversationManager 单例（线程安全）"""
    def _create():
        return ConversationManager(
            max_history=Config.CONVERSATION_MAX_HISTORY,
            ttl_seconds=Config.CONVERSATION_TTL,
            max_sessions=Config.CONVERSATION_MAX_SESSIONS,
        )
    return _lazy_init(_conversation_mgr_lock, _conversation_mgr_holder, "inst", _create)


def _get_llm_client() -> OpenAI:
    """延迟初始化 OpenAI 客户端（线程安全）(P0: 支持 Key 轮转)"""
    def _create():
        rotator = get_key_rotator()
        if rotator:
            api_key = rotator.current_key
            api_base = rotator.current_base or Config.LLM_API_BASE or Config.SILICONFLOW_API_BASE
        else:
            api_key = Config.LLM_API_KEY or Config.SILICONFLOW_API_KEY
            api_base = Config.LLM_API_BASE or Config.SILICONFLOW_API_BASE
        logger.info("初始化 OpenAI 客户端: %s", api_base)
        return OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=Config.LLM_TIMEOUT,
            max_retries=0
        )
    return _lazy_init(_llm_client_lock, _llm_client_holder, "inst", _create)


def _get_history_text() -> str:
    """获取当前会话的历史记录文本"""
    mgr = _get_conversation_mgr()
    return mgr.format_history_for_prompt(_session_ctx.get())


def _get_history_messages_for_rewrite(limit: int = None) -> list[dict]:
    """获取用于查询重写的历史消息（最近 N 轮）"""
    mgr = _get_conversation_mgr()
    messages = mgr.get_history(_session_ctx.get())
    if limit and len(messages) > limit * 2:
        messages = messages[-(limit * 2):]
    result = []
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            result.append({"role": msg["role"], "content": msg["content"]})
    return result


_QUERY_REWRITE_PROMPT = """你是查询改写助手。根据对话历史和当前查询，将当前查询改写为完整、自包含、检索友好的查询。

【改写规则】
1. 指代消解：将"它/那个/这个/刚才/上一个/上面"等指代词替换为具体实体（套餐名、档位、运营商等）
2. 省略补全：补全被省略的关键信息（如"59元那个多少流量" → "59套餐包含多少流量"）
3. 同义扩展：补充常用同义词（如"话费"→"月租 月基本费 资费"、"宽带"→"宽带 融合 安装"）
4. 保持意图：不改变用户核心意图，不添加原查询无关内容
5. 简洁优先：改写后查询控制在 50 字以内

【对话历史】
{history}

【当前查询】
{query}

【改写后查询】（仅输出改写后的查询，不要解释，不要格式化）："""


def _rewrite_query(query: str) -> str:
    """
    查询重写/改写：利用对话历史将当前查询改写为自包含、检索友好的形式。
    处理：指代消解、省略补全、同义扩展。
    """
    if not Config.QUERY_REWRITE_ENABLED:
        return query

    history_messages = _get_history_messages_for_rewrite(Config.QUERY_REWRITE_MAX_HISTORY)
    if not history_messages:
        return query

    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
        for m in history_messages
    )

    prompt = _QUERY_REWRITE_PROMPT.format(history=history_text, query=query)

    try:
        @_retry_on_rate_limit
        def _do_rewrite():
            client = _get_llm_client()
            response = client.chat.completions.create(
                model=Config.QUERY_REWRITE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.QUERY_REWRITE_TEMPERATURE,
                max_tokens=Config.QUERY_REWRITE_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

        rewritten = _do_rewrite()
        # 基础清理：去除可能的引号、前缀
        rewritten = rewritten.strip('"\'').strip()
        if rewritten and len(rewritten) <= Config.MAX_QUERY_LENGTH:
            logger.info("查询重写: '%s' → '%s'", query, rewritten)
            return rewritten
        logger.warning("查询重写结果异常，使用原查询: %s", rewritten)
        return query

    except Exception as e:
        logger.warning("查询重写失败，使用原查询: %s", e)
        return query


def _record_exchange(question: str, answer: str):
    """记录一轮对话到历史"""
    mgr = _get_conversation_mgr()
    mgr.add_exchange(_session_ctx.get(), question, answer)


def _get_history_messages() -> list[dict]:
    """获取当前会话的历史记录，转换为 OpenAI 消息格式"""
    mgr = _get_conversation_mgr()
    messages = mgr.get_history(_session_ctx.get())
    result = []
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            result.append({"role": msg["role"], "content": msg["content"]})
    return result


# ═══════════════════════════════════════════════════
# 数据隔离：防止跨套餐数据混淆
# ═══════════════════════════════════════════════════

def _deduplicate_answer(answer: str) -> str:
    """
    去除回答中的重复段落。
    检测连续重复的句子/段落并只保留一份。
    """
    if not answer:
        return answer

    lines = answer.split('\n')
    deduped = []
    seen_chunks = set()
    current_chunk = []

    for line in lines:
        stripped = line.strip()
        # 遇到空行或标记行时，处理当前块
        if not stripped or stripped.startswith('|') or stripped.startswith('---') or stripped.startswith('#'):
            if current_chunk:
                chunk_key = '\n'.join(current_chunk).strip()
                if chunk_key and chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    deduped.extend(current_chunk)
                current_chunk = []
            deduped.append(line)
        else:
            current_chunk.append(line)

    # 处理最后一个块
    if current_chunk:
        chunk_key = '\n'.join(current_chunk).strip()
        if chunk_key and chunk_key not in seen_chunks:
            deduped.extend(current_chunk)

    result = '\n'.join(deduped)

    # 检测并截断过长的回答（可能是重复导致）
    max_len = Config.ANSWER_MAX_LENGTH
    if len(result) > max_len:
        # 尝试在段落边界截断
        paragraphs = result.split('\n\n')
        truncated = []
        total_len = 0
        for p in paragraphs:
            if total_len + len(p) > max_len * 0.9:
                break
            truncated.append(p)
            total_len += len(p)
        if truncated:
            result = '\n\n'.join(truncated)
            if not result.endswith(('。', '）', '！', '？', '…', '|')):
                result += '\n\n（以上为精简展示，完整信息请咨询客服）'

    return result


# ═══════════════════════════════════════════════════
# 工具实现
# ═══════════════════════════════════════════════════


def _tool_套餐知识查询(query: str) -> str:
    """套餐知识查询工具实现（使用 SimpleRAG 检索，更稳定）(P0: 熔断器保护)"""
    # P0: RAG 熔断器检查
    rag_breaker = get_rag_breaker()
    if not rag_breaker.allow_request():
        return "知识库查询暂时不可用（熔断中），请稍后重试。"

    rag = _get_simple_rag()

    @_retry_on_rate_limit
    def _do_query(q: str):
        return rag.query(q)

    try:
        result = _do_query(query)
        if not result.success:
            rag_breaker.record_failure()
            return result.answer if result.answer else "查询失败，请稍后重试。"
        rag_breaker.record_success()
        return result.answer

    except Exception as e:
        rag_breaker.record_failure()
        logger.error("套餐知识查询异常: %s", e, exc_info=True)
        return "查询出错，请稍后重试。如持续失败请联系对应运营商客服。"


def _tool_套餐对比直达(plan_numbers: str) -> str:
    """套餐对比直达工具实现"""
    # 解析输入
    numbers = [n.strip() for n in re.split(r'[,，、\s]+', plan_numbers) if n.strip()]
    if len(numbers) < 2:
        return "请至少提供两个套餐档位进行对比，例如 '59,129'。"

    result = _format_plan_comparison(numbers)
    if not result:
        return f"未找到套餐 {', '.join(numbers)} 的结构化数据，请用「套餐知识查询」工具检索。"

    return result


# 计算器安全函数（模块级常量，避免每次调用重建）
_SAFE_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": pow, "int": int, "float": float, "sum": sum,
}
_SAFE_NAMES = {"pi": math.pi, "e": math.e}
_SAFE_NAMES_BY_KEY = set(_SAFE_FUNCTIONS.keys()) | set(_SAFE_NAMES.keys())


def _tool_费用计算器(expression: str) -> str:
    """费用计算器工具实现"""

    expr = expression.strip()
    expr = expr.replace("×", "*").replace("÷", "/")

    if "__" in expr:
        return "计算出错: 表达式包含非法模式。"

    for token in re.findall(r"[a-zA-Z_]\w*", expr):
        if token not in _SAFE_NAMES_BY_KEY:
            safe_list = "、".join(sorted(_SAFE_NAMES_BY_KEY))
            return f"计算出错: 未知的函数或变量「{token}」。仅支持: {safe_list}"

    if re.search(r"[a-zA-Z_]\w*\.", expr):
        return "计算出错: 不支持属性访问操作。"

    try:
        result = simple_eval(expr, functions=_SAFE_FUNCTIONS, names=_SAFE_NAMES)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算出错: {e}。请检查表达式格式，例如 '39 * 12' 或 '129 - 40'。"


def _tool_套餐统计() -> str:
    """套餐统计工具实现"""
    workflow = _get_workflow()
    try:
        stats = workflow.get_stats()
        cache_stats = get_all_cache_stats()

        result_lines = []
        if stats.get("status") == "ready":
            result_lines.extend([
                f"知识库状态: 就绪",
                f"文档块数量: {stats.get('document_count', '未知')}",
                f"LLM模型: {stats.get('llm_model', '未知')}",
                f"嵌入模型: {stats.get('embedding_model', '未知')}",
                f"分块大小: {stats.get('chunk_size', '未知')}",
                f"检索数量: {stats.get('retrieval_k', '未知')}",
            ])
        else:
            result_lines.append(f"知识库状态: {stats.get('status', '未知')}")

        # 添加缓存统计
        if cache_stats:
            result_lines.append("\n--- 缓存统计 ---")
            for name, s in cache_stats.items():
                result_lines.append(
                    f"{name}: 命中率={s['hit_rate']}, "
                    f"大小={s['size']}/{s['max_size']}, "
                    f"命中={s['hits']}, 未命中={s['misses']}"
                )

        return "\n".join(result_lines)
    except Exception as e:
        return f"获取统计信息失败: {e}"


def _register_tools():
    """注册所有工具"""
    _TOOLS["套餐知识查询"] = {
        "func": _tool_套餐知识查询,
        "openai": {
            "type": "function",
            "function": {
                "name": "套餐知识查询",
                "description": "查询运营商套餐知识库，检索套餐的流量、通话、宽带、资费等详细信息。"
                               "输入应为关于套餐的具体问题，如'59套餐包含多少流量'、'129有宽带吗'。"
                               "也适用于查询办理流程、转网规则、投诉渠道、副卡政策、套外收费、橙分期补贴等。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "关于套餐的具体问题"}},
                    "required": ["query"]
                }
            }
        }
    }
    _TOOLS["套餐对比直达"] = {
        "func": _tool_套餐对比直达,
        "openai": {
            "type": "function",
            "function": {
                "name": "套餐对比直达",
                "description": "对比两个或多个套餐的结构化数据（流量、通话、宽带、副卡、资费等）。"
                               "直接从结构化数据读取，无需检索，速度极快。"
                               "输入格式：套餐档位数字，用逗号分隔。例如 '59,129' 或 '99,199,299'。",
                "parameters": {
                    "type": "object",
                    "properties": {"plan_numbers": {"type": "string", "description": "套餐档位数字，用逗号分隔，如 '59,129'"}},
                    "required": ["plan_numbers"]
                }
            }
        }
    }
    _TOOLS["费用计算器"] = {
        "func": _tool_费用计算器,
        "openai": {
            "type": "function",
            "function": {
                "name": "费用计算器",
                "description": "计算数学表达式，用于计算套餐费用、年消费、折扣差价等。"
                               "支持 + - * / ** 和 sqrt/abs/round 等函数。"
                               "示例: '39*12'、'129-40'、'39*12+9.9*12'",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                    "required": ["expression"]
                }
            }
        }
    }
    _TOOLS["套餐统计"] = {
        "func": _tool_套餐统计,
        "openai": {
            "type": "function",
            "function": {
                "name": "套餐统计",
                "description": "查询当前知识库的统计信息，包括文档数量、模型配置、缓存状态等。",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
    }


# 模块加载时注册工具（一次性，无延迟）
_register_tools()


# ═══════════════════════════════════════════════════
# 纯 Python Agent 实现
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个运营商套餐智能助手。你的职责是帮助用户查询套餐信息和计算费用。

重要规则：
1. 套餐相关问题（流量、通话、宽带、资费、副卡、转网、投诉等），使用「套餐知识查询」工具
2. 费用计算（年费、差价、折扣），先用「套餐知识查询」获取数据，再用「费用计算器」计算
3. 【优先】对比两个或多个套餐时，使用「套餐对比直达」工具，输入档位数字如 "59,129"。速度快、数据准确。只有当「套餐对比直达」返回数据不足时，才用「套餐知识查询」补充
4. 无关问题，直接回答"抱歉，我只能回答运营商套餐相关的问题"
5. 回答要简洁友好，基于真实数据，不要编造
6. 如果没有相关信息，坦诚告知并建议联系对应运营商客服
7. 【费用计算规则】计算总费用时，必须使用套餐原价（月基本费），不要使用"实付价"或"优惠后价格"。只有当用户明确问"实付多少"、"优惠后多少"时，才使用对应方案的价格
8. 【档位真实性】只能推荐知识库中实际存在的套餐档位，禁止编造不存在的档位。如果30元以内没有合适套餐，如实告知"目前最低档位是29元"
9. 【角色边界】你是电信套餐助手，只能推荐电信的套餐。注意："携号转网"是电信核心业务，用户说"移动号/联通号想转过来"是指要转到电信来，应正常推荐电信套餐并说明转网规则。只有当用户明确问"移动有什么套餐"、"联通套餐推荐"时才拒绝

【数据来源约束 — 极其重要】
- 所有数字（补贴金额、月租、流量、通话等）必须来自工具返回的结果
- 禁止自行推算、估算或编造任何数字
- 如果工具未返回某项具体数据（如"月权益金"、"每月返还金额"），必须回答"该信息需咨询对应运营商客服"
- 禁止发明知识库中不存在的概念（如"月权益金"、"月返"等）

【互斥方案隔离 — 极其重要】
- "全额预存"和"橙分期"是两种互斥方案，用户只能选择其中一种
- 绝不能将两种方案的价格或优惠混在一起计算
- 橙分期的补贴是一次性购机补贴，不是每月返还
- 全额预存的"实付XX元"是话费赠送后的月租，与橙分期无关
- 对比套餐时，必须明确区分是哪种方案下的数据

【工具调用格式】
当你需要调用工具时，请严格按照以下格式输出：

<tool_call>
{"name": "工具名称", "arguments": {"参数名": "参数值"}}
</tool_call>

例如：
<tool_call>
{"name": "套餐知识查询", "arguments": {"query": "59套餐包含多少流量"}}
</tool_call>

请只在需要调用工具时使用 <tool_call> 标签，其他时候直接回答用户问题。"""


class PurePythonAgent:
    """纯 Python Agent 实现"""

    def __init__(self, max_iterations: int = None):
        self.client = _get_llm_client()
        self.model = Config.LLM_MODEL
        self.max_iterations = max_iterations or Config.AGENT_MAX_ITERATIONS
        self.temperature = Config.AGENT_TEMPERATURE
        self.max_tokens = Config.AGENT_MAX_TOKENS

    def _parse_tool_call(self, text: str) -> tuple[str, dict] | None:
        """
        从 LLM 输出中解析工具调用
        返回 (工具名, 参数字典) 或 None

        使用精确的 start/end 查找替代正则，防止跨块误匹配。
        """
        start_tag = "<tool_call>"
        end_tag = "</think>"  # 修复: 使用精确查找替代正则

        start_idx = text.find(start_tag)
        if start_idx == -1:
            return None

        json_start = start_idx + len(start_tag)
        end_idx = text.find(end_tag, json_start)
        if end_idx == -1:
            return None

        json_str = text[json_start:end_idx].strip()
        try:
            data = json.loads(json_str)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                logger.warning("工具调用格式异常: name=%s, arguments=%s", type(name), type(arguments))
                return None
            return name, arguments
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("解析工具调用失败: %s, json_str=%s", e, json_str[:200])
            return None

    def _call_llm(self, messages: list[dict], tools: list[dict] = None) -> dict:
        """调用 LLM API (P0: Token 指标采集)"""
        @_retry_on_rate_limit
        def _do_call():
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            # P0: 记录 Token 消耗
            if hasattr(response, 'usage') and response.usage:
                record_llm_tokens(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                )
            return response

        return _do_call()

    def run(self, query: str, verbose: bool = True) -> str:
        """
        运行 Agent 处理用户查询

        Args:
            query: 用户问题
            verbose: 是否输出详细日志

        Returns:
            Agent 的回答
        """
        # 构建消息列表
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 添加历史对话
        history = _get_history_messages()
        messages.extend(history)

        # 添加当前问题
        messages.append({"role": "user", "content": query})

        # 获取可用工具
        tools = _get_openai_tools()

        # Agent 循环
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n🔄 Agent 迭代 {iteration + 1}/{self.max_iterations}")

            # 调用 LLM
            response = self._call_llm(messages, tools)
            assistant_message = response.choices[0].message

            # 添加助手消息到列表
            messages.append(assistant_message.model_dump())

            # 检查是否有工具调用
            # 方式1: 检查 OpenAI 原生 tool_calls
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    func = tool_call.function
                    tool_name = func.name
                    try:
                        tool_args = json.loads(func.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    if verbose:
                        print(f"  🔧 调用工具: {tool_name}({tool_args})")

                    # 执行工具
                    result = _execute_tool(tool_name, tool_args)

                    if verbose:
                        preview = result[:100] + "..." if len(result) > 100 else result
                        print(f"     → {preview}")

                    # 添加工具结果到消息列表
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                continue  # 继续下一轮迭代

            # 方式2: 检查正则匹配的工具调用（兼容不支持 tool_calls 的模型）
            content = assistant_message.content or ""
            tool_call = self._parse_tool_call(content)

            if tool_call:
                tool_name, tool_args = tool_call

                if verbose:
                    print(f"  🔧 调用工具: {tool_name}({tool_args})")

                # 执行工具
                result = _execute_tool(tool_name, tool_args)

                if verbose:
                    preview = result[:100] + "..." if len(result) > 100 else result
                    print(f"     → {preview}")

                # 移除包含工具调用的消息，替换为工具结果
                messages.pop()  # 移除助手消息
                messages.append({
                    "role": "user",
                    "content": f"工具 {tool_name} 返回结果:\n{result}\n\n请根据这个结果回答用户的问题。"
                })

                continue  # 继续下一轮迭代

            # 没有工具调用，返回最终回答
            return content

        # 达到最大迭代次数
        return "抱歉，处理过程中遇到了问题，请稍后重试或联系对应运营商客服。"


# ═══════════════════════════════════════════════════
# 查询分类：简单查询走 fast-path，复杂查询走 Agent
# ═══════════════════════════════════════════════════

def _classify_query(query: str) -> str:
    """判断查询复杂度，返回 'simple'、'comparison' 或 'complex'。

    simple：单跳事实查询，直接 RAG 即可
    comparison：对比型查询，走对比快速路径（结构化表格 + 单次 LLM）
    complex：需要多步推理/计算，走 Agent
    """
    # 对比型：有对比关键词或隐含对比模式 → comparison 快速路径
    comparison_keywords = ["对比", "比较", "区别", "差异", "分别", "各是"]
    is_comparison = any(kw in query for kw in comparison_keywords)
    # 隐含对比："A和B的补贴"、"实付X和实付Y"
    if not is_comparison and re.search(r'(\d+).+?和.+?(\d+)', query):
        is_comparison = True

    # 提取套餐档位
    numbers = re.findall(r'(\d+)\s*(?:元|套餐)', query)
    unique = set()
    for n in numbers:
        if n in Config.PLAN_TIERS:
            unique.add(n)
        else:
            try:
                if Config.PLAN_PRICE_MIN <= int(n) <= Config.PLAN_PRICE_MAX:
                    unique.add(n)
            except ValueError:
                continue

    # 对比 + 多个已知档位 → comparison 快速路径
    if is_comparison and len(unique) >= 2:
        return "comparison"

    # 对比关键词但档位不足 → 仍走 comparison（用 RAG 补充）
    if is_comparison:
        return "comparison"

    # 计算型 → Agent
    if re.search(r'[\+\-\*\/]', query) or any(kw in query for kw in ["年费", "年消费", "差价", "算一下", "计算"]):
        return "complex"

    # 多个套餐档位但非对比 → Agent
    if len(unique) >= 2:
        return "complex"

    # 推荐/选择型 → Agent
    recommend_keywords = ["推荐", "合适", "怎么选", "选哪个", "建议", "适合", "划算"]
    if any(kw in query for kw in recommend_keywords):
        return "complex"

    # 默认走 fast-path
    return "simple"


def _fast_path_rag(query: str) -> str:
    """快速路径：直接调用 SimpleRAG，跳过 Agent 多轮推理。(P0: 熔断器保护)"""
    rag_breaker = get_rag_breaker()
    if not rag_breaker.allow_request():
        return "知识库暂时不可用（熔断中），请稍后重试。"

    rag = _get_simple_rag()
    try:
        result = rag.query(query)
        if result.success:
            rag_breaker.record_success()
        else:
            rag_breaker.record_failure()
        return result.answer if result.success else (result.answer or "查询失败，请稍后重试。")
    except Exception as e:
        rag_breaker.record_failure()
        logger.error("fast-path 异常: %s", e, exc_info=True)
        return "查询出错，请稍后重试。"


# 对比查询专用 prompt
_COMPARISON_PROMPT = """你是运营商套餐助手。根据以下上下文信息，对比用户询问的套餐。

【要求】
1. 用表格呈现对比结果，包含：流量、通话、宽带、副卡、实付月租
2. 只用上下文中的数据，不要编造数字
3. 【极其重要】上下文中可能包含多个套餐的数据。你必须只提取与用户询问的套餐档位完全匹配的数据。例如用户问"59和99对比"，你只能用标题中明确写着"59套餐"或"99套餐"的数据，绝不能把"199套餐"的数据当成"99套餐"使用
4. 区分"基础流量"和"最大流量（含赠送）"
5. 区分"主卡通话"和"最高通话（含副卡）"
6. 默认只展示"全额预存"方案的数据，除非用户明确问"橙分期"再展示橙分期方案
7. 如果某个套餐确实找不到数据，标注"暂无数据"，不要用其他套餐的数据冒充
8. 回答简洁，表格后附一句话建议

【上下文】
{context}

【用户问题】
{question}

【对比结果】"""


def _resolve_shifu_to_tiers(query: str) -> list[str]:
    """将"实付XX元"映射为实际套餐档位。例如"实付89元" → "129"。"""
    # 已知的实付→原价映射
    shifu_map = {
        "39": "59",   # 实付39元 = 59元套餐全额预存
        "59": "79",   # 实付59元 = 79元套餐全额预存
        "69": "99",   # 实付69元 = 99元套餐全额预存
        "89": "129",  # 实付89元 = 129元套餐全额预存
        "129": "169", # 实付129元 = 169元套餐全额预存
        "139": "199", # 实付139元 = 199元套餐全额预存
        "199": "299", # 实付199元 = 299元套餐全额预存
    }
    resolved = []
    for price, tier in shifu_map.items():
        if re.search(rf'实付\s*{price}\s*元', query):
            resolved.append(tier)
    return resolved


def _fast_path_comparison(query: str) -> str:
    """对比快速路径：使用 SimpleRAG 检索（无档位过滤），单次 LLM 生成对比表。(P0: 熔断器保护)"""
    rag_breaker = get_rag_breaker()
    if not rag_breaker.allow_request():
        return "知识库暂时不可用（熔断中），请稍后重试。"

    rag = _get_simple_rag()

    try:
        # 提取套餐档位：先尝试"实付XX元"映射，再提取原始档位
        shifu_tiers = _resolve_shifu_to_tiers(query)
        direct_tiers = extract_plan_tiers(query)
        tiers = shifu_tiers if shifu_tiers else direct_tiers

        if len(tiers) >= 2:
            # 对比查询：用档位名作为查询词分别检索，确保每个档位都能召回
            seen = set()
            docs = []
            for tier in tiers:
                tier_query = f"{tier}元套餐 流量 通话 宽带 副卡 月租"
                tier_docs = rag.hybrid_retriever.search(tier_query, Config.RETRIEVAL_K)
                for d in tier_docs:
                    key = d.page_content.strip()
                    if key not in seen:
                        seen.add(key)
                        docs.append(d)
        else:
            # 无明确档位，用原始查询检索
            docs = rag.hybrid_retriever.search(query, Config.RETRIEVAL_K)

        if not docs:
            return "未找到相关套餐对比信息，请稍后重试。"

        # 构建上下文，限制文档数量减少 token（历史已通过查询重写处理，不重复传入）
        context = "\n\n".join(doc.page_content for doc in docs[:8])
        prompt = _COMPARISON_PROMPT.format(context=context, question=query)

        # 单次 LLM 调用
        @_retry_on_rate_limit
        def _do_call():
            client = _get_llm_client()
            response = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.AGENT_TEMPERATURE,
                max_tokens=Config.AGENT_MAX_TOKENS,
            )
            return response.choices[0].message.content

        answer = _do_call()
        rag_breaker.record_success()
        return answer

    except Exception as e:
        rag_breaker.record_failure()
        logger.error("对比快速路径异常: %s", e, exc_info=True)
        return "对比查询出错，请稍后重试。"


# ═══════════════════════════════════════════════════
# 查询入口
# ═══════════════════════════════════════════════════


class QueryResult(TypedDict):
    """查询结果结构"""
    answer: str
    success: bool
    processing_time: float


def run_single(query: str, verbose: bool = True) -> QueryResult:
    """单次查询，返回结构化结果。

    Args:
        query: 用户问题
        verbose: 是否输出详细日志到控制台（交互模式为 True，服务模式为 False）

    P0: 每次查询生成 trace_id，贯穿整个调用链
    """
    # P0: trace_id
    trace_id = set_trace_id()
    original_query = query  # 保存原始查询用于历史记录

    if verbose:
        print(f"\n{'═' * 55}")
        print(f"💬 用户: {query}")
        print(f"{'═' * 55}\n")
    else:
        logger.info("用户查询: %s", query)

    # 查询重写：指代消解、省略补全、同义扩展
    rewritten_query = _rewrite_query(query)
    if rewritten_query != query:
        if verbose:
            print(f"🔄 重写: {rewritten_query}\n")
        else:
            logger.info("查询重写: %s → %s", query, rewritten_query)
    query = rewritten_query

    # 检查缓存（使用重写后的查询作为 key）
    cache = get_query_cache()
    cache_key = make_query_cache_key(query)
    cached = cache.get(cache_key)
    if cached:
        if verbose:
            print("⚡ 缓存命中\n")
        else:
            logger.info("缓存命中: %s", query[:30])
        # 更新会话历史（存原始查询）
        _record_exchange(original_query, cached["answer"])
        return QueryResult(**cached)

    start = time.time()
    answer = ""
    route = "agent"

    # 路由决策：simple → fast-path, comparison → 对比快速路径, complex → Agent
    query_type = _classify_query(query)

    try:
        if query_type == "simple":
            route = "fast-path"
            if verbose:
                print("⚡ fast-path: 直接 RAG 查询（跳过 Agent）\n")
            else:
                logger.info("fast-path: 直接 RAG 查询")
            answer = _fast_path_rag(query)

        elif query_type == "comparison":
            route = "comparison"
            if verbose:
                print("📊 comparison: 对比快速路径（跳过 Agent）\n")
            else:
                logger.info("comparison: 对比快速路径")
            answer = _fast_path_comparison(query)

        else:
            route = "agent"
            if verbose:
                print("🤖 agent: 多步推理查询\n")
            else:
                logger.info("agent: 多步推理查询")

            agent = PurePythonAgent()
            answer = agent.run(query, verbose=verbose)

    except Exception as e:
        logger.warning("%s 调用失败，降级到直接 RAG: %s", route, e)
        try:
            result = _fast_path_rag(query)
            answer = result if result else "查询失败，请稍后重试。"
        except Exception as e2:
            logger.error("RAG 降级也失败: %s", e2)
            answer = "查询失败，请稍后重试。如持续失败请联系对应运营商客服。"

    # 后处理：去重
    answer = _deduplicate_answer(answer)

    elapsed = time.time() - start

    # 只记录成功的对话，失败的不污染历史
    success = not answer.startswith(("查询失败", "查询出错"))
    if answer and success:
        _record_exchange(original_query, answer)
        # 缓存成功结果
        cache.set(cache_key, {
            "answer": answer,
            "success": success,
            "processing_time": elapsed
        })

    if verbose:
        print(f"\n{'─' * 55}")
        print(f"💬 回答:\n{answer}")
        print(f"{'─' * 55}")
        print(f"⏱  耗时: {elapsed:.2f}s  [路由: {route}]")
    else:
        logger.info("查询完成，耗时 %.2fs，success=%s，路由=%s", elapsed, success, route)

    return QueryResult(answer=answer, success=success, processing_time=elapsed)


def run_interactive():
    """交互模式"""
    _session_ctx.set(f"cli_{uuid.uuid4().hex[:12]}")

    print("╔═══════════════════════════════════════════════════════╗")
    print("║      运营商套餐 Agent (纯Python版) — 交互模式         ║")
    print("║                                                       ║")
    print("║  我可以帮你：                                          ║")
    print("║    🔍 查询套餐详情（流量/通话/宽带/资费）               ║")
    print("║    🧮 计算费用（年费/差价/折扣）                        ║")
    print("║    💡 套餐推荐和对比                                    ║")
    print("║    📊 查看知识库统计                                    ║")
    print("║    🗑️  清除缓存 (输入 clear cache)                     ║")
    print("║                                                       ║")
    print("║  输入 quit 或 exit 退出                                ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    # 预热：初始化 workflow
    try:
        _get_workflow()
    except Exception as e:
        logger.error("初始化失败: %s", e)
        print(f"⚠️  警告: 初始化失败 ({e})，部分功能可能不可用\n")

    while True:
        try:
            query = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q", "退出"):
            print("👋 再见！")
            break
        # 缓存管理命令
        if query.lower() in ("clear cache", "清除缓存", "清缓存"):
            clear_all_caches()
            print("✅ 缓存已清除\n")
            continue
        if query.lower() in ("cache stats", "缓存统计"):
            stats = get_all_cache_stats()
            print("\n📊 缓存统计:")
            for name, s in stats.items():
                print(f"  {name}: 命中率={s['hit_rate']}, 大小={s['size']}/{s['max_size']}")
            print()
            continue
        run_single(query)
        print()


# ═══════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        Config.validate()
        _register_tools()
    except Exception as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    if len(sys.argv) > 1:
        run_single(" ".join(sys.argv[1:]), verbose=True)
    else:
        run_interactive()
