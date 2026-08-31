#!/usr/bin/env python3
"""
运营商套餐 Agent — 基于 LangChain AgentExecutor
将 RAGWorkflow 的工程化检索能力 + 计算器封装为工具，由 Agent 自主规划调用

架构：
  Agent (LLM Tool Calling 自主规划)
    ├── Tool: 套餐知识查询 → RAGWorkflow.query()
    ├── Tool: 费用计算器   → simpleeval 安全计算
    └── Tool: 套餐统计     → RAGWorkflow.get_stats()

用法:
  python taocan_agent.py              # 进入交互模式
  python taocan_agent.py "59套餐多少流量"  # 单次查询
"""

import contextvars
import csv
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
import warnings
from functools import wraps
from typing import TypedDict

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from simpleeval import simple_eval

from config import Config
from conversation_history import ConversationManager
from workflow_langchain import RAGWorkflow
from cache_manager import get_query_cache, make_query_cache_key, clear_all_caches, get_all_cache_stats

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")

# ──────────── 日志 ────────────
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format=Config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 结构化套餐数据加载（启动时加载一次，供 套餐对比直达 工具使用）
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


# 已知套餐档位
_PLAN_NUMBERS = ['29', '39', '59', '79', '99', '129', '169', '199', '229', '299']

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


def _extract_tiers(query: str) -> list[str]:
    """从用户输入中提取套餐档位数字列表。"""
    tiers = []
    for num in re.findall(r'(\d+)', query):
        if num in _PLAN_NUMBERS and num not in tiers:
            tiers.append(num)
    return tiers


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
    """装饰器：对 429 限流错误进行指数退避重试"""
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
                    wait = min(delay * (backoff ** attempt), max_delay)
                    logger.warning(
                        "遇到 429 限流，第 %d/%d 次重试，等待 %.1fs...",
                        attempt + 1, max_retries, wait
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
    return wrapper


# ═══════════════════════════════════════════════════
# 全局实例（延迟初始化，double-checked locking 保证线程安全）
# ═══════════════════════════════════════════════════

_workflow: RAGWorkflow | None = None
_conversation_mgr: ConversationManager | None = None
_agent_executor: AgentExecutor | None = None
_workflow_lock = threading.Lock()
_conversation_mgr_lock = threading.Lock()
_agent_executor_lock = threading.Lock()

# 使用 contextvars 替代全局变量，保证并发安全
_session_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="cli_default")


def _get_workflow() -> RAGWorkflow:
    """延迟初始化 RAGWorkflow 单例（线程安全）"""
    global _workflow
    if _workflow is None:
        with _workflow_lock:
            if _workflow is None:
                logger.info("初始化 RAGWorkflow ...")
                _workflow = RAGWorkflow()
    return _workflow


def _get_conversation_mgr() -> ConversationManager:
    """延迟初始化 ConversationManager 单例（线程安全）"""
    global _conversation_mgr
    if _conversation_mgr is None:
        with _conversation_mgr_lock:
            if _conversation_mgr is None:
                _conversation_mgr = ConversationManager(
                    max_history=Config.CONVERSATION_MAX_HISTORY,
                    ttl_seconds=Config.CONVERSATION_TTL,
                    max_sessions=Config.CONVERSATION_MAX_SESSIONS,
                )
    return _conversation_mgr


def _get_agent_executor() -> AgentExecutor:
    """延迟初始化 AgentExecutor 单例（线程安全）"""
    global _agent_executor
    if _agent_executor is None:
        with _agent_executor_lock:
            if _agent_executor is None:
                logger.info("构建 AgentExecutor ...")
                _agent_executor = _build_agent()
    return _agent_executor


def _reset_agent_executor():
    """重置 AgentExecutor（用于热重载）"""
    global _agent_executor
    _agent_executor = None


def _get_history_text() -> str:
    """获取当前会话的历史记录文本"""
    mgr = _get_conversation_mgr()
    return mgr.format_history_for_prompt(_session_ctx.get())


def _record_exchange(question: str, answer: str):
    """记录一轮对话到历史"""
    mgr = _get_conversation_mgr()
    mgr.add_exchange(_session_ctx.get(), question, answer)


def _get_history_messages() -> list:
    """获取当前会话的历史记录，转换为 LangChain 消息格式"""
    mgr = _get_conversation_mgr()
    messages = mgr.get_history(_session_ctx.get())
    result = []
    for msg in messages:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result


# ═══════════════════════════════════════════════════
# 数据隔离：防止跨套餐数据混淆
# ═══════════════════════════════════════════════════

# 套餐价格合理范围（用于宽松匹配）
_PLAN_PRICE_MIN = 19
_PLAN_PRICE_MAX = 599


def _extract_plan_number(query: str) -> str | None:
    """从问题中提取套餐档位数字，如 59、99、129 等。

    优先匹配已知档位；若不在已知列表中，只要在合理价格范围内也接受。
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
            # 已知档位直接返回
            if num in _PLAN_NUMBERS:
                return num
            # 合理价格范围内也接受（应对新增档位）
            try:
                price = int(num)
                if _PLAN_PRICE_MIN <= price <= _PLAN_PRICE_MAX:
                    return num
            except ValueError:
                continue
    return None


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
# 带重试的查询辅助函数
# ═══════════════════════════════════════════════════


@_retry_on_rate_limit
def _query_with_retry(workflow: RAGWorkflow, query: str, history: str) -> dict:
    """带限流重试的 RAG 查询"""
    return workflow.query(query, conversation_history=history)


# ═══════════════════════════════════════════════════
# Tool 1：套餐知识查询（包装 RAGWorkflow）
# ═══════════════════════════════════════════════════


@tool
def 套餐知识查询(query: str) -> str:
    """查询运营商套餐知识库，检索套餐的流量、通话、宽带、资费等详细信息。
    输入应为关于套餐的具体问题，如"59套餐包含多少流量"、"129有宽带吗"。
    也适用于查询办理流程、转网规则、投诉渠道、副卡政策、套外收费、橙分期补贴等。
    也适用于查询投诉降档、APP操作、携号转网等跨运营商通用业务流程。
    注意：对比两个或多个套餐请优先使用「套餐对比直达」工具。

    使用技巧：
    - 推荐型问题请查询"大流量套餐"或"高性价比套餐"获取多个档位信息
    - 查询副卡政策时请同时查询"副卡数量 副卡流量 副卡费用 副卡加赠流量"
    - 查询套外收费时请查询"套外流量 套外通话 收费标准"
    - 查询橙分期时请查询"橙分期 直降金额 月权益金 补贴"
    - 查询全部套餐资费时请查询"全部套餐档位 资费 流量 通话 29 39 59 79 99 129 169 199 229 299"
    - 查询投诉降档、套餐变更、APP操作时请查询"投诉 套餐降档 业务变更 APP 我的投诉" """
    workflow = _get_workflow()
    history = _get_history_text()

    try:
        result = _query_with_retry(workflow, query, history)
        if not result.get("success"):
            return result.get("answer", "查询失败，请稍后重试。")

        answer = result["answer"]

        # 去重：防止 LLM 输出重复内容
        answer = _deduplicate_answer(answer)

        return answer

    except Exception as e:
        logger.error("套餐知识查询异常: %s", e, exc_info=True)
        return "查询出错，请稍后重试。如持续失败请联系客服 10000。"


# ═══════════════════════════════════════════════════
# Tool 1.5：套餐对比直达（跳过RAG，直接读结构化数据）
# ═══════════════════════════════════════════════════


@tool
def 套餐对比直达(plan_numbers: str) -> str:
    """对比两个或多个套餐的结构化数据（流量、通话、宽带、副卡、资费等）。
    直接从结构化数据读取，无需检索，速度极快。

    输入格式：套餐档位数字，用逗号分隔。例如 "59,129" 或 "99,199,299"。
    适用场景：用户明确要对比几个套餐时，优先使用此工具而非「套餐知识查询」。

    示例：
    - "对比59和129套餐" → 输入 "59,129"
    - "99和199哪个好" → 输入 "99,199"
    - "对比所有套餐" → 输入 "59,99,129,199,299"
    """
    # 解析输入
    numbers = [n.strip() for n in re.split(r'[,，、\s]+', plan_numbers) if n.strip()]
    if len(numbers) < 2:
        return "请至少提供两个套餐档位进行对比，例如 '59,129'。"

    result = _format_plan_comparison(numbers)
    if not result:
        return f"未找到套餐 {', '.join(numbers)} 的结构化数据，请用「套餐知识查询」工具检索。"

    return result


# ═══════════════════════════════════════════════════
# 查询分类：简单查询走 fast-path，复杂查询走 Agent
# ═══════════════════════════════════════════════════

def _classify_query(query: str) -> str:
    """判断查询复杂度，返回 'simple' 或 'complex'。

    simple：单跳事实查询，直接 RAG 即可
    complex：需要多步推理/对比/计算，走 Agent
    """
    # 复杂条件：有一个就走 Agent
    complex_keywords = ["对比", "比较", "区别", "差异", "推荐", "合适", "怎么选", "选哪个", "建议", "适合", "划算"]
    if any(kw in query for kw in complex_keywords):
        return "complex"

    # 计算型
    if re.search(r'[\+\-\*\/]', query) or any(kw in query for kw in ["年费", "年消费", "差价", "算一下", "计算"]):
        return "complex"

    # 涉及多个套餐档位
    numbers = re.findall(r'(\d+)\s*(?:元|套餐)', query)
    unique = set()
    for n in numbers:
        if n in _PLAN_NUMBERS:
            unique.add(n)
        else:
            try:
                if _PLAN_PRICE_MIN <= int(n) <= _PLAN_PRICE_MAX:
                    unique.add(n)
            except ValueError:
                continue
    if len(unique) >= 2:
        return "complex"

    # 默认走 fast-path
    return "simple"


def _fast_path_rag(query: str) -> str:
    """快速路径：直接调用 RAG，跳过 Agent 多轮推理。"""
    workflow = _get_workflow()
    history = _get_history_text()

    try:
        result = _query_with_retry(workflow, query, history)
        if not result.get("success"):
            return result.get("answer", "查询失败，请稍后重试。")

        answer = result["answer"]
        answer = _deduplicate_answer(answer)

        return answer

    except Exception as e:
        logger.error("fast-path RAG 异常: %s", e, exc_info=True)
        return "查询出错，请稍后重试。如持续失败请联系客服 10000。"


# ═══════════════════════════════════════════════════
# Tool 2：费用计算器（保留 simpleeval 安全沙箱）
# ═══════════════════════════════════════════════════


@tool
def 费用计算器(expression: str) -> str:
    """计算数学表达式，用于计算套餐费用、年消费、折扣差价等。
    支持 + - * / ** 和 sqrt/abs/round 等函数。
    示例: "39*12"、"129-40"、"39*12+9.9*12" """
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "int": int,
        "float": float,
        "sum": sum,
    }
    SAFE_NAMES = {"pi": math.pi, "e": math.e}
    SAFE_NAMES_BY_KEY = set(SAFE_FUNCTIONS.keys()) | set(SAFE_NAMES.keys())

    expr = expression.strip()
    expr = expr.replace("×", "*").replace("÷", "/")

    if "__" in expr:
        return "计算出错: 表达式包含非法模式。"

    for token in re.findall(r"[a-zA-Z_]\w*", expr):
        if token not in SAFE_NAMES_BY_KEY:
            safe_list = "、".join(sorted(SAFE_NAMES_BY_KEY))
            return f"计算出错: 未知的函数或变量「{token}」。仅支持: {safe_list}"

    if re.search(r"[a-zA-Z_]\w*\.", expr):
        return "计算出错: 不支持属性访问操作。"

    try:
        result = simple_eval(expr, functions=SAFE_FUNCTIONS, names=SAFE_NAMES)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算出错: {e}。请检查表达式格式，例如 '39 * 12' 或 '129 - 40'。"


# ═══════════════════════════════════════════════════
# Tool 3：套餐统计
# ═══════════════════════════════════════════════════


@tool
def 套餐统计() -> str:
    """查询当前知识库的统计信息，包括文档数量、模型配置、缓存状态等。"""
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


# ═══════════════════════════════════════════════════
# 构建 Agent（使用 Tool Calling）
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个运营商套餐智能助手。你的职责是帮助用户查询套餐信息和计算费用。

重要规则：
1. 套餐相关问题（流量、通话、宽带、资费、副卡、转网、投诉等），使用「套餐知识查询」工具
2. 费用计算（年费、差价、折扣），先用「套餐知识查询」获取数据，再用「费用计算器」计算
3. 【优先】对比两个或多个套餐时，使用「套餐对比直达」工具，输入档位数字如 "59,129"。速度快、数据准确。只有当「套餐对比直达」返回数据不足时，才用「套餐知识查询」补充
4. 无关问题，直接回答"抱歉，我只能回答运营商套餐相关的问题"
5. 回答要简洁友好，基于真实数据，不要编造
6. 如果没有相关信息，坦诚告知并建议联系客服 10000

【数据来源约束 — 极其重要】
- 所有数字（补贴金额、月租、流量、通话等）必须来自工具返回的结果
- 禁止自行推算、估算或编造任何数字
- 如果工具未返回某项具体数据（如"月权益金"、"每月返还金额"），必须回答"该信息需咨询客服 10000"
- 禁止发明知识库中不存在的概念（如"月权益金"、"月返"等）

【互斥方案隔离 — 极其重要】
- "全额预存"和"橙分期"是两种互斥方案，用户只能选择其中一种
- 绝不能将两种方案的价格或优惠混在一起计算
- 橙分期的补贴是一次性购机补贴，不是每月返还
- 全额预存的"实付XX元"是话费赠送后的月租，与橙分期无关
- 对比套餐时，必须明确区分是哪种方案下的数据"""


def _build_agent() -> AgentExecutor:
    """构建 LangChain Agent"""
    api_key = Config.LLM_API_KEY or Config.SILICONFLOW_API_KEY
    api_base = Config.LLM_API_BASE or Config.SILICONFLOW_API_BASE

    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=Config.AGENT_TEMPERATURE,
        max_tokens=Config.AGENT_MAX_TOKENS,
        max_retries=0,  # 由外层统一控制重试，避免双重重试
        request_timeout=60,
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [套餐知识查询, 套餐对比直达, 费用计算器, 套餐统计]
    agent = create_tool_calling_agent(llm, tools, prompt_template)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=Config.AGENT_VERBOSE,
        max_iterations=Config.AGENT_MAX_ITERATIONS,
        return_intermediate_steps=True,
    )


# ═══════════════════════════════════════════════════
# 查询入口
# ═══════════════════════════════════════════════════


class QueryResult(TypedDict):
    """查询结果结构"""
    answer: str
    success: bool
    processing_time: float


@_retry_on_rate_limit
def _invoke_agent(query: str, chat_history: list) -> dict:
    """调用 Agent（带限流重试）"""
    executor = _get_agent_executor()
    return executor.invoke({"input": query, "chat_history": chat_history})


@_retry_on_rate_limit
def _fallback_rag_query(query: str) -> dict:
    """降级查询：直接调用 RAGWorkflow（跳过 Agent，带限流重试）"""
    workflow = _get_workflow()
    history = _get_history_text()
    return workflow.query(query, conversation_history=history)


def run_single(query: str, verbose: bool = True) -> QueryResult:
    """单次查询，返回结构化结果。

    Args:
        query: 用户问题
        verbose: 是否输出详细日志到控制台（交互模式为 True，服务模式为 False）
    """
    if verbose:
        print(f"\n{'═' * 55}")
        print(f"💬 用户: {query}")
        print(f"{'═' * 55}\n")
    else:
        logger.info("用户查询: %s", query)

    # 检查缓存
    cache = get_query_cache()
    cache_key = make_query_cache_key(query)
    cached = cache.get(cache_key)
    if cached:
        if verbose:
            print("⚡ 缓存命中\n")
        else:
            logger.info("缓存命中: %s", query[:30])
        # 更新会话历史
        _record_exchange(query, cached["answer"])
        return QueryResult(**cached)

    start = time.time()
    answer = ""
    route = "agent"

    # 路由决策：简单查询走 fast-path，复杂查询走 Agent
    query_type = _classify_query(query)

    try:
        if query_type == "simple":
            route = "fast-path"
            if verbose:
                print("⚡ fast-path: 直接 RAG 查询（跳过 Agent）\n")
            else:
                logger.info("fast-path: 直接 RAG 查询")
            answer = _fast_path_rag(query)

        else:
            route = "agent"
            if verbose:
                print("🤖 agent: 多步推理查询\n")
            else:
                logger.info("agent: 多步推理查询")

            chat_history = _get_history_messages()
            result = _invoke_agent(query, chat_history)
            answer = result["output"]

            # 输出中间步骤
            steps = result.get("intermediate_steps", [])
            if steps:
                if verbose:
                    print(f"\n🔧 工具调用 ({len(steps)} 步):")
                for i, (action, observation) in enumerate(steps, 1):
                    obs_preview = observation[:100] + "..." if isinstance(observation, str) and len(observation) > 100 else observation
                    if verbose:
                        print(f"  {i}. {action.tool}({action.tool_input})")
                        print(f"     → {obs_preview}")
                    else:
                        logger.debug("工具调用 %d: %s(%s) → %s", i, action.tool, action.tool_input, obs_preview)

    except Exception as e:
        logger.warning("%s 调用失败，降级到直接 RAG: %s", route, e)
        try:
            result = _fallback_rag_query(query)
            answer = result.get("answer", "")
            if not answer:
                answer = "查询失败，请稍后重试。"
        except Exception as e2:
            logger.error("RAG 降级也失败: %s", e2)
            answer = "查询失败，请稍后重试。如持续失败请联系客服 10000。"

    # 后处理：去重
    answer = _deduplicate_answer(answer)

    elapsed = time.time() - start

    # 只记录成功的对话，失败的不污染历史
    success = not answer.startswith(("查询失败", "查询出错"))
    if answer and success:
        _record_exchange(query, answer)
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
    print("║      运营商套餐 Agent — 交互模式                       ║")
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
        logger.error("RAGWorkflow 初始化失败: %s", e)
        print(f"⚠️  警告: 知识库初始化失败 ({e})，部分功能可能不可用\n")

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
    except Exception as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    if len(sys.argv) > 1:
        run_single(" ".join(sys.argv[1:]), verbose=True)
    else:
        run_interactive()
