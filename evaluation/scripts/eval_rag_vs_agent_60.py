#!/usr/bin/env python3
"""
配对评测：纯RAG（SimpleRAG） vs Agent（dx_agent），60题同题双管道。

特性：
- 检索hook：包裹两条管道的 HybridRetriever.search/bare_search，记录每次检索的
  完整chunk与耗时（不改动业务代码）
- 限流处理：429/rate limit 异常指数退避重试（8s起，最多6次）
- 断点续跑：每题完成即写 evaluation/results/eval60_progress.json，重启自动跳过已完成项
- 五维百分比评分（客观公式，无人工打分）：
    召回率   = 命中expected_keywords的检索上下文关键词数 / 关键词总数 × 100   （检索层）
    完整性   = 命中expected_keywords的最终答案关键词数 / 关键词总数 × 100     （答案层）
    幻觉控制 = 100 − 50 × 幻觉点；幻觉点 = anti_keyword命中 + 答案中无出处数字（每题上限2点）
    准确率   = 完整性 × (1 − 幻觉点/2)
    检索耗时 = t≤3s→100；t≥15s→0；中间线性：100×(15−t)/12
    综合     = 五维等权平均
"""

import os
import re
import sys
import json
import time
import argparse
import statistics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR)
for _p in ("core", "infra", "evaluation/tests"):
    _p = os.path.join(BASE_DIR, _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_questions_60 import TEST_CASES

RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")
PROGRESS_FILE = os.path.join(RESULTS_DIR, "eval60_progress.json")
REPORT_FILE = os.path.join(BASE_DIR, "evaluation", "reports", "eval_60_rag_vs_agent_report.md")

RATE_LIMIT_PATTERNS = re.compile(
    r"429|rate.?limit|too many requests|限流|请求过快|throttl", re.IGNORECASE
)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


# ───────────────────────── 限流重试 ─────────────────────────

def with_rate_limit_retry(fn, max_retries=6, base_delay=8.0, stats=None):
    """指数退避重试：识别限流信号后等待重试。"""
    for attempt in range(max_retries + 1):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if RATE_LIMIT_PATTERNS.search(msg) and attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), 120.0)
                if stats is not None:
                    stats["rate_limit_retries"] = stats.get("rate_limit_retries", 0) + 1
                print(f"    ⚠️ 疑似限流（{msg[:90]}），退避 {delay:.0f}s 后重试 "
                      f"({attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            raise
    return None, "max_retries_exceeded"


# ───────────────────────── 检索 hook ─────────────────────────

class RetrievalRecorder:
    """包裹 HybridRetriever 的 search/bare_search，按题记录检索证据与耗时。"""

    def __init__(self, hybrid_retriever):
        self._retriever = hybrid_retriever
        self.calls = []

    def start_question(self):
        self.calls = []

    def collect(self):
        chunks, elapsed = [], 0.0
        seen = set()
        for c in self.calls:
            elapsed += c["elapsed"]
            for d in c["docs"]:
                key = id(d)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append({
                    "content": d.page_content,
                    "source": d.metadata.get("source", ""),
                })
        return {"chunks": chunks, "retrieval_elapsed": round(elapsed, 3), "n_calls": len(self.calls)}

    def wrap(self):
        retriever = self._retriever
        # 先保存原绑定方法，避免包装函数调用到被覆盖后的自身（无限递归）
        orig_search = retriever.search
        orig_bare = retriever.bare_search

        def wrapped_search(query, top_k=None, *a, **kw):
            t = time.time()
            docs = orig_search(query, top_k, *a, **kw) if top_k is not None else orig_search(query, *a, **kw)
            self.calls.append({"query": query, "docs": list(docs or []), "elapsed": time.time() - t})
            return docs

        def wrapped_bare(query, top_k=None, *a, **kw):
            t = time.time()
            docs = orig_bare(query, top_k, *a, **kw) if top_k is not None else orig_bare(query, *a, **kw)
            self.calls.append({"query": query, "docs": list(docs or []), "elapsed": time.time() - t})
            return docs

        retriever.search = wrapped_search
        retriever.bare_search = wrapped_bare


# ───────────────────────── 评分 ─────────────────────────

def _anti_fires(answer: str, anti_kw: str, tiers) -> bool:
    """张冠李戴检测：anti_keyword 仅在与'题目所问档位'同一句中出现时才算幻觉点。

    例：问'99元套餐的1000M宽带多少钱'，答案若写'99元套餐可加装1000M宽带，月费49.9元'
    （同句含99+49.9）→ 计幻觉；若仅推荐199/299时提到49.9 → 不计。
    """
    tier_pats = [rf"(?<!\d){re.escape(t)}(?!\d)" for t in tiers]
    for sentence in re.split(r"[。！？；\n]", answer):
        if anti_kw in sentence and any(re.search(p, sentence) for p in tier_pats):
            return True
    return False


def score_case(tc, answer, retrieval):
    keywords = tc.get("expected_keywords", [])
    anti = tc.get("anti_keywords", [])
    context_text = "\n".join(c["content"] for c in retrieval["chunks"])

    total = max(1, len(keywords))
    hit_ctx = sum(1 for k in keywords if k in context_text)
    hit_ans = sum(1 for k in keywords if k in answer)
    recall = round(hit_ctx / total * 100, 1)
    completeness = round(hit_ans / total * 100, 1)

    # 题目所问档位（用于限定 anti 的生效范围）
    tiers = [m for m in set(NUM_RE.findall(tc["question"]))
             if len(m) >= 2 and 19 <= int(float(m)) <= 599]
    anti_hits = sum(1 for k in anti if _anti_fires(answer, k, tiers))

    fabricated = 0
    if context_text:
        scope = context_text + " " + tc.get("reference", "") + " " + " ".join(keywords)
        for m in set(NUM_RE.findall(answer)):
            if len(m) >= 2 and m not in scope and not any(m in w for w in keywords):
                fabricated += 1
    points = min(2, anti_hits + fabricated)
    hallucination_score = max(0.0, 100 - 50 * points)
    accuracy = round(completeness * (1 - points / 2), 1)

    t = retrieval["retrieval_elapsed"]
    if t <= 3:
        latency = 100.0
    elif t >= 15:
        latency = 0.0
    else:
        latency = round(100 * (15 - t) / 12, 1)

    composite = round((recall + completeness + accuracy + hallucination_score + latency) / 5, 1)

    return {
        "recall": recall, "completeness": completeness, "accuracy": accuracy,
        "hallucination": round(hallucination_score, 1), "latency_score": latency,
        "composite": composite,
        "hit_context": f"{hit_ctx}/{total}", "hit_answer": f"{hit_ans}/{total}",
        "anti_hits": anti_hits, "fabricated_numbers": fabricated,
    }


# ───────────────────────── 断点续跑 ─────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"A": {}, "B": {}, "stats": {}}


def save_progress(progress):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PROGRESS_FILE)


# ───────────────────────── 主流程 ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", default="A,B")
    ap.add_argument("--limit", type=int, default=0, help="只跑前N题（冒烟用）")
    ap.add_argument("--qids", default="", help="逗号分隔，只跑指定题")
    ap.add_argument("--delay", type=float, default=1.5, help="题间间隔秒")
    args = ap.parse_args()

    pipelines = [p.strip().upper() for p in args.pipelines.split(",") if p.strip()]
    cases = TEST_CASES
    if args.qids:
        want = {int(x) for x in args.qids.split(",")}
        cases = [c for c in cases if c["id"] in want]
    elif args.limit:
        cases = cases[:args.limit]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    progress = load_progress()
    stats = progress.setdefault("stats", {})
    stats.setdefault("rate_limit_retries", 0)

    # ── 初始化两条管道（共享同一向量库/嵌入API） ──
    print("初始化 SimpleRAG（A管道）...")
    from simple_rag import SimpleRAG
    rag_a = SimpleRAG()
    rec_a = RetrievalRecorder(rag_a.hybrid_retriever)
    rec_a.wrap()

    rec_b = None
    if "B" in pipelines:
        print("初始化 dx_agent（B管道）...")
        import dx_agent
        rag_b = dx_agent._get_simple_rag()  # 预建单例，便于挂钩子
        rec_b = RetrievalRecorder(rag_b.hybrid_retriever)
        rec_b.wrap()

    def run_a(question):
        r = rag_a.query(question)
        return r.answer, bool(r.success), float(r.processing_time)

    def run_b(question):
        import dx_agent
        dx_agent._get_conversation_mgr().clear_history("cli_default")
        r = dx_agent.run_single(question, verbose=False)
        return r["answer"], bool(r["success"]), float(r["processing_time"])

    runners = {"A": (run_a, rec_a), "B": (run_b, rec_b)}

    done = 0
    for tc in cases:
        qid, q = tc["id"], tc["question"]
        for pl in pipelines:
            key = str(qid)
            if key in progress[pl]:
                continue
            runner, rec = runners[pl]
            rec.start_question()
            print(f"\n▶ [{pl}{qid}] {q}")
            t0 = time.time()
            try:
                (answer, success, e2e), _ = with_rate_limit_retry(
                    lambda: runner(q), stats=stats)
            except Exception as e:
                print(f"    ❌ 失败: {e}")
                progress[pl][key] = {
                    "question": q, "answer": f"<<EXEC_ERROR>> {e}", "success": False,
                    "e2e_seconds": round(time.time() - t0, 2), "retrieval": {"chunks": [], "retrieval_elapsed": 0, "n_calls": 0},
                }
                save_progress(progress)
                continue

            retrieval = rec.collect()
            score = score_case(tc, answer, retrieval)
            progress[pl][key] = {
                "question": q, "answer": answer, "success": success,
                "e2e_seconds": round(e2e, 2), "retrieval": retrieval,
                "route": dx_agent._classify_query(q) if pl == "B" else None,
                "scores": score,
            }
            save_progress(progress)
            s = score
            print(f"    ✅ {e2e:.1f}s(端到端) 检索{retrieval['retrieval_elapsed']:.1f}s "
                  f"召回{s['recall']}% 完整{s['completeness']}% 准确{s['accuracy']}% "
                  f"幻觉{s['hallucination']}% 耗时分{s['latency_score']}% 综合{s['composite']}%")
            done += 1
            time.sleep(args.delay)

    print(f"\n本轮新完成 {done} 题次。进度：A={len(progress['A'])}/60 B={len(progress['B'])}/60")
    print(f"限流重试累计：{stats.get('rate_limit_retries', 0)}")


if __name__ == "__main__":
    main()
