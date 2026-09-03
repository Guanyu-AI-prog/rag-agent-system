#!/usr/bin/env python3
"""
从 eval60_progress.json 的已存答案+检索证据重算五维评分并生成对比报告（Markdown）。
评分公式与 eval_rag_vs_agent_60.score_case 一致，另含：
- 计算题豁免：reference 含 '=' 时跳过'无出处数字'检测（算术得数不是幻觉）
"""

import os
import re
import sys
import json
import statistics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(BASE_DIR, "evaluation", "tests") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "evaluation", "tests"))

from test_questions_60 import TEST_CASES

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
RESULTS = os.path.join(BASE_DIR, "evaluation", "results", "eval60_progress.json")
OUT_JSON = os.path.join(BASE_DIR, "evaluation", "results", "eval60_scored.json")
OUT_MD = os.path.join(BASE_DIR, "evaluation", "reports", "eval_60_rag_vs_agent_report.md")

TYPE_ORDER = ["单点查询", "对比型", "多跳推理", "流程型", "场景型", "边界/异常"]
DIMS = ["recall", "completeness", "hallucination", "accuracy", "latency_score"]
DIM_CN = {"recall": "召回率", "completeness": "完整性", "hallucination": "幻觉控制",
          "accuracy": "准确率", "latency_score": "检索耗时", "composite": "综合"}


def _anti_fires(answer, anti_kw, tiers):
    pats = [rf"(?<!\d){re.escape(t)}(?!\d)" for t in tiers]
    for sentence in re.split(r"[。！？；\n]", answer):
        if anti_kw in sentence and any(re.search(p, sentence) for p in pats):
            return True
    return False


def score_case(tc, answer, retrieval):
    keywords = tc.get("expected_keywords", [])
    anti = tc.get("anti_keywords", [])
    context_text = "\n".join(c["content"] for c in retrieval.get("chunks", []))

    total = max(1, len(keywords))
    hit_ctx = sum(1 for k in keywords if k in context_text)
    hit_ans = sum(1 for k in keywords if k in answer)
    recall = round(hit_ctx / total * 100, 1)
    completeness = round(hit_ans / total * 100, 1)

    tiers = [m for m in set(NUM_RE.findall(tc["question"]))
             if len(m) >= 2 and 19 <= int(float(m)) <= 599]
    anti_hits = sum(1 for k in anti if _anti_fires(answer, k, tiers))

    fabricated = 0
    is_calc = "=" in tc.get("reference", "")
    if context_text and not is_calc:
        scope = context_text + " " + tc.get("reference", "") + " " + " ".join(keywords)
        for m in set(NUM_RE.findall(answer)):
            if len(m) >= 2 and m not in scope and not any(m in w for w in keywords):
                fabricated += 1
    points = min(2, anti_hits + fabricated)
    hallucination_score = max(0.0, 100 - 50 * points)
    accuracy = round(completeness * (1 - points / 2), 1)

    t = retrieval.get("retrieval_elapsed", 0)
    latency = 100.0 if t <= 3 else (0.0 if t >= 15 else round(100 * (15 - t) / 12, 1))
    composite = round((recall + completeness + accuracy + hallucination_score + latency) / 5, 1)

    return {"recall": recall, "completeness": completeness, "accuracy": accuracy,
            "hallucination": round(hallucination_score, 1), "latency_score": latency,
            "composite": composite, "hit_context": f"{hit_ctx}/{total}",
            "hit_answer": f"{hit_ans}/{total}", "points": points}


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 1) if vals else None


def main():
    with open(RESULTS, "r", encoding="utf-8") as f:
        progress = json.load(f)
    tc_by_id = {tc["id"]: tc for tc in TEST_CASES}

    scored = {"A": {}, "B": {}, "stats": progress.get("stats", {})}
    for pl in ("A", "B"):
        for qid, rec in progress[pl].items():
            tc = tc_by_id[int(qid)]
            s = score_case(tc, rec["answer"], rec.get("retrieval", {"chunks": []}))
            rec = dict(rec)
            rec["scores"] = s
            rec["type"] = tc["type"]
            rec["group"] = tc.get("group", "")
            scored[pl][qid] = rec
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=1)

    # ── 汇总 ──
    def dims(pl, subset=None):
        items = [(k, v) for k, v in scored[pl].items()
                 if subset is None or int(k) in subset]
        row = {d: mean([v["scores"].get(d) for _, v in items]) for d in DIMS}
        row["composite"] = mean([v["scores"].get("composite") for _, v in items])
        row["n"] = len(items)
        return row

    a_all, b_all = dims("A"), dims("B")
    aA, bA = dims("A", {t["id"] for t in TEST_CASES if t["group"] == "A"}), dims("B", {t["id"] for t in TEST_CASES if t["group"] == "A"})
    aB, bB = dims("A", {t["id"] for t in TEST_CASES if t["group"] == "B"}), dims("B", {t["id"] for t in TEST_CASES if t["group"] == "B"})

    lines = []
    w = lines.append
    w("# 60题配对评测报告：纯RAG（SimpleRAG） vs Agent（dx_agent）\n")
    w("> 同一题分别调用两条管道；检索证据经 hook 捕获（不改业务代码）；评分全部为客观公式，无人工打分。\n")
    w(f"> 样本：A组（原30题）+ B组（新增30题，2026-09对照向量库定稿）；完成 {len(scored['A'])}+{len(scored['B'])} 题次；限流重试 {scored['stats'].get('rate_limit_retries', 0)} 次\n")

    w("## 评分标准（客观公式）\n")
    w("| 维度 | 公式 |")
    w("|---|---|")
    w("| 召回率 | expected_keywords 出现在**检索上下文**的比例（检索层：证据找齐了吗） |")
    w("| 完整性 | expected_keywords 出现在**最终答案**的比例（答案层：要点覆盖了几个） |")
    w("| 幻觉控制 | 100 − 50×幻觉点；幻觉点 = anti_keyword 与所问档位同句出现（张冠李戴）+ 答案中无出处数字（计算题豁免），每题上限2点 |")
    w("| 准确率 | 完整性 × (1 − 幻觉点/2) |")
    w("| 检索耗时 | t≤3s→100；t≥15s→0；中间线性 100×(15−t)/12 |")
    w("| 综合 | 五维等权平均 |\n")

    def table(row_a, row_b, title):
        w(f"### {title}\n")
        w("| 维度 | 纯RAG (A) | Agent (B) | 差值 (B−A) |")
        w("|---|---|---|---|")
        for d in DIMS + ["composite"]:
            va, vb = row_a.get(d), row_b.get(d)
            diff = round(vb - va, 1) if (va is not None and vb is not None) else "-"
            w(f"| {DIM_CN[d]} | {va}% | {vb}% | {diff} |")
        w("")

    table(a_all, b_all, "总体（60题）")
    table(aA, bA, "A组（原30题）")
    table(aB, bB, "B组（新增30题，难度更高）\n")

    w("### 分题型表现\n")
    w("| 题型 | 题数 | A召回 | B召回 | A完整 | B完整 | A准确 | B准确 | A幻觉 | B幻觉 | A综合 | B综合 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TYPE_ORDER:
        ids = {tc["id"] for tc in TEST_CASES if tc["type"] == t}
        ra, rb = dims("A", ids), dims("B", ids)
        w(f"| {t} | {ra['n']} | {ra['recall']}% | {rb['recall']}% | {ra['completeness']}% | {rb['completeness']}% "
          f"| {ra['accuracy']}% | {rb['accuracy']}% | {ra['hallucination']}% | {rb['hallucination']}% "
          f"| {ra['composite']}% | {rb['composite']}% |")
    w("")

    # 路由分布（B管道）
    w("### Agent 路由分布与得分（B管道）\n")
    routes = {}
    for qid, v in scored["B"].items():
        r = v.get("route") or "unknown"
        routes.setdefault(r, []).append(v["scores"]["composite"])
    w("| 路由 | 题数 | 平均综合 |")
    w("|---|---|---|")
    for r, vals in sorted(routes.items()):
        w(f"| {r} | {len(vals)} | {mean(vals)}% |")
    w("")

    # 配对差值
    w("### 配对比较（同题 B−A 综合分）\n")
    wins, losses, ties = 0, 0, 0
    diffs = []
    for qid in scored["A"]:
        if qid in scored["B"]:
            d = round(scored["B"][qid]["scores"]["composite"] - scored["A"][qid]["scores"]["composite"], 1)
            diffs.append((int(qid), d))
            wins += d > 0
            losses += d < 0
            ties += d == 0
    w(f"- Agent 占优 {wins} 题，纯RAG 占优 {losses} 题，持平 {ties} 题")
    w(f"- 平均差值 {mean([d for _, d in diffs])}pp，中位数 {statistics.median([d for _, d in diffs])}pp\n")

    # 端到端耗时
    w("### 耗时（端到端，秒）\n")
    w("| 管道 | 平均 | p50 | p95 | 最长 |")
    w("|---|---|---|---|---|")
    for pl, label in (("A", "纯RAG"), ("B", "Agent")):
        ts = sorted(v["e2e_seconds"] for v in scored[pl].values())
        if ts:
            w(f"| {label} | {round(statistics.mean(ts),1)} | {ts[len(ts)//2]} | {ts[int(len(ts)*0.95)]} | {ts[-1]} |")
    w("")

    # 低分题明细
    w("### 低分题 TOP10（按两管道综合均值）\n")
    both = [(int(k), (scored["A"][k]["scores"]["composite"] + scored["B"][k]["scores"]["composite"]) / 2)
            for k in scored["A"] if k in scored["B"]]
    low = sorted(both, key=lambda x: x[1])[:10]
    w("| 题号 | 题型 | 题目 | A综合 | B综合 |")
    w("|---|---|---|---|---|")
    for qid, avg in low:
        tc = tc_by_id[qid]
        w(f"| {qid} | {tc['type']} | {tc['question'][:38]} | {scored['A'][str(qid)]['scores']['composite']}% | {scored['B'][str(qid)]['scores']['composite']}% |")
    w("")

    w("### 已知评分局限（诚实声明）\n")
    w("1. **题17（CRM流程）**：A组历史关键词（菜单/客户信息/保存）与现库措辞（导购/标识/受理岗）脱节，两管道同罚，属题库遗留缺陷，不影响配对结论")
    w("2. **计算题豁免**：参考答案含'='的题不做无出处数字检测（算术得数不是幻觉），此类题的幻觉分仅由 anti_keyword 覆盖")
    w("3. **B管道嵌入缓存**：对比路径的分层子查询会命中进程内 embedding 缓存，检索耗时可低至毫秒级，属真实运行时特性")
    w("4. **关键词判定**：keyword 命中是'必要非充分'信号——答对关键词不代表语义完全正确，答漏关键词可能只是换了等价表述\n")

    w("### 完整答案与证据\n")
    w("- 全部60题×2管道的完整答案、检索chunk、逐题五维分：`evaluation/results/eval60_scored.json`")
    w("- 原始进度文件（含限流重试计数）：`evaluation/results/eval60_progress.json`\n")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("报告已生成:", OUT_MD)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
