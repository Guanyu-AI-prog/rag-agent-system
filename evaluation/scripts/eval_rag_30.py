#!/usr/bin/env python3
"""
RAG 评测脚本 — 30道题，5个维度打分
维度：召回、耗时、完整性、准确率、幻觉
"""

import sys, os, json, time, re
os.chdir('/root/langchain_rag_code')
sys.path.insert(0, '/root/langchain_rag_code')

# 路径引导：simple_rag 在 core/，其 config 依赖在 infra/
for _p in ('/root/langchain_rag_code/core', '/root/langchain_rag_code/infra'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simple_rag import SimpleRAG

# ═══════════════════════════════════════════════════
# 30道评测题（含标准答案）
# ═══════════════════════════════════════════════════

EVAL_QUESTIONS = [
    # ── 基础套餐查询（10题）──
    {
        "id": 1,
        "q": "59元套餐包含多少流量？",
        "category": "基础查询",
        "expected_keywords": ["10GB", "10G", "流量"],
        "expected_answer": "基础10GB通用流量",
        "anti_keywords": []  # 不应出现的关键词（幻觉检测）
    },
    {
        "id": 2,
        "q": "99元套餐月租多少？",
        "category": "基础查询",
        "expected_keywords": ["99元"],
        "expected_answer": "月租99元",
        "anti_keywords": ["69元"]  # 不应把优惠价当主套餐价
    },
    {
        "id": 3,
        "q": "129套餐包含多少通话分钟？",
        "category": "基础查询",
        "expected_keywords": ["500分钟", "500"],
        "expected_answer": "500分钟",
        "anti_keywords": []
    },
    {
        "id": 4,
        "q": "199套餐有多少流量？",
        "category": "基础查询",
        "expected_keywords": ["60GB", "60G"],
        "expected_answer": "基础60GB通用流量",
        "anti_keywords": []
    },
    {
        "id": 5,
        "q": "299套餐通话时长是多少？",
        "category": "基础查询",
        "expected_keywords": ["1500分钟", "1500"],
        "expected_answer": "1500分钟",
        "anti_keywords": []
    },
    {
        "id": 6,
        "q": "59元套餐可以办几张副卡？",
        "category": "基础查询",
        "expected_keywords": ["2张", "2"],
        "expected_answer": "可办2张副卡",
        "anti_keywords": ["4张"]
    },
    {
        "id": 7,
        "q": "129套餐可以办几张副卡？",
        "category": "基础查询",
        "expected_keywords": ["4张", "4"],
        "expected_answer": "可办4张副卡",
        "anti_keywords": ["2张"]
    },
    {
        "id": 8,
        "q": "99元套餐套外流量怎么收费？",
        "category": "基础查询",
        "expected_keywords": ["3元", "1GB", "1G"],
        "expected_answer": "3元/1GB",
        "anti_keywords": []
    },
    {
        "id": 9,
        "q": "199套餐副卡月功能费多少？",
        "category": "基础查询",
        "expected_keywords": ["10元", "10"],
        "expected_answer": "10元/月/张",
        "anti_keywords": []
    },
    {
        "id": 10,
        "q": "299套餐基础流量是多少？",
        "category": "基础查询",
        "expected_keywords": ["180GB", "180G"],
        "expected_answer": "180GB",
        "anti_keywords": []
    },

    # ── 对比查询（5题）──
    {
        "id": 11,
        "q": "对比99和129套餐",
        "category": "对比查询",
        "expected_keywords": ["99元", "129元", "20GB", "30GB", "400分钟", "500分钟"],
        "expected_answer": "99套餐20GB/400分钟，129套餐30GB/500分钟",
        "anti_keywords": []
    },
    {
        "id": 12,
        "q": "59和99套餐有什么区别？",
        "category": "对比查询",
        "expected_keywords": ["59元", "99元", "10GB", "20GB"],
        "expected_answer": "59套餐10GB/200分钟，99套餐20GB/400分钟",
        "anti_keywords": []
    },
    {
        "id": 13,
        "q": "129和199套餐哪个流量多？",
        "category": "对比查询",
        "expected_keywords": ["199", "60GB", "30GB"],
        "expected_answer": "199套餐流量更多，60GB vs 30GB",
        "anti_keywords": []
    },
    {
        "id": 14,
        "q": "199和299套餐通话时长差多少？",
        "category": "对比查询",
        "expected_keywords": ["1000分钟", "1500分钟", "500分钟"],
        "expected_answer": "199套餐1000分钟，299套餐1500分钟，差500分钟",
        "anti_keywords": []
    },
    {
        "id": 15,
        "q": "各套餐副卡数量分别是多少？",
        "category": "对比查询",
        "expected_keywords": ["59", "99", "129", "2张", "4张"],
        "expected_answer": "59/99套餐2张，129/199/299套餐4张",
        "anti_keywords": []
    },

    # ── 优惠方案查询（5题）──
    {
        "id": 16,
        "q": "99套餐全额预存实付多少？",
        "category": "优惠查询",
        "expected_keywords": ["69元", "69"],
        "expected_answer": "实付69元/月",
        "anti_keywords": []
    },
    {
        "id": 17,
        "q": "129套餐橙分期补贴多少？",
        "category": "优惠查询",
        "expected_keywords": ["960", "1440"],
        "expected_answer": "24个月960元，36个月1440元",
        "anti_keywords": []
    },
    {
        "id": 18,
        "q": "199套餐全额预存实付多少？",
        "category": "优惠查询",
        "expected_keywords": ["139元", "139"],
        "expected_answer": "实付139元/月",
        "anti_keywords": []
    },
    {
        "id": 19,
        "q": "全额预存和橙分期可以一起办吗？",
        "category": "优惠查询",
        "expected_keywords": ["不可以", "互斥", "不能"],
        "expected_answer": "不可以，两种优惠互斥",
        "anti_keywords": ["可以"]
    },
    {
        "id": 20,
        "q": "实付89元对应哪个套餐？",
        "category": "优惠查询",
        "expected_keywords": ["129", "129元"],
        "expected_answer": "原价129元套餐的优惠价格",
        "anti_keywords": []
    },

    # ── 宽带/融合查询（3题）──
    {
        "id": 21,
        "q": "99套餐有宽带吗？",
        "category": "宽带查询",
        "expected_keywords": ["100M", "宽带", "城中村"],
        "expected_answer": "城中村可装100M宽带",
        "anti_keywords": ["没有", "无宽带"]
    },
    {
        "id": 22,
        "q": "199套餐可以装千兆宽带吗？",
        "category": "宽带查询",
        "expected_keywords": ["1000M", "千兆", "可以"],
        "expected_answer": "可以装1000M宽带",
        "anti_keywords": ["不可以", "没有"]
    },
    {
        "id": 23,
        "q": "129套餐宽带速率是多少？",
        "category": "宽带查询",
        "expected_keywords": ["300M", "300"],
        "expected_answer": "300M宽带",
        "anti_keywords": ["100M", "1000M"]
    },

    # ── 转网/规则查询（4题）──
    {
        "id": 24,
        "q": "99套餐可以转几个携号转网？",
        "category": "转网查询",
        "expected_keywords": ["1个", "1", "移动"],
        "expected_answer": "只能转1个移动",
        "anti_keywords": []
    },
    {
        "id": 25,
        "q": "199套餐最多可以转几个号？",
        "category": "转网查询",
        "expected_keywords": ["3个", "3"],
        "expected_answer": "最多转3个",
        "anti_keywords": []
    },
    {
        "id": 26,
        "q": "副卡减免规则是什么？",
        "category": "规则查询",
        "expected_keywords": ["129元及以上", "2张以上", "99元及以下"],
        "expected_answer": "129元及以上可开2张以上副卡，99元及以下只能开2张",
        "anti_keywords": []
    },
    {
        "id": 27,
        "q": "携号转网需要什么条件？",
        "category": "规则查询",
        "expected_keywords": ["授权码", "合约", "副卡"],
        "expected_answer": "需获取授权码、解除合约、先转副卡再转主卡",
        "anti_keywords": []
    },

    # ── 边界/复杂查询（3题）──
    {
        "id": 28,
        "q": "29元套餐包含什么？",
        "category": "边界查询",
        "expected_keywords": ["10GB", "100分钟", "29元"],
        "expected_answer": "10GB流量，100分钟通话",
        "anti_keywords": []
    },
    {
        "id": 29,
        "q": "169套餐是融合套餐吗？流量多少？",
        "category": "复杂查询",
        "expected_keywords": ["169", "40GB", "40G", "融合"],
        "expected_answer": "169元融合套餐，40GB流量",
        "anti_keywords": []
    },
    {
        "id": 30,
        "q": "哪个套餐有黄金会员？",
        "category": "复杂查询",
        "expected_keywords": ["99", "99元", "黄金"],
        "expected_answer": "99元套餐对应黄金会员",
        "anti_keywords": []
    },
]


def run_evaluation():
    """执行评测"""
    print("=" * 70)
    print("RAG 评测 — 30道题 × 5个维度")
    print("=" * 70)

    rag = SimpleRAG()
    results = []

    for i, item in enumerate(EVAL_QUESTIONS):
        q = item["q"]
        print(f"\n[{i+1}/30] {q}")

        start = time.time()
        try:
            result = rag.query(q)
            elapsed = time.time() - start
            answer = result.answer if hasattr(result, 'answer') else str(result)
            sources = result.sources if hasattr(result, 'sources') else []
            success = result.success if hasattr(result, 'success') else True
        except Exception as e:
            elapsed = time.time() - start
            answer = f"ERROR: {e}"
            sources = []
            success = False

        print(f"  回答: {answer[:150]}...")
        print(f"  耗时: {elapsed:.2f}s")

        results.append({
            "id": item["id"],
            "question": q,
            "category": item["category"],
            "answer": answer,
            "sources_count": len(sources),
            "elapsed": round(elapsed, 3),
            "success": success,
            "expected_keywords": item["expected_keywords"],
            "expected_answer": item["expected_answer"],
            "anti_keywords": item["anti_keywords"],
        })

    return results


def score_results(results):
    """对结果进行5维度打分"""
    print("\n" + "=" * 70)
    print("评分结果")
    print("=" * 70)

    scored = []
    category_scores = {}

    for r in results:
        answer = r["answer"]
        expected_kw = r["expected_keywords"]
        anti_kw = r["anti_keywords"]

        # 1. 召回分（0-10）：答案是否包含期望关键词
        kw_hits = sum(1 for kw in expected_kw if kw in answer)
        kw_ratio = kw_hits / len(expected_kw) if expected_kw else 1
        recall_score = round(kw_ratio * 10, 1)

        # 2. 耗时分（0-10）：<3s满分，>15s零分
        elapsed = r["elapsed"]
        if elapsed <= 3:
            latency_score = 10
        elif elapsed <= 6:
            latency_score = 8
        elif elapsed <= 10:
            latency_score = 6
        elif elapsed <= 15:
            latency_score = 4
        else:
            latency_score = 2

        # 3. 完整性分（0-10）：答案长度和信息密度
        answer_len = len(answer)
        if answer_len >= 100:
            completeness_score = 10
        elif answer_len >= 50:
            completeness_score = 8
        elif answer_len >= 20:
            completeness_score = 6
        elif answer_len >= 5:
            completeness_score = 4
        else:
            completeness_score = 2

        # 4. 准确率分（0-10）：关键词命中率 × 无幻觉
        accuracy_score = recall_score  # 基础分等于召回分
        # 如果有反向关键词出现（幻觉），扣分
        hallucination_hits = [kw for kw in anti_kw if kw in answer]
        if hallucination_hits:
            accuracy_score = max(0, accuracy_score - len(hallucination_hits) * 3)

        # 5. 幻觉分（0-10）：检查是否包含明显错误信息
        hallucination_score = 10
        # 检查反向关键词
        if hallucination_hits:
            hallucination_score = max(0, 10 - len(hallucination_hits) * 5)
        # 检查是否包含"抱歉"、"未找到"等失败标志
        fail_markers = ["抱歉", "未找到", "查询失败", "ERROR", "错误"]
        if any(m in answer for m in fail_markers):
            hallucination_score = max(0, hallucination_score - 3)
        # 检查是否有明显矛盾信息（如同时出现互斥的价格）
        if "实付" in answer and "原价" in answer:
            # 正常，两种价格都可能出现
            pass

        score = {
            "id": r["id"],
            "question": r["question"],
            "category": r["category"],
            "recall": recall_score,
            "latency": latency_score,
            "completeness": completeness_score,
            "accuracy": accuracy_score,
            "hallucination": hallucination_score,
            "total": round((recall_score + latency_score + completeness_score + accuracy_score + hallucination_score) / 5, 1),
            "elapsed": r["elapsed"],
            "answer_preview": answer[:100],
            "kw_hits": kw_hits,
            "kw_total": len(expected_kw),
            "hallucination_hits": hallucination_hits,
        }
        scored.append(score)

        # 按类别统计
        cat = r["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(score)

    return scored, category_scores


def print_report(scored, category_scores):
    """打印评测报告"""
    print("\n" + "=" * 70)
    print("详细评分表")
    print("=" * 70)
    print(f"{'ID':>3} {'类别':<8} {'召回':>4} {'耗时':>4} {'完整':>4} {'准确':>4} {'幻觉':>4} {'总分':>5} {'耗时s':>6} {'命中':>5}")
    print("-" * 70)

    for s in scored:
        print(f"{s['id']:>3} {s['category']:<8} {s['recall']:>4.1f} {s['latency']:>4} {s['completeness']:>4} {s['accuracy']:>4.1f} {s['hallucination']:>4} {s['total']:>5.1f} {s['elapsed']:>6.2f} {s['kw_hits']}/{s['kw_total']}")

    # 总体统计
    print("\n" + "=" * 70)
    print("总体统计")
    print("=" * 70)

    dims = ["recall", "latency", "completeness", "accuracy", "hallucination", "total"]
    for dim in dims:
        vals = [s[dim] for s in scored]
        avg = sum(vals) / len(vals)
        print(f"  {dim:<15} 平均: {avg:.2f}  最低: {min(vals):.1f}  最高: {max(vals):.1f}")

    avg_latency = sum(s["elapsed"] for s in scored) / len(scored)
    print(f"  {'avg_latency':<15} 平均: {avg_latency:.2f}s")

    # 按类别统计
    print("\n" + "=" * 70)
    print("按类别统计")
    print("=" * 70)

    for cat, items in sorted(category_scores.items()):
        avg_total = sum(s["total"] for s in items) / len(items)
        avg_recall = sum(s["recall"] for s in items) / len(items)
        avg_latency = sum(s["elapsed"] for s in items) / len(items)
        print(f"  {cat:<10} {len(items)}题  总分:{avg_total:.1f}  召回:{avg_latency:.1f}  耗时:{avg_latency:.2f}s")

    # 低分题目
    print("\n" + "=" * 70)
    print("低分题目（总分 < 6）")
    print("=" * 70)

    low_scores = [s for s in scored if s["total"] < 6]
    if low_scores:
        for s in low_scores:
            print(f"  [{s['id']}] {s['question']}")
            print(f"      总分:{s['total']} 召回:{s['recall']} 准确:{s['accuracy']} 幻觉:{s['hallucination']}")
            print(f"      回答: {s['answer_preview']}...")
            if s["hallucination_hits"]:
                print(f"      ⚠️ 幻觉关键词: {s['hallucination_hits']}")
    else:
        print("  无低分题目 ✓")

    # 幻觉题目
    print("\n" + "=" * 70)
    print("幻觉检测（hallucination < 10）")
    print("=" * 70)

    halluc = [s for s in scored if s["hallucination"] < 10]
    if halluc:
        for s in halluc:
            print(f"  [{s['id']}] {s['question']}")
            print(f"      幻觉分:{s['hallucination']}  幻觉词: {s['hallucination_hits']}")
            print(f"      回答: {s['answer_preview']}...")
    else:
        print("  无幻觉 ✓")

    return scored


def save_results(scored, output_path="eval_rag_30_results.json"):
    """保存结果到JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    results = run_evaluation()
    scored, category_scores = score_results(results)
    print_report(scored, category_scores)
    save_results(scored)
