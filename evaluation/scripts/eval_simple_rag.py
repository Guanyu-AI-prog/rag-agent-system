#!/usr/bin/env python3
"""
评测 simple_rag.py — 30道题，5个维度打分
维度：召回率、检索耗时、完整性、准确率、幻觉率
"""

import time
import json
import sys
import os
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 路径引导：simple_rag 在 core/，其 config 依赖在 infra/
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_BASE, 'core'), os.path.join(_BASE, 'infra')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_questions import TEST_CASES


# ========== 5维度评分 ==========

def score_single(tc, answer, elapsed, sources_count):
    """对单条结果进行5维度打分"""
    expected_kw = tc["expected_keywords"]
    reference = tc.get("reference", "")

    # 1. 召回率（0-10）：答案是否包含期望关键词
    kw_hits = sum(1 for kw in expected_kw if kw in answer)
    kw_ratio = kw_hits / len(expected_kw) if expected_kw else 1
    recall_score = round(kw_ratio * 10, 1)

    # 2. 检索耗时分（0-10）：<3s满分，>15s零分
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
    if answer_len >= 150:
        completeness_score = 10
    elif answer_len >= 100:
        completeness_score = 8
    elif answer_len >= 50:
        completeness_score = 6
    elif answer_len >= 20:
        completeness_score = 4
    else:
        completeness_score = 2

    # 4. 准确率分（0-10）：关键词命中率
    accuracy_score = recall_score

    # 5. 幻觉分（0-10）：检查失败标志和矛盾信息
    hallucination_score = 10
    fail_markers = ["抱歉", "未找到", "查询失败", "ERROR", "错误", "无法"]
    hit_count = sum(1 for m in fail_markers if m in answer)
    if hit_count:
        hallucination_score = max(0, hallucination_score - hit_count * 3)

    # 如果期望关键词一个都没命中，且答案不为空，可能是幻觉
    if kw_hits == 0 and answer_len > 20:
        hallucination_score = max(0, hallucination_score - 4)

    total = round((recall_score + latency_score + completeness_score + accuracy_score + hallucination_score) / 5, 1)

    return {
        "recall": recall_score,
        "latency_score": latency_score,
        "completeness": completeness_score,
        "accuracy": accuracy_score,
        "hallucination": hallucination_score,
        "total": total,
        "kw_hits": kw_hits,
        "kw_total": len(expected_kw),
    }


# ========== 评测主流程 ==========

def evaluate_simple_rag():
    """评测 simple_rag.py"""
    print("\n" + "=" * 70)
    print("🚀 评测 simple_rag.py — 30道题 × 5维度")
    print("=" * 70)

    from simple_rag import SimpleRAG
    rag = SimpleRAG()

    results = []
    for tc in TEST_CASES:
        print(f"\n[{tc['id']:2d}/30] [{tc['type']}] {tc['question']}")

        try:
            start = time.time()
            result = rag.query(tc['question'])
            elapsed = time.time() - start

            answer = result.answer
            sources_count = len(result.sources) if result.sources else 0

            # 5维度打分
            scores = score_single(tc, answer, elapsed, sources_count)

            status = "✅" if scores["total"] >= 6 else "❌"
            print(f"{status} 总分:{scores['total']:4.1f} | 召回:{scores['recall']:4.1f} 耗时:{scores['latency_score']:2d} 完整:{scores['completeness']:2d} 准确:{scores['accuracy']:4.1f} 幻觉:{scores['hallucination']:2d} | {elapsed:.2f}s")
            print(f"   命中: {scores['kw_hits']}/{scores['kw_total']} | 回答: {answer[:120]}...")

            results.append({
                "id": tc['id'],
                "type": tc['type'],
                "question": tc['question'],
                "answer": answer[:500],
                "elapsed": round(elapsed, 3),
                "sources_count": sources_count,
                "expected_keywords": tc['expected_keywords'],
                "reference": tc.get('reference', ''),
                **scores,
            })

        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({
                "id": tc['id'],
                "type": tc['type'],
                "question": tc['question'],
                "answer": str(e),
                "elapsed": 0,
                "sources_count": 0,
                "expected_keywords": tc['expected_keywords'],
                "reference": tc.get('reference', ''),
                "recall": 0, "latency_score": 0, "completeness": 0,
                "accuracy": 0, "hallucination": 0, "total": 0,
                "kw_hits": 0, "kw_total": len(tc['expected_keywords']),
            })

        # 题间等待，避免 API 限流
        if tc['id'] < len(TEST_CASES):
            time.sleep(3)

    return results


# ========== 报告输出 ==========

def print_report(results):
    """打印完整评测报告"""
    total = len(results)

    # ===== 详细评分表 =====
    print(f"\n{'=' * 80}")
    print("📊 详细评分表")
    print(f"{'=' * 80}")
    print(f"{'ID':>3} {'类别':<8} {'召回':>4} {'耗时':>4} {'完整':>4} {'准确':>4} {'幻觉':>4} {'总分':>5} {'耗时s':>6} {'命中':>5}")
    print("-" * 80)

    for r in results:
        print(f"{r['id']:>3} {r['type']:<8} {r['recall']:>4.1f} {r['latency_score']:>4} {r['completeness']:>4} {r['accuracy']:>4.1f} {r['hallucination']:>4} {r['total']:>5.1f} {r['elapsed']:>6.2f} {r['kw_hits']}/{r['kw_total']}")

    # ===== 5维度总体统计 =====
    print(f"\n{'=' * 80}")
    print("📈 5维度总体统计")
    print(f"{'=' * 80}")

    dims = [
        ("召回率 (recall)", "recall"),
        ("检索耗时 (latency)", "latency_score"),
        ("完整性 (completeness)", "completeness"),
        ("准确率 (accuracy)", "accuracy"),
        ("幻觉率 (hallucination)", "hallucination"),
        ("总分 (total)", "total"),
    ]

    for label, key in dims:
        vals = [r[key] for r in results]
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"  {label:<28} 平均: {avg:5.2f}  最低: {mn:4.1f}  最高: {mx:4.1f}")

    # 耗时统计
    latencies = [r['elapsed'] for r in results if r['elapsed'] > 0]
    if latencies:
        print(f"\n  {'实际耗时':<28} 平均: {sum(latencies)/len(latencies):5.2f}s  总计: {sum(latencies):.2f}s  最快: {min(latencies):.2f}s  最慢: {max(latencies):.2f}s")

    # ===== 按类别统计 =====
    print(f"\n{'=' * 80}")
    print("📋 按类别统计")
    print(f"{'=' * 80}")

    by_type = {}
    for r in results:
        t = r['type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r)

    for t, items in sorted(by_type.items()):
        avg_total = sum(r['total'] for r in items) / len(items)
        avg_recall = sum(r['recall'] for r in items) / len(items)
        avg_latency = sum(r['elapsed'] for r in items) / len(items)
        avg_complete = sum(r['completeness'] for r in items) / len(items)
        avg_accuracy = sum(r['accuracy'] for r in items) / len(items)
        avg_halluc = sum(r['hallucination'] for r in items) / len(items)
        passed = sum(1 for r in items if r['total'] >= 6)
        print(f"  {t:<10} {len(items)}题  通过:{passed}/{len(items)}  总分:{avg_total:.1f}  召回:{avg_recall:.1f}  耗时:{avg_latency:.2f}s  完整:{avg_complete:.1f}  准确:{avg_accuracy:.1f}  幻觉:{avg_halluc:.1f}")

    # ===== 通过率 =====
    passed = sum(1 for r in results if r['total'] >= 6)
    print(f"\n{'=' * 80}")
    print(f"✅ 通过率: {passed}/{total} ({passed/total*100:.1f}%)  [总分≥6为通过]")
    print(f"{'=' * 80}")

    # ===== 低分题目 =====
    low = [r for r in results if r['total'] < 6]
    if low:
        print(f"\n{'=' * 80}")
        print(f"❌ 低分题目（总分 < 6）共 {len(low)} 题")
        print(f"{'=' * 80}")
        for r in low:
            print(f"  [{r['id']:2d}] {r['type']} | 总分:{r['total']} 召回:{r['recall']} 准确:{r['accuracy']} 幻觉:{r['hallucination']}")
            print(f"       问题: {r['question']}")
            print(f"       回答: {r['answer'][:150]}...")

    # ===== 幻觉题目 =====
    halluc = [r for r in results if r['hallucination'] < 10]
    if halluc:
        print(f"\n{'=' * 80}")
        print(f"⚠️  幻觉检测（hallucination < 10）共 {len(halluc)} 题")
        print(f"{'=' * 80}")
        for r in halluc:
            print(f"  [{r['id']:2d}] 幻觉分:{r['hallucination']} | {r['question']}")
            print(f"       回答: {r['answer'][:150]}...")

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total * 100, 1),
    }


# ========== 主函数 ==========

def main():
    print("🚀 开始 simple_rag.py 评测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"题目数量: {len(TEST_CASES)}")

    results = evaluate_simple_rag()
    stats = print_report(results)

    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": len(TEST_CASES),
        "stats": stats,
        "results": results,
    }

    output_file = f"eval_simple_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评测完成，结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
