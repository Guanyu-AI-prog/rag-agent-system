#!/usr/bin/env python3
"""
评测 dx_agent.py (纯 Python Agent 版本)
使用 test_questions.py 中的 30 道标准测试题
"""

import time
import json
import sys
import os
from datetime import datetime

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from test_questions import TEST_CASES


def evaluate_dx_agent():
    """评测 dx_agent.py"""
    print("\n" + "=" * 70)
    print("🚀 评测 dx_agent.py (纯 Python Agent 版本)")
    print("=" * 70)

    from dx_agent import run_single

    results = []
    for tc in TEST_CASES:
        print(f"\n[{tc['id']}/30] [{tc['type']}] {tc['question']}")

        try:
            start = time.time()
            result = run_single(tc['question'], verbose=False)
            elapsed = time.time() - start

            answer = result['answer']
            hits = sum(1 for kw in tc['expected_keywords'] if kw in answer)
            hit_rate = hits / len(tc['expected_keywords'])
            passed = hit_rate >= 0.5

            status = "✅" if passed else "❌"
            print(f"{status} 命中 {hits}/{len(tc['expected_keywords'])} | {elapsed:.2f}s")
            print(f"   回答: {answer[:150]}...")

            results.append({
                "id": tc['id'],
                "type": tc['type'],
                "question": tc['question'],
                "passed": passed,
                "hit_rate": hit_rate,
                "elapsed": elapsed,
                "answer": answer[:300],
                "hits": hits,
                "expected_count": len(tc['expected_keywords']),
                "expected_keywords": tc['expected_keywords'],
            })

        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({
                "id": tc['id'],
                "type": tc['type'],
                "question": tc['question'],
                "passed": False,
                "hit_rate": 0,
                "elapsed": 0,
                "answer": str(e),
                "hits": 0,
                "expected_count": len(tc['expected_keywords']),
                "expected_keywords": tc['expected_keywords'],
            })

        # 题间等待，避免 API 限流
        if tc['id'] < len(TEST_CASES):
            time.sleep(5)

    return results


def print_statistics(results, name):
    """打印统计结果"""
    print(f"\n{'=' * 70}")
    print(f"📊 {name} 评测结果")
    print(f"{'=' * 70}")

    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed

    print(f"\n总计: {total} | ✅ 通过: {passed} | ❌ 失败: {failed}")
    print(f"通过率: {passed/total*100:.1f}%")

    # 延时统计
    latencies = [r['elapsed'] for r in results if r['elapsed'] > 0]
    if latencies:
        print(f"\n⏱  延时统计:")
        print(f"   总耗时: {sum(latencies):.2f}s")
        print(f"   平均延时: {sum(latencies)/len(latencies):.2f}s")
        print(f"   最快: {min(latencies):.2f}s")
        print(f"   最慢: {max(latencies):.2f}s")

    # 按题型统计
    print(f"\n📋 按题型统计:")
    by_type = {}
    for r in results:
        q_type = r['type']
        if q_type not in by_type:
            by_type[q_type] = {"total": 0, "passed": 0, "latencies": []}
        by_type[q_type]["total"] += 1
        if r['passed']:
            by_type[q_type]["passed"] += 1
        if r['elapsed'] > 0:
            by_type[q_type]["latencies"].append(r['elapsed'])

    for q_type, stats in by_type.items():
        rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_lat = sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0
        print(f"   {q_type}: {stats['passed']}/{stats['total']} ({rate:.1f}%) | 平均 {avg_lat:.2f}s")

    # 失败题目详情
    failed_list = [r for r in results if not r['passed']]
    if failed_list:
        print(f"\n❌ 失败题目详情:")
        for r in failed_list:
            print(f"  [{r['id']:2d}] {r['type']} | 命中 {r['hits']}/{r['expected_count']}")
            print(f"       问题: {r['question']}")
            print(f"       期望: {r['expected_keywords']}")
            print(f"       回答: {r['answer'][:120]}...")

    return {"total": total, "passed": passed, "failed": failed, "pass_rate": passed/total*100}


def main():
    """主函数"""
    print("🚀 开始 dx_agent.py 评测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"题目数量: {len(TEST_CASES)}")

    # 评测 dx_agent.py
    results = evaluate_dx_agent()
    stats = print_statistics(results, "dx_agent.py (纯 Python Agent)")

    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": len(TEST_CASES),
        "dx_agent": {"stats": stats, "results": results},
    }

    output_file = f"eval_dx_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评测完成，结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
