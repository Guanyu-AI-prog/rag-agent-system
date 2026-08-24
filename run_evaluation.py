#!/usr/bin/env python3
"""
完整评测脚本：对比 taocan_agent.py 和 simple_rag.py
"""

import time
import json
import sys
import os
from datetime import datetime
from test_questions import TEST_CASES

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def evaluate_taocan_agent():
    """评测 taocan_agent.py（LangChain 版本）"""
    print("\n" + "=" * 70)
    print("🚀 评测 taocan_agent.py (LangChain 版本)")
    print("=" * 70)

    from taocan_agent import run_single

    results = []
    for tc in TEST_CASES:
        print(f"\n[{tc['id']}/30] [{tc['type']}] {tc['question']}")

        try:
            start = time.time()
            result = run_single(tc['question'], verbose=False)
            elapsed = time.time() - start

            hits = sum(1 for kw in tc['expected_keywords'] if kw in result['answer'])
            hit_rate = hits / len(tc['expected_keywords'])
            passed = hit_rate >= 0.5

            status = "✅" if passed else "❌"
            print(f"{status} 命中 {hits}/{len(tc['expected_keywords'])} | {elapsed:.2f}s")
            print(f"   回答: {result['answer'][:100]}...")

            results.append({
                "id": tc['id'],
                "type": tc['type'],
                "question": tc['question'],
                "passed": passed,
                "hit_rate": hit_rate,
                "elapsed": elapsed,
                "answer": result['answer'][:200]
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
                "answer": str(e)
            })

        # 题间等待，避免 API 限流
        if tc['id'] < len(TEST_CASES):
            time.sleep(5)

    return results


def evaluate_simple_rag():
    """评测 simple_rag.py（纯 Python 版本）"""
    print("\n" + "=" * 70)
    print("🚀 评测 simple_rag.py (纯 Python 版本)")
    print("=" * 70)

    from simple_rag import SimpleRAG

    rag = SimpleRAG()

    results = []
    for tc in TEST_CASES:
        print(f"\n[{tc['id']}/30] [{tc['type']}] {tc['question']}")

        try:
            start = time.time()
            result = rag.query(tc['question'])
            elapsed = time.time() - start

            hits = sum(1 for kw in tc['expected_keywords'] if kw in result.answer)
            hit_rate = hits / len(tc['expected_keywords'])
            passed = hit_rate >= 0.5

            status = "✅" if passed else "❌"
            print(f"{status} 命中 {hits}/{len(tc['expected_keywords'])} | {elapsed:.2f}s")
            print(f"   回答: {result.answer[:100]}...")

            results.append({
                "id": tc['id'],
                "type": tc['type'],
                "question": tc['question'],
                "passed": passed,
                "hit_rate": hit_rate,
                "elapsed": elapsed,
                "answer": result.answer[:200]
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
                "answer": str(e)
            })

        # 题间等待
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

    return {"total": total, "passed": passed, "failed": failed, "pass_rate": passed/total*100}


def compare_results(results1, results2, name1, name2):
    """对比两个系统的评测结果"""
    print(f"\n{'=' * 70}")
    print(f"📊 对比分析: {name1} vs {name2}")
    print(f"{'=' * 70}")

    # 整体对比
    stats1 = {"total": len(results1), "passed": sum(1 for r in results1 if r['passed'])}
    stats2 = {"total": len(results2), "passed": sum(1 for r in results2 if r['passed'])}

    print(f"\n整体对比:")
    print(f"  {name1}: {stats1['passed']}/{stats1['total']} ({stats1['passed']/stats1['total']*100:.1f}%)")
    print(f"  {name2}: {stats2['passed']}/{stats2['total']} ({stats2['passed']/stats2['total']*100:.1f}%)")

    # 延时对比
    lat1 = [r['elapsed'] for r in results1 if r['elapsed'] > 0]
    lat2 = [r['elapsed'] for r in results2 if r['elapsed'] > 0]

    print(f"\n延时对比:")
    print(f"  {name1}: 平均 {sum(lat1)/len(lat1):.2f}s")
    print(f"  {name2}: 平均 {sum(lat2)/len(lat2):.2f}s")
    print(f"  速度提升: {(1 - (sum(lat2)/len(lat2)) / (sum(lat1)/len(lat1))) * 100:.1f}%")

    # 差异题目
    print(f"\n差异题目:")
    diff_count = 0
    for r1, r2 in zip(results1, results2):
        if r1['passed'] != r2['passed']:
            diff_count += 1
            status1 = "✅" if r1['passed'] else "❌"
            status2 = "✅" if r2['passed'] else "❌"
            print(f"  [{r1['id']}] {r1['type']}")
            print(f"       {name1}: {status1} | {name2}: {status2}")

    if diff_count == 0:
        print(f"  无差异（两系统表现一致）")


def main():
    """主函数"""
    print("🚀 开始完整评测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"题目数量: {len(TEST_CASES)}")

    # 评测 taocan_agent.py
    results1 = evaluate_taocan_agent()
    stats1 = print_statistics(results1, "taocan_agent.py (LangChain)")

    # 评测 simple_rag.py
    results2 = evaluate_simple_rag()
    stats2 = print_statistics(results2, "simple_rag.py (纯 Python)")

    # 对比分析
    compare_results(results1, results2, "LangChain", "纯Python")

    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": len(TEST_CASES),
        "taocan_agent": {"stats": stats1, "results": results1},
        "simple_rag": {"stats": stats2, "results": results2}
    }

    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评测完成，结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
