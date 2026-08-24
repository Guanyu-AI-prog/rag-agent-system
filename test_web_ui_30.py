#!/usr/bin/env python3
"""
Web UI 30道题完整评测
"""

import requests
import json
import time
import sys
import os

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 配置
BASE_URL = "http://8.163.101.180:8001"

# 导入测试题目
from test_questions import TEST_CASES


def query_web_ui(question: str, session_id: str = None) -> dict:
    """查询 Web UI API"""
    if session_id is None:
        session_id = f"test_{int(time.time())}"

    try:
        payload = {
            "question": question,
            "session_id": session_id
        }
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=180
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"状态码: {response.status_code}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def evaluate_answer(answer: str, expected_keywords: list) -> dict:
    """评估回答质量"""
    if not answer:
        return {"hits": 0, "total": len(expected_keywords), "hit_rate": 0, "passed": False}

    hits = sum(1 for kw in expected_keywords if kw in answer)
    hit_rate = hits / len(expected_keywords) if expected_keywords else 0

    return {
        "hits": hits,
        "total": len(expected_keywords),
        "hit_rate": hit_rate,
        "passed": hit_rate >= 0.5
    }


def run_evaluation():
    """运行完整评测"""
    print("\n" + "=" * 70)
    print("🚀 Web UI 30道题完整评测")
    print(f"目标地址: {BASE_URL}")
    print("=" * 70)

    # 检查连接
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ 无法连接到 Web UI")
            return
        print("✅ Web UI 连接正常")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 检查配置
    try:
        stats = requests.get(f"{BASE_URL}/stats", timeout=5).json()
        print(f"📊 当前配置:")
        print(f"   LLM 模型: {stats.get('llm_model', '未知')}")
        print(f"   嵌入模型: {stats.get('embedding_model', '未知')}")
        print(f"   文档数量: {stats.get('document_count', '未知')}")
    except Exception:
        pass

    print("\n" + "-" * 70)

    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "by_type": {},
        "latency": [],
        "details": []
    }

    for tc in TEST_CASES:
        q_type = tc["type"]
        if q_type not in results["by_type"]:
            results["by_type"][q_type] = {"total": 0, "passed": 0, "failed": 0, "latency": []}

        results["total"] += 1
        results["by_type"][q_type]["total"] += 1

        print(f"\n[{tc['id']}/30] [{q_type}] {tc['question']}")

        # 查询
        start_time = time.time()
        response = query_web_ui(tc["question"])
        elapsed = time.time() - start_time

        results["latency"].append(elapsed)
        results["by_type"][q_type]["latency"].append(elapsed)

        # 检查是否有错误
        if "error" in response or not response.get("success", True):
            error_msg = response.get("error", "未知错误")
            print(f"❌ 查询失败: {error_msg}")
            results["failed"] += 1
            results["by_type"][q_type]["failed"] += 1
            results["details"].append({
                "id": tc["id"],
                "type": q_type,
                "question": tc["question"],
                "status": "error",
                "error": error_msg,
                "elapsed": elapsed
            })
            time.sleep(2)
            continue

        answer = response.get("answer", "")
        print(f"回答: {answer[:150]}...")

        # 评估
        eval_result = evaluate_answer(answer, tc["expected_keywords"])

        if eval_result["passed"]:
            results["passed"] += 1
            results["by_type"][q_type]["passed"] += 1
            status = "✅ PASS"
        else:
            results["failed"] += 1
            results["by_type"][q_type]["failed"] += 1
            status = "❌ FAIL"

        print(f"{status} (命中 {eval_result['hits']}/{eval_result['total']}) ⏱ {elapsed:.2f}s")

        results["details"].append({
            "id": tc["id"],
            "type": q_type,
            "question": tc["question"],
            "status": "pass" if eval_result["passed"] else "fail",
            "hits": eval_result["hits"],
            "total": eval_result["total"],
            "elapsed": elapsed,
            "answer": answer[:200]
        })

        # 间隔，避免限流
        time.sleep(2)

    # 打印统计
    print(f"\n{'=' * 70}")
    print(f"📊 Web UI 评测结果")
    print(f"{'=' * 70}")
    print(f"\n总计: {results['total']} | ✅ 通过: {results['passed']} | ❌ 失败: {results['failed']}")
    print(f"通过率: {results['passed']/results['total']*100:.1f}%")

    # 延时统计
    if results["latency"]:
        avg_lat = sum(results["latency"]) / len(results["latency"])
        print(f"\n⏱  延时统计:")
        print(f"   总耗时: {sum(results['latency']):.2f}s")
        print(f"   平均延时: {avg_lat:.2f}s")
        print(f"   最快: {min(results['latency']):.2f}s")
        print(f"   最慢: {max(results['latency']):.2f}s")

    # 按题型统计
    print(f"\n📋 按题型统计:")
    for q_type, stats in results["by_type"].items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_lat = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else 0
        print(f"   {q_type}: {stats['passed']}/{stats['total']} ({rate:.1f}%) | 平均 {avg_lat:.2f}s")

    # 保存结果
    output_file = f"web_ui_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 评测结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    run_evaluation()
