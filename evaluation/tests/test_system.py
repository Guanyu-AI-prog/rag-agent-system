"""
RAG系统测试脚本
用于验证系统各组件是否正常工作
"""

import os
import sys
import time
import requests
from typing import Dict, Any

API_BASE_URL = "http://localhost:8001"
TEST_QUESTIONS = [
    "有哪些套餐？",
    "流量怎么算？",
    "如何办理携号转网？"
]


def test_health() -> bool:
    print("测试健康检查接口...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"健康检查通过: {data}")
            return True
        else:
            print(f"健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"连接失败: {e}")
        return False


def test_stats() -> Dict[str, Any]:
    print("测试统计接口...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"统计信息: {data}")
            return data
        else:
            print(f"统计接口失败: {response.status_code}")
            return {}
    except Exception as e:
        print(f"统计接口错误: {e}")
        return {}


def test_query(question: str) -> Dict[str, Any]:
    print(f"测试查询: {question}")
    try:
        payload = {"question": question}
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=60
        )
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            processing_time = end_time - start_time
            print(f"查询成功 (耗时: {processing_time:.2f}秒)")
            print(f"   问题: {question}")
            print(f"   回答: {data.get('answer', '无回答')[:100]}...")
            print(f"   来源: {len(data.get('sources', []))} 个文档")
            return data
        else:
            print(f"查询失败: {response.status_code}")
            print(f"   错误: {response.text}")
            return {}
    except Exception as e:
        print(f"查询错误: {e}")
        return {}


def test_batch_query():
    print("测试批量查询...")
    try:
        payload = TEST_QUESTIONS[:2]
        response = requests.post(
            f"{API_BASE_URL}/batch_query",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            print(f"批量查询成功，处理 {len(data.get('results', []))} 个问题")
            return True
        else:
            print(f"批量查询失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"批量查询错误: {e}")
        return False


def test_cache():
    print("测试缓存功能...")
    try:
        response = requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
        if response.status_code == 200:
            print("缓存清空成功")

        question = TEST_QUESTIONS[0]

        print(f"   第一次查询: {question}")
        start1 = time.time()
        test_query(question)
        time1 = time.time() - start1

        print(f"   第二次查询: {question}")
        start2 = time.time()
        test_query(question)
        time2 = time.time() - start2

        print(f"   第一次耗时: {time1:.2f}秒")
        print(f"   第二次耗时: {time2:.2f}秒")

        if time2 < time1 * 0.1:
            print("缓存功能正常")
            return True
        else:
            print("缓存可能未生效")
            return False

    except Exception as e:
        print(f"缓存测试错误: {e}")
        return False


def run_all_tests():
    print("=" * 50)
    print("RAG系统测试套件")
    print("=" * 50)

    if not test_health():
        print("\n服务未运行，请先启动: python api.py")
        return False

    print("\n" + "=" * 50)

    stats = test_stats()

    print("\n" + "=" * 50)

    for question in TEST_QUESTIONS[:1]:
        test_query(question)
        print()

    print("=" * 50)

    test_batch_query()

    print("\n" + "=" * 50)

    test_cache()

    print("\n" + "=" * 50)
    print("所有测试完成")
    print("=" * 50)

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            test_health()
        elif sys.argv[1] == "stats":
            test_stats()
        elif sys.argv[1] == "query":
            if len(sys.argv) > 2:
                question = " ".join(sys.argv[2:])
                test_query(question)
            else:
                print("用法: python test_system.py query <问题>")
        else:
            print("用法: python test_system.py [health|stats|query <问题>]")
    else:
        run_all_tests()
