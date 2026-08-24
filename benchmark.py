#!/usr/bin/env python3
"""
RAG API 并发压测脚本 - 绕过缓存版本
"""

import asyncio
import aiohttp
import time
import statistics
import uuid

API_URL = "http://localhost:8001/query"

# 用随机 UUID 绕过缓存
def make_question(base: str) -> str:
    return f"{base} [测试{uuid.uuid4().hex[:8]}]"


async def send_request(session: aiohttp.ClientSession, question: str) -> dict:
    start = time.time()
    try:
        async with session.post(API_URL, json={"question": question}, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            data = await resp.json()
            return {
                "status": resp.status,
                "latency": time.time() - start,
                "success": resp.status == 200,
                "answer_len": len(data.get("answer", ""))
            }
    except asyncio.TimeoutError:
        return {"status": 0, "latency": time.time() - start, "success": False, "error": "timeout"}
    except Exception as e:
        return {"status": 0, "latency": time.time() - start, "success": False, "error": str(e)}


async def run_test(concurrency: int, total: int):
    bases = [
        "29元套餐有多少流量",
        "59元套餐包含什么",
        "129元套餐宽带",
        "学生推荐什么套餐",
        "副卡怎么收费",
        "橙分期是什么",
        "流量不够用",
        "携号转网",
        "最便宜的套餐",
        "199元和299元区别",
    ]

    questions = [make_question(bases[i % len(bases)]) for i in range(total)]
    connector = aiohttp.TCPConnector(limit=concurrency)
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        async def bounded(q):
            async with sem:
                return await send_request(session, q)

        start = time.time()
        tasks = [bounded(q) for q in questions]
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - start

    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    lats = [r["latency"] for r in ok]

    print(f"\n并发={concurrency} | 请求={total} | 成功={len(ok)} | 失败={len(fail)} | 总耗时={wall_time:.1f}s")

    if lats:
        lats_sorted = sorted(lats)
        p50 = lats_sorted[len(lats_sorted)//2]
        p95 = lats_sorted[int(len(lats_sorted)*0.95)]
        p99 = lats_sorted[int(len(lats_sorted)*0.99)]
        print(f"  延迟: 均={statistics.mean(lats):.1f}s | P50={p50:.1f}s | P95={p95:.1f}s | P99={p99:.1f}s | 最大={max(lats):.1f}s")
        print(f"  QPS: {len(ok)/wall_time:.2f} (按墙钟时间)")

    if fail:
        errs = set(r.get("error", "HTTP "+str(r["status"])) for r in fail)
        print(f"  错误: {errs}")


async def main():
    print("=== RAG API 并发压测（绕过缓存）===")
    print(f"目标: {API_URL}\n")

    # 预热
    async with aiohttp.ClientSession() as s:
        r = await send_request(s, make_question("预热"))
        print(f"预热: {'成功' if r['success'] else '失败'} | {r['latency']:.1f}s\n")

    # 测试不同并发级别
    for c in [1, 2, 5, 10, 15, 20]:
        await run_test(concurrency=c, total=20)
        await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
