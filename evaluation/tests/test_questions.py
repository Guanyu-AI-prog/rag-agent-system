#!/usr/bin/env python3
"""
套餐Agent测试问题集
覆盖6种题型，共30道
支持评测延时统计
"""

import time

TEST_CASES = [
    # ═══════════════════════════════════════════════════
    # 单点查询（6题）- 精确检索 + 事实回答
    # ═══════════════════════════════════════════════════
    {
        "id": 1,
        "type": "单点查询",
        "question": "59元套餐的月基本费是多少？包含多少流量和通话？",
        "expected_keywords": ["59元", "10GB", "200分钟"],
        "reference": "5G畅享59元套餐：月基本费59元，国内通用流量10GB，语音200分钟"
    },
    {
        "id": 2,
        "type": "单点查询",
        "question": "129元套餐最多可以办理几张副卡？",
        "expected_keywords": ["4张"],
        "reference": "129元套餐最多可办理4张副卡"
    },
    {
        "id": 3,
        "type": "单点查询",
        "question": "99元套餐的套外流量怎么收费？",
        "expected_keywords": ["3元", "1GB", "100MB"],
        "reference": "前100MB按0.03元/MB收费，达到100MB（3元）时额外赠送924MB（即3元/1GB）"
    },
    {
        "id": 4,
        "type": "单点查询",
        "question": "副卡的月功能费是多少？",
        "expected_keywords": ["10元", "月"],
        "reference": "副卡月功能费10元/月/张"
    },
    {
        "id": 5,
        "type": "单点查询",
        "question": "199元套餐可以加装多少M的宽带？",
        "expected_keywords": ["300M", "1000M"],
        "reference": "199元套餐支持300M或1000M宽带"
    },
    {
        "id": 6,
        "type": "单点查询",
        "question": "星卡39元套餐的定向流量包含哪些应用？",
        "expected_keywords": ["爱奇艺", "腾讯", "优酷", "抖音", "快手"],
        "reference": "爱奇艺、腾讯视频、优酷、西瓜视频、抖音、快手、网易云音乐等"
    },

    # ═══════════════════════════════════════════════════
    # 对比型（4题）- 多文档聚合 + 结构化对比
    # ═══════════════════════════════════════════════════
    {
        "id": 7,
        "type": "对比型",
        "question": "59元套餐和99元套餐有什么区别？",
        "expected_keywords": ["流量", "通话", "宽带"],
        "reference": "应从月租、流量、通话、宽带支持等方面对比"
    },
    {
        "id": 8,
        "type": "对比型",
        "question": "全额预存和橙分期两种方案有什么不同？",
        "expected_keywords": ["实付", "补贴", "互斥"],
        "reference": "全额预存实付更低无补贴，橙分期原价有购机补贴，两种互斥"
    },
    {
        "id": 9,
        "type": "对比型",
        "question": "哪个套餐支持1000M宽带？",
        "expected_keywords": ["199", "299"],
        "reference": "199元和299元套餐支持1000M宽带"
    },
    {
        "id": 10,
        "type": "对比型",
        "question": "129元和199元套餐的副卡数量和流量有什么区别？",
        "expected_keywords": ["4张", "110GB", "160GB"],
        "reference": "都是4张副卡，129最高110GB，199最高160GB"
    },

    # ═══════════════════════════════════════════════════
    # 多跳推理（5题）- 多步检索 + 计算推理
    # ═══════════════════════════════════════════════════
    {
        "id": 11,
        "type": "多跳推理",
        "question": "59元套餐办两张副卡，每月一共要交多少钱？",
        "expected_keywords": ["79元", "59", "10", "2"],
        "reference": "59 + 10×2 = 79元"
    },
    {
        "id": 12,
        "type": "多跳推理",
        "question": "129元套餐全额预存，一年的总费用是多少？",
        "expected_keywords": ["1068", "89", "12"],
        "reference": "89×12 = 1068元"
    },
    {
        "id": 13,
        "type": "多跳推理",
        "question": "59元套餐办两张副卡，最多能有多少流量和通话？",
        "expected_keywords": ["60GB", "600分钟"],
        "reference": "流量50-60GB，通话200+200×2=600分钟"
    },
    {
        "id": 14,
        "type": "多跳推理",
        "question": "129元套餐办4张副卡，每张副卡都办流量达人，最多能有多少流量？",
        "expected_keywords": ["110", "20GB"],
        "reference": "主卡90GB + 副卡增加20GB = 110GB"
    },
    {
        "id": 15,
        "type": "多跳推理",
        "question": "橙分期办199元套餐36个月，总补贴是多少？平均每月补贴多少？",
        "expected_keywords": ["2160", "60"],
        "reference": "补贴2160元，平均每月60元"
    },

    # ═══════════════════════════════════════════════════
    # 流程型（4题）- 流程提取 + 步骤组织
    # ═══════════════════════════════════════════════════
    {
        "id": 16,
        "type": "流程型",
        "question": "移动号码转网到电信需要走什么流程？",
        "expected_keywords": ["解约", "发送", "10086"],
        "reference": "解除合约→发送SQXZ#姓名#身份证到10086"
    },
    {
        "id": 17,
        "type": "流程型",
        "question": "在CRM系统里怎么创建意向单？",
        "expected_keywords": ["菜单", "客户信息", "保存"],
        "reference": "点击菜单→创建意向单→填写信息→保存"
    },
    {
        "id": 18,
        "type": "流程型",
        "question": "副卡转网需要怎么操作？",
        "expected_keywords": ["fkqr", "zhqr", "主卡同意"],
        "reference": "副卡发fkqr到10086，主卡发zhqr#副卡号到10086"
    },
    {
        "id": 19,
        "type": "流程型",
        "question": "用户想降低移动套餐，怎么通过投诉渠道办理？",
        "expected_keywords": ["中国移动APP", "我的投诉", "业务变更"],
        "reference": "打开APP→我的→我的投诉→业务变更及退订→套餐变更"
    },

    # ═══════════════════════════════════════════════════
    # 场景型（5题）- 意图理解 + 推荐能力
    # ═══════════════════════════════════════════════════
    {
        "id": 20,
        "type": "场景型",
        "question": "我是学生，预算有限，有什么套餐推荐？",
        "expected_keywords": ["39", "学生", "星卡"],
        "reference": "推荐星卡39元套餐，仅限在校学生开通"
    },
    {
        "id": 21,
        "type": "场景型",
        "question": "家里有老人和小孩，想办个全家用的套餐，怎么选？",
        "expected_keywords": ["副卡", "共享", "129"],
        "reference": "推荐129元套餐，可办4张副卡，全家共享流量通话"
    },
    {
        "id": 22,
        "type": "场景型",
        "question": "我每个月流量用得很多，经常看视频，哪个套餐合适？",
        "expected_keywords": ["199", "299", "大流量"],
        "reference": "推荐199或299元套餐，流量大"
    },
    {
        "id": 23,
        "type": "场景型",
        "question": "我想买个新手机，有什么购机优惠？",
        "expected_keywords": ["橙分期", "补贴", "合约"],
        "reference": "推荐橙分期方案，有购机补贴"
    },
    {
        "id": 24,
        "type": "场景型",
        "question": "我家住城中村，想装宽带，有什么选择？",
        "expected_keywords": ["300M", "9.9", "129"],
        "reference": "129及以上套餐可加装300M宽带，月费9.9元"
    },

    # ═══════════════════════════════════════════════════
    # 边界/异常（6题）- 拒答能力 + 异常处理
    # ═══════════════════════════════════════════════════
    {
        "id": 25,
        "type": "边界/异常",
        "question": "29元套餐可以办副卡吗？",
        "expected_keywords": ["不能", "不支持", "29"],
        "reference": "29元套餐不支持办理副卡"
    },
    {
        "id": 26,
        "type": "边界/异常",
        "question": "59元套餐可以装宽带吗？",
        "expected_keywords": ["不能", "不支持"],
        "reference": "59元套餐不支持宽带"
    },
    {
        "id": 27,
        "type": "边界/异常",
        "question": "电信的5G套餐多少钱一个月？",
        "expected_keywords": ["多种", "29", "299"],
        "reference": "应列出多个档位，不能只回答一个价格"
    },
    {
        "id": 28,
        "type": "边界/异常",
        "question": "我想办一个联通的套餐，有什么推荐？",
        "expected_keywords": ["电信", "联通"],
        "reference": "应说明本系统是电信套餐，无法推荐联通套餐"
    },
    {
        "id": 29,
        "type": "边界/异常",
        "question": "今天天气怎么样？",
        "expected_keywords": ["套餐", "天气"],
        "reference": "应说明无法回答天气问题，引导回套餐咨询"
    },
    {
        "id": 30,
        "type": "边界/异常",
        "question": "129元套餐副卡每张增加多少流量？",
        "expected_keywords": ["10GB", "20GB", "最多"],
        "reference": "每张副卡增加10GB，最多增加20GB（前2张有效）"
    },
]


def run_test(agent_func=None, verbose=True):
    """运行测试，支持延时统计"""
    if agent_func is None:
        from dx_agent import run_single
        agent_func = lambda q: run_single(q, verbose=False)["answer"]

    results = {"total": 0, "passed": 0, "failed": 0, "by_type": {}, "latency": []}

    for tc in TEST_CASES:
        q_type = tc["type"]
        if q_type not in results["by_type"]:
            results["by_type"][q_type] = {"total": 0, "passed": 0, "failed": 0, "latency": []}

        results["total"] += 1
        results["by_type"][q_type]["total"] += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{tc['id']}] [{q_type}] {tc['question']}")
            print(f"{'='*60}")

        try:
            start_time = time.time()
            answer = agent_func(tc["question"])
            elapsed = time.time() - start_time

            # 记录延时
            results["latency"].append(elapsed)
            results["by_type"][q_type]["latency"].append(elapsed)

            # 检查关键词命中
            hits = sum(1 for kw in tc["expected_keywords"] if kw in answer)
            hit_rate = hits / len(tc["expected_keywords"])
            passed = hit_rate >= 0.5  # 命中50%以上关键词算通过

            if passed:
                results["passed"] += 1
                results["by_type"][q_type]["passed"] += 1
            else:
                results["failed"] += 1
                results["by_type"][q_type]["failed"] += 1

            if verbose:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status} (命中 {hits}/{len(tc['expected_keywords'])} 关键词) ⏱ {elapsed:.2f}s")
                print(f"回答: {answer[:200]}...")

        except Exception as e:
            results["failed"] += 1
            results["by_type"][q_type]["failed"] += 1
            if verbose:
                print(f"❌ ERROR: {e}")

        # 题间延时，避免API限流
        if tc["id"] < len(TEST_CASES):
            if verbose:
                print(f"⏳ 等待20秒后继续下一题...")
            time.sleep(20)

    # 打印统计
    print(f"\n{'='*60}")
    print(f"测试结果统计")
    print(f"{'='*60}")
    print(f"总计: {results['total']} | 通过: {results['passed']} | 失败: {results['failed']}")
    print(f"通过率: {results['passed']/results['total']*100:.1f}%")

    # 延时统计
    latencies = results["latency"]
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        total_lat = sum(latencies)
        print(f"\n延时统计 (共 {len(latencies)} 题):")
        print(f"  总耗时: {total_lat:.2f}s")
        print(f"  平均延时: {avg_lat:.2f}s")
        print(f"  最快: {min_lat:.2f}s")
        print(f"  最慢: {max_lat:.2f}s")

    print(f"\n按题型统计:")
    for q_type, stats in results["by_type"].items():
        rate = stats["passed"]/stats["total"]*100 if stats["total"] > 0 else 0
        type_lat = stats["latency"]
        if type_lat:
            type_avg = sum(type_lat) / len(type_lat)
            print(f"  {q_type}: {stats['passed']}/{stats['total']} ({rate:.1f}%) | 平均 {type_avg:.2f}s")
        else:
            print(f"  {q_type}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    return results


if __name__ == "__main__":
    run_test()
