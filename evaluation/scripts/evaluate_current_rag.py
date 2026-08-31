#!/usr/bin/env python3
"""
RAG评测脚本 - 评估当前运行在端口8000的RAG系统
"""

import json
import time
import os
import requests

# 配置
API_URL = "http://localhost:8001/query"  # 改为8001端口
TIMEOUT_SECONDS = 120  # 2分钟超时（增加到120秒）
RETRY_ATTEMPTS = 2     # 重试次数
SLEEP_BETWEEN = 3      # 问题间隔

# 所有问题（与rag_eval_v2.py相同）
ALL_QUESTIONS = [
    (1, "单点问题", "5G畅享29套餐内容都有什么"),
    (2, "单点问题", "畅享59套餐有包含多少通话分钟数"),
    (3, "单点问题", "畅享99套餐包含多少流量"),
    (4, "单点问题", "畅享129套餐安装宽带多少钱"),
    (5, "单点问题", "哪个套餐可以安装千兆宽带"),
    (6, "套餐对比", "畅享99和畅享129的流量对比"),
    (7, "套餐对比", "5G畅享29和畅享59的流量对比"),
    (8, "套餐对比", "畅享99和畅享199的通话时间对比"),
    (9, "套餐对比", "畅享129和畅享199的宽带对比"),
    (10, "权益规则", "畅享99套餐橙分期可以开什么内容"),
    (11, "权益规则", "全额预存需要什么条件才可以办理"),
    (12, "权益规则", "199的橙分期还能办理预存吗"),
    (13, "模糊问题", "一家三口可以用哪些套餐"),
    (14, "模糊问题", "哪个套餐可以开通两张副卡"),
    (15, "模糊问题", "有没有100G+1000分钟+宽带千兆的套餐"),
    (16, "边界问题", "59套餐可以装宽带吗"),
    (17, "边界问题", "哪个套餐可以有免费短信"),
    (18, "边界问题", "预算150以内哪个套餐流量多"),
    (19, "多轮问题", "畅享99套餐可以安装300M的流量吗"),
    (20, "多轮问题", "通话可以折算成流量吗"),
]

def call_rag_api(question):
    """调用RAG API获取回答"""
    try:
        payload = {"question": question}
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", ""), True
    except requests.exceptions.Timeout:
        return "TIMEOUT", False
    except Exception as e:
        return f"ERROR: {e}", False

def run_single_question(question_id, question):
    """运行单个问题，支持重试"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            print(f"    尝试 {attempt+1}/{RETRY_ATTEMPTS}...", end=" ", flush=True)
            answer, success = call_rag_api(question)
            
            # 检查是否有效回答
            if len(answer) > 50 and not answer.startswith(('ERROR', 'TIMEOUT')):
                return answer, True
            else:
                print("回答太短或格式错误")
                
        except Exception as e:
            print(f"错误: {e}")
        
        # 重试前等待
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(5)
    
    return "TIMEOUT", False

def main():
    print("🚀 RAG评测脚本 (当前系统)")
    print(f"⏱️  超时时间: {TIMEOUT_SECONDS}秒 ({TIMEOUT_SECONDS//60}分钟)")
    print(f"🔄 重试次数: {RETRY_ATTEMPTS}")
    print(f"📊 问题总数: {len(ALL_QUESTIONS)}")
    print(f"🌐 API地址: {API_URL}")
    print()
    
    # 加载已有结果
    results_file = '/root/langchain_rag_code/evaluation_results_current.json'
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"📁 已有 {len(results)} 个结果")
    else:
        results = []
    
    # 处理每个问题
    for qid, qtype, question in ALL_QUESTIONS:
        # 检查是否已完成且成功
        existing = next((r for r in results if r['id'] == qid), None)
        if existing and len(existing['answer']) > 50 and not existing['answer'].startswith(('ERROR', 'TIMEOUT')):
            print(f"[{qid:2d}/20] ✅ 已完成: {question[:30]}...")
            continue
        
        print(f"[{qid:2d}/20] 🔄 处理: {question}")
        
        answer, success = run_single_question(qid, question)
        
        # 更新结果
        entry = {
            "id": qid,
            "type": qtype,
            "question": question,
            "answer": answer
        }
        results = [r for r in results if r['id'] != qid]
        results.append(entry)
        results.sort(key=lambda x: x['id'])
        
        # 保存结果
        with open(results_file, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        if success:
            print(f"  ✅ 成功 ({len(answer)}字)")
        else:
            print(f"  ❌ 超时/失败")
        
        # 间隔
        time.sleep(SLEEP_BETWEEN)
    
    # 统计结果
    print("\n" + "="*50)
    print("📊 评测完成统计")
    print("="*50)
    
    success_count = 0
    timeout_count = 0
    
    for r in results:
        if len(r['answer']) > 50 and not r['answer'].startswith(('ERROR', 'TIMEOUT')):
            success_count += 1
        else:
            timeout_count += 1
    
    print(f"✅ 成功: {success_count}/20")
    print(f"❌ 超时: {timeout_count}/20")
    print(f"📈 成功率: {success_count/20*100:.1f}%")
    
    # 列出超时的问题
    if timeout_count > 0:
        print("\n❌ 超时问题列表:")
        for r in results:
            if r['answer'] == "TIMEOUT":
                print(f"  - [{r['id']:2d}] {r['question']}")
    
    print(f"\n💾 结果已保存到: {results_file}")
    print("🎉 所有问题处理完成！")

if __name__ == "__main__":
    main()