#!/usr/bin/env python3
"""诊断脚本：排查"对比99和129套餐"检索问题"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from simple_rag import SimpleRAG, Config, HybridRetriever, VectorStore, BM25Retriever, EmbeddingModel, Reranker

def main():
    query = "对比99和129套餐"

    print("=" * 60)
    print(f"诊断查询: {query}")
    print("=" * 60)

    # 初始化
    print("\n[1] 初始化 RAG 系统...")
    rag = SimpleRAG()

    # Step 1: 查询改写
    print("\n[2] 查询改写:")
    rewritten = rag._rewrite_query(query)
    print(f"  原始: {query}")
    print(f"  改写: {rewritten}")

    # Step 2: 意图分类
    intent = rag._classify_intent(rewritten)
    print(f"\n[3] 意图分类: {intent}")

    # Step 3: 查询扩展
    queries = rag._expand_queries(rewritten, intent)
    print(f"\n[4] 查询扩展 ({len(queries)} 个子查询):")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")

    # Step 4: 分词检查
    import jieba
    print(f"\n[5] jieba 分词检查:")
    for q in queries:
        tokens = list(jieba.cut(q))
        print(f"  '{q}' → {tokens}")

    # Step 5: 向量检索（单独）
    print(f"\n[6] 向量检索 (VECTOR_TOP_K={Config.VECTOR_TOP_K}):")
    for q in queries:
        docs = rag.vector_store.search(q, Config.VECTOR_TOP_K)
        print(f"\n  查询: '{q}' → {len(docs)} 个结果")
        for i, doc in enumerate(docs):
            content_preview = doc.page_content[:120].replace('\n', ' ')
            has_99 = '99' in doc.page_content
            has_流量 = '流量' in doc.page_content
            has_通话 = '语音' in doc.page_content or '通话' in doc.page_content or '分钟' in doc.page_content
            has_宽带 = '宽带' in doc.page_content
            tags = []
            if has_99: tags.append('99')
            if has_流量: tags.append('流量')
            if has_通话: tags.append('通话')
            if has_宽带: tags.append('宽带')
            print(f"    [{i}] [{'|'.join(tags)}] {content_preview}...")

    # Step 6: BM25 检索（单独）
    print(f"\n[7] BM25 检索 (BM25_TOP_K={Config.BM25_TOP_K}):")
    for q in queries:
        docs = rag.bm25_retriever.search(q, Config.BM25_TOP_K)
        print(f"\n  查询: '{q}' → {len(docs)} 个结果")
        for i, doc in enumerate(docs):
            content_preview = doc.page_content[:120].replace('\n', ' ')
            has_99 = '99' in doc.page_content
            has_流量 = '流量' in doc.page_content
            has_通话 = '语音' in doc.page_content or '通话' in doc.page_content or '分钟' in doc.page_content
            has_宽带 = '宽带' in doc.page_content
            tags = []
            if has_99: tags.append('99')
            if has_流量: tags.append('流量')
            if has_通话: tags.append('通话')
            if has_宽带: tags.append('宽带')
            print(f"    [{i}] [{'|'.join(tags)}] {content_preview}...")

    # Step 7: 混合检索 + Rerank（完整链路）
    print(f"\n[8] 完整混合检索 + Rerank (RETRIEVAL_K={Config.RETRIEVAL_K}):")
    if len(queries) > 1:
        docs = rag._multi_query_retrieve(queries)
    else:
        docs = rag.hybrid_retriever.search(rewritten, Config.RETRIEVAL_K)
    print(f"  最终返回: {len(docs)} 个文档")
    for i, doc in enumerate(docs):
        content_preview = doc.page_content[:150].replace('\n', ' ')
        has_99 = '99' in doc.page_content
        has_流量 = '流量' in doc.page_content
        has_通话 = '语音' in doc.page_content or '通话' in doc.page_content or '分钟' in doc.page_content
        has_宽带 = '宽带' in doc.page_content
        tags = []
        if has_99: tags.append('99')
        if has_流量: tags.append('流量')
        if has_通话: tags.append('通话')
        if has_宽带: tags.append('宽带')
        print(f"  [{i}] [{'|'.join(tags)}] {content_preview}...")

    # Step 8: 检查99套餐关键信息是否在最终结果中
    print(f"\n[9] 99套餐信息完整性检查:")
    all_text = "\n".join([d.page_content for d in docs])
    checks = {
        "99元/99套餐": "99" in all_text,
        "流量": "流量" in all_text and ("20G" in all_text or "20GB" in all_text),
        "通话/语音/分钟": any(kw in all_text for kw in ["400分钟", "语音", "通话"]),
        "宽带": "宽带" in all_text,
    }
    for k, v in checks.items():
        status = "✅" if v else "❌ 缺失"
        print(f"  {status} {k}")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
