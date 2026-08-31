#!/usr/bin/env python3
"""
simple_rag.py 测试脚本
"""

import sys
import os
import time
import logging

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_config():
    """测试配置"""
    print("\n" + "=" * 50)
    print("📋 测试 1: 配置加载")
    print("=" * 50)

    from simple_rag import Config

    try:
        Config.validate()
        print("✅ 配置验证通过")
        print(f"   LLM 模型: {Config.LLM_MODEL}")
        print(f"   Embedding 模型: {Config.EMBED_MODEL}")
        print(f"   向量库路径: {Config.VECTOR_DB_PATH}")
        print(f"   数据目录: {Config.DATA_DIR}")
        return True
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False


def test_text_splitter():
    """测试文本切分器"""
    print("\n" + "=" * 50)
    print("📋 测试 2: 文本切分器")
    print("=" * 50)

    from simple_rag import TextSplitter

    # 测试短文本
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    short_text = "这是一段短文本。"
    chunks = splitter.split_text(short_text)
    print(f"✅ 短文本切分: {len(chunks)} 块")

    # 测试长文本
    long_text = "第一段内容。" * 50
    chunks = splitter.split_text(long_text)
    print(f"✅ 长文本切分: 输入 {len(long_text)} 字 → 输出 {len(chunks)} 块")

    # 测试多段落文本
    multi_para = "段落一的内容在这里。\n\n段落二的内容在这里。\n\n段落三的内容在这里。"
    splitter2 = TextSplitter(chunk_size=50, chunk_overlap=10, separators=["\n\n", "\n", "。"])
    chunks = splitter2.split_text(multi_para)
    print(f"✅ 多段落切分: {len(chunks)} 块")
    for i, chunk in enumerate(chunks):
        print(f"   块{i}: {chunk[:30]}...")

    return True


def test_document_processor():
    """测试文档处理器"""
    print("\n" + "=" * 50)
    print("📋 测试 3: 文档加载与处理")
    print("=" * 50)

    from simple_rag import DocumentProcessor, Config

    processor = DocumentProcessor()

    # 加载文档
    documents = processor.load_documents(Config.DATA_DIR)
    print(f"✅ 加载了 {len(documents)} 个文档")

    if not documents:
        print("⚠️  没有找到文档，跳过切分测试")
        return True

    # 显示文档类型统计
    type_count = {}
    for doc in documents:
        doc_type = doc.metadata.get("type", "unknown")
        type_count[doc_type] = type_count.get(doc_type, 0) + 1
    print(f"   文档类型: {type_count}")

    # 切分文档
    chunks = processor.split_documents(documents)
    print(f"✅ 切分完成: {len(documents)} 个文档 → {len(chunks)} 个文本块")

    return True


def test_embedding_model():
    """测试嵌入模型"""
    print("\n" + "=" * 50)
    print("📋 测试 4: 嵌入模型")
    print("=" * 50)

    from simple_rag import EmbeddingModel

    try:
        model = EmbeddingModel()

        # 测试单条嵌入
        start = time.time()
        embedding = model.embed_query("测试查询")
        elapsed = time.time() - start
        print(f"✅ 单条嵌入成功: 维度={len(embedding)}, 耗时={elapsed:.2f}s")

        # 测试批量嵌入
        texts = ["测试文本1", "测试文本2", "测试文本3"]
        start = time.time()
        embeddings = model.embed_documents(texts)
        elapsed = time.time() - start
        print(f"✅ 批量嵌入成功: {len(embeddings)} 条, 耗时={elapsed:.2f}s")

        # 测试缓存
        start = time.time()
        embedding2 = model.embed_query("测试查询")  # 应该命中缓存
        elapsed = time.time() - start
        print(f"✅ 缓存测试: 耗时={elapsed:.4f}s (应该很快)")

        return True
    except Exception as e:
        print(f"❌ 嵌入模型测试失败: {e}")
        return False


def test_vector_store():
    """测试向量存储"""
    print("\n" + "=" * 50)
    print("📋 测试 5: 向量存储 (Chroma)")
    print("=" * 50)

    from simple_rag import VectorStore, EmbeddingModel, Document

    try:
        embedding_model = EmbeddingModel()
        store = VectorStore(embedding_model)

        # 创建测试集合
        test_collection = "test_collection"
        store.create_or_load(test_collection)
        print(f"✅ 集合创建成功")

        # 添加测试文档
        test_docs = [
            Document(page_content="59元套餐包含20GB流量", metadata={"plan": "59"}),
            Document(page_content="99元套餐包含40GB流量", metadata={"plan": "99"}),
            Document(page_content="129元套餐包含60GB流量", metadata={"plan": "129"}),
        ]
        store.add_documents(test_docs)
        print(f"✅ 添加了 {len(test_docs)} 个文档")

        # 检索测试
        results = store.search("59元套餐流量", top_k=2)
        print(f"✅ 检索成功: 返回 {len(results)} 个结果")
        for i, doc in enumerate(results):
            print(f"   结果{i}: {doc.page_content[:30]}...")

        # 清理测试集合
        store.client.delete_collection(test_collection)
        print(f"✅ 测试集合已清理")

        return True
    except Exception as e:
        print(f"❌ 向量存储测试失败: {e}")
        return False


def test_bm25_retriever():
    """测试 BM25 检索器"""
    print("\n" + "=" * 50)
    print("📋 测试 6: BM25 检索器")
    print("=" * 50)

    from simple_rag import BM25Retriever, Document

    try:
        # 创建测试文档
        docs = [
            Document(page_content="59元套餐包含20GB国内通用流量", metadata={}),
            Document(page_content="99元套餐包含40GB国内通用流量和100分钟通话", metadata={}),
            Document(page_content="129元套餐包含60GB流量和200分钟通话", metadata={}),
            Document(page_content="宽带套餐需要额外办理", metadata={}),
        ]

        retriever = BM25Retriever(docs)
        print(f"✅ BM25 初始化成功，文档数: {len(docs)}")

        # 检索测试
        results = retriever.search("59元套餐流量", top_k=2)
        print(f"✅ 检索成功: 返回 {len(results)} 个结果")
        for i, doc in enumerate(results):
            print(f"   结果{i}: {doc.page_content}")

        # 测试缓存
        results2 = retriever.search("59元套餐流量", top_k=2)
        print(f"✅ 缓存测试: 二次检索成功")

        return True
    except Exception as e:
        print(f"❌ BM25 测试失败: {e}")
        return False


def test_reranker():
    """测试 Rerank 重排序"""
    print("\n" + "=" * 50)
    print("📋 测试 7: Rerank 重排序")
    print("=" * 50)

    from simple_rag import Reranker, Document

    try:
        reranker = Reranker()

        # 创建测试文档
        docs = [
            Document(page_content="59元套餐包含20GB国内通用流量", metadata={}),
            Document(page_content="99元套餐包含40GB国内通用流量和100分钟通话", metadata={}),
            Document(page_content="129元套餐包含60GB流量和200分钟通话", metadata={}),
        ]

        # 重排序测试
        query = "59元套餐包含多少流量"
        start = time.time()
        reranked = reranker.rerank(query, docs, top_k=3)
        elapsed = time.time() - start
        print(f"✅ Rerank 成功: 返回 {len(reranked)} 个结果, 耗时={elapsed:.2f}s")
        for i, doc in enumerate(reranked):
            print(f"   结果{i}: {doc.page_content[:30]}...")

        # 测试缓存
        start = time.time()
        reranked2 = reranker.rerank(query, docs, top_k=3)
        elapsed = time.time() - start
        print(f"✅ 缓存测试: 耗时={elapsed:.4f}s (应该很快)")

        return True
    except Exception as e:
        print(f"❌ Rerank 测试失败: {e}")
        return False


def test_llm_generator():
    """测试 LLM 生成器"""
    print("\n" + "=" * 50)
    print("📋 测试 8: LLM 生成器")
    print("=" * 50)

    from simple_rag import LLMGenerator

    try:
        llm = LLMGenerator()

        # 测试生成
        query = "59元套餐包含多少流量"
        context = "59元套餐包含20GB国内通用流量和100分钟通话"
        history = ""

        start = time.time()
        answer = llm.generate(query, context, history)
        elapsed = time.time() - start

        print(f"✅ LLM 生成成功，耗时={elapsed:.2f}s")
        print(f"   问题: {query}")
        print(f"   上下文: {context[:50]}...")
        print(f"   回答: {answer[:100]}...")

        return True
    except Exception as e:
        print(f"❌ LLM 测试失败: {e}")
        return False


def test_hybrid_retriever():
    """测试混合检索器"""
    print("\n" + "=" * 50)
    print("📋 测试 9: 混合检索器")
    print("=" * 50)

    from simple_rag import HybridRetriever, VectorStore, BM25Retriever, Reranker, EmbeddingModel, Document

    try:
        # 创建测试数据
        docs = [
            Document(page_content="59元套餐包含20GB国内通用流量", metadata={"plan": "59"}),
            Document(page_content="99元套餐包含40GB国内通用流量和100分钟通话", metadata={"plan": "99"}),
            Document(page_content="129元套餐包含60GB流量和200分钟通话", metadata={"plan": "129"}),
        ]

        # 初始化组件
        embedding_model = EmbeddingModel()
        vector_store = VectorStore(embedding_model)
        vector_store.create_or_load("test_hybrid")
        vector_store.add_documents(docs)

        bm25_retriever = BM25Retriever(docs)
        reranker = Reranker()

        # 创建混合检索器
        hybrid = HybridRetriever(vector_store, bm25_retriever, reranker)
        print("✅ 混合检索器初始化成功")

        # 测试检索
        query = "59元套餐流量"
        start = time.time()
        results = hybrid.search(query, top_k=3)
        elapsed = time.time() - start

        print(f"✅ 混合检索成功: 返回 {len(results)} 个结果, 耗时={elapsed:.2f}s")
        for i, doc in enumerate(results):
            print(f"   结果{i}: {doc.page_content}")

        # 清理
        vector_store.client.delete_collection("test_hybrid")

        return True
    except Exception as e:
        print(f"❌ 混合检索测试失败: {e}")
        return False


def test_full_rag():
    """测试完整 RAG 流程"""
    print("\n" + "=" * 50)
    print("📋 测试 10: 完整 RAG 流程")
    print("=" * 50)

    from simple_rag import SimpleRAG, Config

    try:
        # 初始化 RAG
        rag = SimpleRAG()
        print("✅ RAG 系统初始化成功")

        # 检查知识库状态
        stats = rag.get_stats()
        print(f"   文档数量: {stats['document_count']}")

        # 如果知识库为空，加载知识库
        if stats['document_count'] == 0:
            print("   知识库为空，开始加载...")
            count = rag.load_knowledge_base(Config.DATA_DIR)
            print(f"   加载完成: {count} 个文本块")
            stats = rag.get_stats()

        # 测试查询
        test_queries = [
            "59元套餐包含多少流量",
            "99元套餐有宽带吗",
            "副卡怎么办理",
        ]

        for query in test_queries:
            print(f"\n   查询: {query}")
            start = time.time()
            result = rag.query(query)
            elapsed = time.time() - start
            print(f"   耗时: {elapsed:.2f}s")
            print(f"   成功: {result.success}")
            print(f"   回答: {result.answer[:80]}...")
            print(f"   来源: {len(result.sources)} 个")

        return True
    except Exception as e:
        print(f"❌ 完整 RAG 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始测试 simple_rag.py")
    print("=" * 60)

    tests = [
        ("配置加载", test_config),
        ("文本切分器", test_text_splitter),
        ("文档处理器", test_document_processor),
        ("嵌入模型", test_embedding_model),
        ("向量存储", test_vector_store),
        ("BM25 检索器", test_bm25_retriever),
        ("Rerank 重排序", test_reranker),
        ("LLM 生成器", test_llm_generator),
        ("混合检索器", test_hybrid_retriever),
        ("完整 RAG 流程", test_full_rag),
    ]

    results = []
    start_time = time.time()

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
            results.append((name, False))

    elapsed = time.time() - start_time

    # 打印测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    print(f"\n总计: {passed} 通过, {failed} 失败")
    print(f"总耗时: {elapsed:.2f}s")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
