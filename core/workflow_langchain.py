"""
LangChain RAG工作流核心实现
支持文档加载、分割、向量化、检索和生成
"""

import os
import sys
from typing import List, Dict, Any
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# 延迟导入 langchain 组件（优化启动速度）
from langchain_core.documents import Document

from config import Config, extract_plan_tier
from cache_manager import (
    get_rerank_cache,
    make_rerank_cache_key
)
import requests
import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """基于 BM25 的关键词检索器"""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        # jieba 分词（预计算）
        self.tokenized_docs = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        # 查询分词缓存
        self._query_token_cache: Dict[str, List[str]] = {}
        logger.info(f"BM25 检索器初始化完成，文档数: {len(documents)}")

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        # 缓存查询分词结果（满时清空，避免 O(n) 逐条删除）
        if query not in self._query_token_cache:
            if len(self._query_token_cache) >= 1000:
                self._query_token_cache.clear()
            self._query_token_cache[query] = list(jieba.cut(query))

        tokenized_query = self._query_token_cache[query]
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


class RAGWorkflow:
    """RAG工作流类"""

    def __init__(self):
        logger.info("初始化RAG工作流...")
        Config.validate()

        self.embedding = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.retriever = None
        self._executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
        self.bm25_retriever: BM25Retriever | None = None

        # 延迟导入 langchain 组件
        self._langchain_imports_done = False

        self._init_embedding()
        self._init_llm()
        self._init_vectorstore()
        self._init_qa_chain()

        logger.info("RAG工作流初始化完成")

    def _ensure_langchain_imports(self):
        """延迟导入 langchain 组件（仅首次调用时执行）"""
        if not self._langchain_imports_done:
            global RecursiveCharacterTextSplitter, TextLoader, DirectoryLoader
            global Chroma, ChatOpenAI, RunnablePassthrough, StrOutputParser, PromptTemplate
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.document_loaders import TextLoader, DirectoryLoader
            from langchain_chroma import Chroma
            from langchain_openai import ChatOpenAI
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            self._langchain_imports_done = True
            logger.info("langchain 组件延迟导入完成")

    def _init_embedding(self):
        logger.info("初始化本地嵌入模型...")
        try:
            from local_embeddings import LocalBGEEmbeddings
            self.embedding = LocalBGEEmbeddings(model_name=Config.EMBED_MODEL)
            logger.info("本地嵌入模型加载成功")
        except ImportError as e:
            raise ImportError(f"本地嵌入模型不可用: {e}。请安装sentence-transformers库。")
        except Exception as e:
            raise RuntimeError(f"本地嵌入模型加载失败: {e}")

    def _init_llm(self):
        self._ensure_langchain_imports()
        logger.info(f"初始化LLM模型: {Config.LLM_MODEL}")
        api_key = Config.LLM_API_KEY or Config.SILICONFLOW_API_KEY
        api_base = Config.LLM_API_BASE or Config.SILICONFLOW_API_BASE
        logger.info(f"LLM API: {api_base}")

        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model_name=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            timeout=Config.LLM_TIMEOUT,  # LLM 调用超时
            max_retries=0  # 禁用内置重试，防止超时叠加
        )

    def _init_vectorstore(self):
        self._ensure_langchain_imports()
        abs_path = os.path.abspath(Config.VECTOR_DB_PATH)
        logger.info(f"初始化向量数据库: {abs_path}")

        if not os.path.exists(abs_path):
            logger.info("向量库不存在，请先运行 build_vectors.py 构建")
            self.vectorstore = None
            return

        try:
            self.vectorstore = Chroma(
                persist_directory=abs_path,
                embedding_function=self.embedding,
                collection_name=Config.COLLECTION_NAME
            )
            count = self.vectorstore._collection.count()
            logger.info(f"向量库加载成功，文档块数量: {count}")

            # 从向量库提取所有文档，构建 BM25 索引
            if count > 0:
                all_data = self.vectorstore._collection.get(include=["documents", "metadatas"])
                all_docs = [
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in zip(all_data["documents"], all_data["metadatas"])
                ]
                self.bm25_retriever = BM25Retriever(all_docs)
        except Exception as e:
            logger.error(f"向量库加载失败: {e}")
            self.vectorstore = None

    def _init_qa_chain(self):
        self._ensure_langchain_imports()
        if not self.vectorstore:
            logger.warning("向量库未初始化，跳过QA链初始化")
            return

        logger.info("初始化检索问答链...")
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVAL_K}
        )

        template = """你是一个专业的运营商套餐助手。请严格基于提供的上下文信息回答。

【重要规则】
1. 只能基于提供的上下文信息回答，不要编造信息
2. 如果上下文中有相关信息，请直接回答
3. 如果上下文中没有精确匹配，但有相关套餐信息，可以推荐相关套餐作为参考
4. 如果用户提到"wifi"或"无线"，应主动询问是否需要了解宽带服务
5. 如果用户问"多少兆"或"几M"，应主动询问是否需要了解宽带速率
6. 如果用户问"多少钱"或"资费"，应主动给出月费信息
7. 回答要简洁（100-200字），友好且专业，使用中文回答
8. 如果上下文信息为空或不相关，根据用户问题关键词推测意图并引导：
   - 提到wifi/无线 → "目前暂无该套餐的宽带信息，您是否想了解我们的宽带融合套餐？例如99元套餐含100M宽带。"
   - 提到兆/M/速率 → "您是否想了解各档位套餐的宽带速率？99元含100M，129元含300M，299元可选1000M。"
   - 提到流量/通话 → "您是否想了解5G畅享29-199元系列套餐？各档位流量和通话时长不同。"
9. 【关键】上下文中可能包含多个套餐档位的数据表格。你必须只提取与用户问题中指定的套餐档位（如99元/59元/129元等）完全匹配的行数据，绝不能把其他档位的数据套用到用户询问的套餐上。

【示例1】
用户问题："畅享99套餐包含多少流量"
上下文信息："畅享99套餐包含20GB国内通用流量"
正确回答："畅享99元套餐包含20GB国内通用流量。"

【示例2】
用户问题："这个套餐wifi多少兆"
上下文信息："（空）"
正确回答："您是否想了解我们的宽带融合套餐？99元套餐含100M宽带，129元套餐含300M宽带。"

【上下文信息】
{context}

【用户问题】
{question}

【回答】"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("QA链初始化完成")

    def _is_plan_level_file(self, source: str) -> bool:
        """判断文件是否使用套餐级别切分"""
        filename = os.path.basename(source)
        return any(kw in filename for kw in Config.PLAN_LEVEL_CHUNK_FILES)

    def _split_by_plan_level(self, doc) -> list:
        """套餐级别切分：用'一、销售品内容'作为分隔符，确保每个套餐的完整信息在一个chunk里，同时合并优惠政策"""
        import re
        content = doc.page_content
        metadata = dict(doc.metadata)

        # 用正则按套餐分隔
        sections = re.split(Config.PLAN_LEVEL_CHUNK_SEPARATOR, content)
        sections = [s.strip() for s in sections if s.strip()]

        # 第一步：提取所有优惠政策
        all_discounts = self._extract_discounts(sections)

        # 第二步：为每个套餐chunk添加优惠政策
        chunks = []
        for i, section in enumerate(sections):
            # 跳过太短的段落
            if len(section) < 50:
                continue

            # 提取套餐档位
            plan_tier = self._extract_plan_tier(section)

            # 如果是大段落，按子标题二次切分
            if len(section) > Config.PLAN_LEVEL_MAX_CHUNK_SIZE:
                subsections = self._split_by_subtitles(section)
                for j, sub in enumerate(subsections):
                    # 检查是否包含副卡信息
                    if '副卡' in sub and plan_tier:
                        # 添加优惠政策
                        for discount in all_discounts:
                            if plan_tier in discount['applicable_plans']:
                                sub = sub + "\n\n【优惠政策】\n" + discount['summary']
                                break
                    chunks.append(Document(
                        page_content=sub,
                        metadata={**metadata, 'plan_section': i, 'sub_section': j}
                    ))
            else:
                # 检查是否包含副卡信息
                if '副卡' in section and plan_tier:
                    # 添加优惠政策
                    for discount in all_discounts:
                        if plan_tier in discount['applicable_plans']:
                            section = section + "\n\n【优惠政策】\n" + discount['summary']
                            break
                chunks.append(Document(
                    page_content=section,
                    metadata={**metadata, 'plan_section': i}
                ))

        return chunks

    def _extract_discounts(self, sections: list) -> list:
        """从段落列表中提取优惠政策"""
        import re
        discounts = []

        for section in sections:
            # 查找副卡减免优惠
            if '副卡功能费减免' in section or '这些套餐才可以做副卡减免' in section:
                # 提取适用套餐
                match = re.search(r'(\d+)元.*?(\d+)元.*?这些套餐才可以做副卡减免', section)
                if match:
                    # 提取所有套餐档位
                    plans = re.findall(r'(\d+)元', section[:5000])
                    # 过滤掉非套餐档位
                    valid_plans = [p for p in plans if int(p) in [59, 79, 99, 129, 169, 199, 229, 299]]

                    # 提取优惠摘要
                    summary_lines = []
                    if '副卡功能费减免至0元' in section:
                        summary_lines.append('副卡功能费可减免至0元')
                    if '129及以上套餐可以开通2张副卡以上' in section:
                        summary_lines.append('129元及以上套餐可开通2张以上副卡')
                    if '99元包括以下套餐，只能开通2张副卡' in section:
                        summary_lines.append('99元及以下套餐只能开通2张副卡')

                    if summary_lines and valid_plans:
                        discounts.append({
                            'applicable_plans': valid_plans,
                            'summary': '；'.join(summary_lines)
                        })

        return discounts

    def _extract_plan_tier(self, section: str) -> str:
        """从套餐段落中提取档位"""
        import re
        match = re.search(r'月基本费[：:](\d+)元', section)
        if match:
            return match.group(1)
        match = re.search(r'档位[：:](\d+)元', section)
        if match:
            return match.group(1)
        return None

    def _split_by_subtitles(self, section: str) -> list:
        """按子标题切分大段落"""
        import re
        # 按"（一）"、"（二）"等子标题切分
        subsections = re.split(r"(?=（[一二三四五六七八九十]）)", section)
        subsections = [s.strip() for s in subsections if s.strip()]

        # 合并小子标题到前一个chunk
        merged = []
        current = ""
        for sub in subsections:
            if len(current) + len(sub) > Config.PLAN_LEVEL_MAX_CHUNK_SIZE and current:
                merged.append(current)
                current = sub
            else:
                current = current + "\n" + sub if current else sub
        if current:
            merged.append(current)

        return merged


    def load_documents(self, data_dir: str = None) -> int:
        self._ensure_langchain_imports()
        if data_dir is None:
            data_dir = Config.DATA_DIR

        logger.info(f"从 {data_dir} 加载文档...")
        if not os.path.exists(data_dir):
            logger.warning(f"数据目录不存在: {data_dir}")
            return 0

        documents = []
        # 加载 .txt 和 .md 文件
        for pattern in ["**/*.txt", "**/*.md"]:
            loader = DirectoryLoader(
                data_dir,
                glob=pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            try:
                documents.extend(loader.load())
            except Exception as e:
                logger.warning(f"加载 {pattern} 文件失败: {e}")

        # 加载 .jsonl 文件（逐行解析为 Document）
        import json
        import glob as glob_mod
        for jsonl_path in glob_mod.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # 将 json 对象转为可读文本
                            if "q" in obj and "a" in obj:
                                text = f"问：{obj['q']}\n答：{obj['a']}"
                            else:
                                text = "\n".join(f"{k}：{v}" for k, v in obj.items() if not k.startswith("_"))
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": jsonl_path, "line": line_num, "source_type": "jsonl"}
                            ))
                        except json.JSONDecodeError:
                            logger.warning(f"JSONL 解析失败: {jsonl_path} 第{line_num}行")
                logger.info(f"加载 JSONL: {jsonl_path}")
            except Exception as e:
                logger.warning(f"加载 JSONL 失败: {jsonl_path}: {e}")

        # 加载 .csv 文件（首行为表头，每行转为可读文本）
        import csv as csv_mod
        for csv_path in glob_mod.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv_mod.DictReader(f)
                    for line_num, row in enumerate(reader, 2):  # 第2行起为数据
                        text = "，".join(f"{k}：{v}" for k, v in row.items() if v and v != "-")
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": csv_path, "line": line_num, "source_type": "csv"}
                        ))
                logger.info(f"加载 CSV: {csv_path}")
            except Exception as e:
                logger.warning(f"加载 CSV 失败: {csv_path}: {e}")

        if not documents:
            logger.warning("未找到文档文件")
            return 0

        logger.info(f"找到 {len(documents)} 个文档，开始分类切分...")

        # 按文件名分组，应用不同的 chunk 策略
        def _get_chunk_profile(source: str) -> str:
            filename = os.path.basename(source)
            for profile, keywords in Config.CHUNK_FILE_RULES.items():
                if any(kw in filename for kw in keywords):
                    return profile
            return "default"

        # JSONL/CSV 每行已是独立语义单元，无需二次切分
        atomic_docs = [doc for doc in documents if doc.metadata.get("source_type") in ("jsonl", "csv")]
        other_docs = [doc for doc in documents if doc.metadata.get("source_type") not in ("jsonl", "csv")]

        # 分离套餐级别文件和其他文件
        plan_level_docs = [doc for doc in other_docs if self._is_plan_level_file(doc.metadata.get("source", ""))]
        normal_docs = [doc for doc in other_docs if not self._is_plan_level_file(doc.metadata.get("source", ""))]

        texts = []

        # 套餐级别切分：用"一、销售品内容"作为分隔符
        if plan_level_docs:
            for doc in plan_level_docs:
                chunks = self._split_by_plan_level(doc)
                texts.extend(chunks)
            logger.info(f"  [plan_level] {len(plan_level_docs)} 个文档 → {len([c for c in texts if 'plan_section' in c.metadata])} 个 chunk")

        # 按文件名分组（仅对普通 txt/md 文件）
        grouped: dict[str, list[Document]] = {}
        for doc in normal_docs:
            profile = _get_chunk_profile(doc.metadata.get("source", ""))
            grouped.setdefault(profile, []).append(doc)

        for profile, docs in grouped.items():
            params = Config.CHUNK_PROFILES[profile]
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
                separators=Config.CHUNK_SEPARATORS
            )
            chunks = splitter.split_documents(docs)
            logger.info(f"  [{profile}] {len(docs)} 个文档 → {len(chunks)} 个 chunk (size={params['chunk_size']}, overlap={params['chunk_overlap']})")
            texts.extend(chunks)

        # 合并超短 chunk：低于 CHUNK_MIN_SIZE 的 chunk 与前一个 chunk 合并
        min_size = Config.CHUNK_MIN_SIZE
        merged_count = 0
        final_texts = []
        for chunk in texts:
            if final_texts and len(chunk.page_content) < min_size:
                final_texts[-1] = Document(
                    page_content=final_texts[-1].page_content + "\n" + chunk.page_content,
                    metadata=final_texts[-1].metadata
                )
                merged_count += 1
            else:
                final_texts.append(chunk)
        if merged_count:
            logger.info(f"  合并超短 chunk: {merged_count} 个 (< {min_size} 字)")
        texts = final_texts

        # JSONL/CSV 文档直接加入，不经过切分器
        if atomic_docs:
            logger.info(f"  [atomic] {len(atomic_docs)} 个文档直接加入（跳过切分）")
            texts.extend(atomic_docs)

        logger.info(f"切分完成，共 {len(texts)} 个文本块")

        if self.vectorstore is None:
            logger.info("创建新向量库...")
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding,
                persist_directory=Config.VECTOR_DB_PATH,
                collection_name=Config.COLLECTION_NAME
            )
        else:
            logger.info("添加到现有向量库...")
            self.vectorstore.add_documents(texts)

        logger.info(f"向量库已保存到: {Config.VECTOR_DB_PATH}")
        self._init_qa_chain()
        return len(texts)

    def _is_comparison_question(self, question: str) -> bool:
        comparison_patterns = [
            r'和.*对比', r'与.*对比', r'对比', r'相比',
            r'哪个更', r'哪个.*多', r'哪个.*少',
            r'多少.*对比', r'区别', r'差异', r'比较', r'高低', r'大小'
        ]
        return any(re.search(p, question) for p in comparison_patterns)

    def _extract_comparison_entities(self, question: str) -> List[str]:
        package_patterns = [
            r'(?:5G)?畅享\d+元?(?:套餐)?',
            r'星卡\d+(?:套餐)?',
            r'\d+G畅享\d+(?:套餐)?',
        ]

        entities = []
        for pattern in package_patterns:
            for match in re.findall(pattern, question):
                clean = match.strip()
                if clean and clean not in entities:
                    entities.append(clean)

        if not entities:
            # 匹配 "{num}元"、"{num}套餐"、以及已知档位数字（如"对比59和129套餐"中的"59"）
            for num in re.findall(r'(\d+)', question):
                if num in Config.PLAN_TIERS:
                    entity = f"{num}元"
                    if entity not in entities:
                        entities.append(entity)

        cleaned = []
        for e in entities:
            is_subsumed = any(other != e and e in other and len(other) > len(e) for other in entities)
            if not is_subsumed:
                cleaned.append(e)
        return cleaned

    def _handle_comparison_question(self, question: str, entities: List[str], conversation_history: str = "") -> Dict[str, Any]:
        logger.info(f"检测到对比问题，实体: {entities}")

        def normalize_entity(e: str) -> str:
            return re.sub(r'^5G', '', e).replace('元', '').replace('套餐', '').strip()

        def entity_matches_doc(entity: str, content: str) -> bool:
            name = normalize_entity(entity)
            if name in content:
                return True
            for num in re.findall(r'\d+', entity):
                if re.search(rf'(?<!\d){re.escape(num)}(?!\d)元', content):
                    return True
                if re.search(rf'(?<!\d){re.escape(num)}(?!\d)套餐', content):
                    return True
            return False

        def search_entity(entity):
            """优化：合并为单次综合查询，而非 4 次独立查询"""
            all_docs = []
            seen = set()
            # 从实体中提取套餐档位，用于元数据过滤
            entity_tier = None
            for num in re.findall(r'(\d+)', entity):
                if num in Config.PLAN_TIERS:
                    entity_tier = num
                    break
            # 合并为 1 次综合查询（覆盖流量、通话、宽带、优惠）
            combined_query = f"{entity} 月基本费 流量 语音 宽带 补贴 优惠"
            docs = self._hybrid_search(combined_query, vector_k=Config.VECTOR_TOP_K, bm25_k=Config.BM25_TOP_K, plan_tier=entity_tier)
            for d in docs:
                if d.page_content not in seen and entity_matches_doc(entity, d.page_content):
                    seen.add(d.page_content)
                    all_docs.append(d)
            return entity, all_docs

        def extract_key_fields(content: str) -> str:
            lines = content.split("\n")
            key_lines = [
                line.strip() for line in lines
                if any(kw in line.strip() for kw in Config.COMPARISON_KEYWORDS)
            ]
            return "\n".join(key_lines) if key_lines else content[:300]

        entity_docs = {}
        all_sources = []

        # 并行搜索各实体
        futures = {self._executor.submit(search_entity, e): e for e in entities}
        for future in as_completed(futures):
            entity, docs = future.result()
            if docs:
                entity_docs[entity] = docs
                all_sources.extend([extract_key_fields(doc.page_content) for doc in docs])

        seen_sources = set()
        deduped_sources = []
        for src in all_sources:
            if src not in seen_sources:
                seen_sources.add(src)
                deduped_sources.append(src)
        all_sources = deduped_sources[:30]  # 降低：50 → 30，减少上下文长度

        context_parts = []
        for entity, docs in entity_docs.items():
            entity_context = f"=== {entity} ===\n"
            key_content = "\n---\n".join([extract_key_fields(doc.page_content) for doc in docs[:10]])  # 降低：15 → 10
            entity_context += key_content
            context_parts.append(entity_context)

        comparison_context = "\n\n".join(context_parts)
        history_block = f"\n【对话历史】\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""你是一个专业的运营商套餐助手。请根据以下上下文信息，对用户询问的套餐进行对比分析。
{history_block}
【对比要求】
1. 首先根据用户问题，识别用户真正关心哪些维度（例如流量、通话、宽带、转网、副卡、月租、合约等），不要只局限于某几个固定维度
2. 从上下文中提取每个套餐档位在用户关心的维度上的具体数据，逐一对比
3. 如果某个维度的信息在上下文中不存在，请标注"暂无数据"
4. 【关键】上下文中可能包含多个套餐档位的数据，你必须只提取与正在对比的套餐档位完全匹配的数据行
5. 用表格形式呈现对比结果，表格列应包含：套餐档位、用户关心的各维度、以及简要建议
6. 如果信息足够，给出适合不同用户群体的建议

【上下文信息】
{comparison_context}

【用户问题】
{question}

【对比分析】"""

        try:
            answer = self.llm.invoke(prompt).content
            return {
                "answer": answer,
                "sources": all_sources[:10],
                "token_usage": {"total_tokens": 0, "total_cost": 0},
                "success": True
            }
        except Exception as e:
            logger.error(f"对比分析失败: {e}")
            return {"answer": f"查询失败：{str(e)}", "sources": [], "success": False}

    def _expand_query_intent(self, question: str) -> str:
        expanded = question
        for keyword, expansion in Config.INTENT_EXPANSIONS.items():
            if keyword in question:
                expanded += f" {expansion}"
                logger.info(f"意图识别: '{keyword}' -> 扩展关键词 '{expansion}'")
        return expanded

    def _rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []

        # 检查缓存
        rerank_cache = get_rerank_cache()
        cache_key = make_rerank_cache_key(query, [doc.page_content for doc in documents])
        cached_result = rerank_cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Rerank 缓存命中")
            return cached_result

        url = Config.RERANK_API_URL
        headers = {
            "Authorization": f"Bearer {Config.SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": Config.RERANK_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_k,
            "return_documents": False
        }

        for attempt in range(Config.RERANK_MAX_RETRIES):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=Config.RERANK_TIMEOUT)
                response.raise_for_status()
                result = response.json()

                reranked_docs = []
                for item in result.get("results", []):
                    index = item["index"]
                    reranked_docs.append(documents[index])
                result_docs = reranked_docs[:top_k]

                # 缓存结果
                rerank_cache.set(cache_key, result_docs)
                return result_docs
            except Exception as e:
                logger.warning(f"Rerank尝试 {attempt + 1}/{Config.RERANK_MAX_RETRIES} 失败: {e}")

        logger.warning("Rerank全部失败，返回原始文档")
        return documents[:top_k]

    def query(self, question: str, conversation_history: str = "") -> Dict[str, Any]:
        if not question or not question.strip():
            return {"answer": "请输入有效的问题", "sources": [], "success": False, "error": "empty_question"}

        if len(question) > Config.MAX_QUERY_LENGTH:
            return {"answer": f"问题长度不能超过{Config.MAX_QUERY_LENGTH}个字符", "sources": [], "success": False, "error": "query_too_long"}

        if not self.vectorstore:
            return {
                "answer": "向量库未初始化，请先运行 build_vectors.py 构建向量库",
                "sources": [], "success": False, "error": "vectorstore_not_initialized"
            }

        rewritten_question = re.sub(r'(?<!\w)(\d+)G(?!\w)', r'\1GB', question)
        rewritten_question = self._expand_query_intent(rewritten_question)
        for key, value in Config.QUERY_SYNONYMS.items():
            if key in rewritten_question and value not in rewritten_question:
                rewritten_question = f"{rewritten_question} {value}"

        price_match = re.search(r'(\d+)\s*元', question)
        if price_match:
            rewritten_question += f" {price_match.group(1)}元"

        if rewritten_question != question:
            logger.info(f"查询改写: '{question}' -> '{rewritten_question}'")
            question = rewritten_question

        logger.info(f"查询: {question[:50]}...")

        if self._is_comparison_question(question):
            entities = self._extract_comparison_entities(question)
            if len(entities) >= 2:
                logger.info(f"检测到对比问题: {entities}")
                return self._handle_comparison_question(question, entities, conversation_history)

        try:
            if Config.USE_RERANK:
                return self._query_with_rerank(question, conversation_history)
            else:
                return self._query_with_chain(question, conversation_history)
        except Exception as e:
            logger.error(f"查询失败: {str(e)}", exc_info=True)
            return {"answer": f"查询失败：{str(e)}", "sources": [], "success": False, "error": str(e)}

    def _hybrid_search(self, query: str, vector_k: int = None, bm25_k: int = None,
                       plan_tier: str = None) -> List[Document]:
        """混合检索：向量检索 + BM25 关键词检索，并行执行后合并去重。
        当指定 plan_tier 时，优先返回该套餐档位的文档。
        """
        vector_k = vector_k or Config.VECTOR_TOP_K
        bm25_k = bm25_k or Config.BM25_TOP_K

        # 真正并行执行向量检索和 BM25 检索
        vector_docs = []
        bm25_docs = []

        def _vector_search():
            try:
                docs = self.vectorstore.similarity_search(query, k=vector_k)
                logger.info(f"向量检索命中: {len(docs)} 个 chunk")
                return docs
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")
                return []

        def _bm25_search():
            try:
                if self.bm25_retriever:
                    docs = self.bm25_retriever.search(query, top_k=bm25_k)
                    logger.info(f"BM25 检索命中: {len(docs)} 个 chunk")
                    return docs
                return []
            except Exception as e:
                logger.warning(f"BM25 检索失败: {e}")
                return []

        # 使用独立线程池并行执行，避免嵌套死锁
        with ThreadPoolExecutor(max_workers=2) as search_executor:
            vector_future = search_executor.submit(_vector_search)
            bm25_future = search_executor.submit(_bm25_search)
            vector_docs = vector_future.result()
            bm25_docs = bm25_future.result()

        # 合并去重（按 page_content 去重，保序）
        seen = set()
        merged = []
        for doc in vector_docs + bm25_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        # 套餐档位过滤：利用 plan_tier 元数据精确过滤
        if plan_tier:
            tier_docs = [d for d in merged if plan_tier in [t.strip() for t in d.metadata.get("plan_tier", "").split(",")]]
            if tier_docs:
                logger.info(f"套餐档位过滤: '{plan_tier}元' 从 {len(merged)} 个文档过滤到 {len(tier_docs)} 个")
                merged = tier_docs
            else:
                logger.warning(f"套餐档位过滤: '{plan_tier}元' 无匹配，保留原始 {len(merged)} 个文档")

        logger.info(f"检索完成: {len(merged)} 个 chunk（向量 {len(vector_docs)} + BM25 {len(bm25_docs)}）")
        return merged

    @staticmethod
    def _extract_price(question: str) -> str | None:
        """从查询中提取套餐价格，返回已知档位数字或None。"""
        return extract_plan_tier(question)

    def _query_with_rerank(self, question: str, conversation_history: str) -> Dict[str, Any]:
        # 提取查询中的套餐价格，用于精确过滤
        plan_tier = self._extract_price(question)

        # 混合检索：向量 + BM25，如有价格则按档位过滤
        docs = self._hybrid_search(question, plan_tier=plan_tier)
        if not docs:
            return {"answer": self._get_fallback_answer(question), "sources": [], "success": True}

        # Rerank 合并后的结果
        reranked_docs = self._rerank_documents(question, docs, top_k=Config.RETRIEVAL_K)

        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        history_block = f"\n【对话历史】\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""你是一个专业的运营商套餐助手。请严格根据以下上下文信息回答用户问题。
{history_block}
【重要规则】
1. 只能基于提供的上下文信息回答，不要编造信息
2. 【关键】上下文中可能包含多个套餐档位的数据表格。你必须只提取与用户问题中指定的套餐档位完全匹配的行数据
3. 如果上下文中没有相关信息，根据用户问题推测意图并引导
4. 回答要简洁（100-200字），直接给出关键信息
5. 使用中文回答

【上下文信息】
{context}

【用户问题】
{question}

【回答】"""

        try:
            answer = self.llm.invoke(prompt).content
            sources = [doc.page_content for doc in reranked_docs]
            return {
                "answer": answer, "sources": sources,
                "token_usage": {"total_tokens": 0, "total_cost": 0}, "success": True
            }
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {"answer": f"查询失败：{str(e)}", "sources": [], "success": False}

    def _query_with_chain(self, question: str, conversation_history: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"answer": "QA链未初始化", "sources": [], "success": False, "error": "qa_chain_not_initialized"}

        try:
            # 只检索一次，同时用于构建上下文和返回来源
            source_docs = self.retriever.invoke(question)
            sources = [doc.page_content for doc in source_docs]
            context = "\n\n".join(sources)

            history_block = f"\n【对话历史】\n{conversation_history}\n" if conversation_history else ""
            query_input = f"{history_block}\n【上下文】\n{context}\n\n【问题】\n{question}"

            answer = self.llm.invoke(query_input).content

            return {
                "answer": answer, "sources": sources,
                "token_usage": {"total_tokens": 0, "total_cost": 0}, "success": True
            }
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {"answer": f"查询失败：{str(e)}", "sources": [], "success": False}

    def _get_fallback_answer(self, question: str) -> str:
        if any(kw in question for kw in ["wifi", "WiFi", "无线"]):
            return "您是否想了解宽带融合套餐？例如99元套餐含100M宽带，129元套餐含300M宽带。"
        if any(kw in question for kw in ["兆", " M", "速率"]):
            return "各档位宽带速率参考：99元含100M，129元含300M，299元可选1000M宽带。您想了解哪个套餐？"
        if any(kw in question for kw in ["流量", "不够", "叠加"]):
            return "您是否想了解5G畅享29-199元系列套餐？不同档位包含不同流量和通话时长。"
        return "抱歉，暂未找到相关信息，如需帮助请咨询人工客服"

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        if len(questions) > Config.MAX_BATCH_SIZE:
            logger.warning(f"批量查询数量超限: {len(questions)}")
            questions = questions[:Config.MAX_BATCH_SIZE]

        logger.info(f"开始批量查询 | 问题数: {len(questions)}")
        # 并行执行批量查询
        futures = {self._executor.submit(self.query, q): i for i, q in enumerate(questions)}
        results = [None] * len(questions)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"批量查询第 {idx+1} 个失败: {e}")
                results[idx] = {"answer": f"查询失败：{e}", "sources": [], "success": False}

        logger.info(f"批量查询完成 | 问题数: {len(questions)}")
        return results

    def get_stats(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"status": "vectorstore_not_initialized"}
        try:
            collection = self.vectorstore._collection
            return {
                "status": "ready",
                "document_count": collection.count(),
                "vector_db_path": Config.VECTOR_DB_PATH,
                "llm_model": Config.LLM_MODEL,
                "embedding_model": Config.EMBED_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_profiles": Config.CHUNK_PROFILES,
                "retrieval_k": Config.RETRIEVAL_K
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


def main():
    if len(sys.argv) < 2:
        print("用法: python workflow_langchain.py <查询问题>")
        print("示例: python workflow_langchain.py '有哪些手机套餐？'")
        return

    question = sys.argv[1]
    workflow = RAGWorkflow()
    result = workflow.query(question)

    print("\n" + "=" * 50)
    print(f"问题: {question}")
    print(f"成功: {result['success']}")
    if result['success']:
        print(f"回答: {result['answer']}")
        print(f"来源文档数量: {len(result['sources'])}")
    else:
        print(f"错误: {result.get('error', '未知错误')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
