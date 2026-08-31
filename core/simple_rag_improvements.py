#!/usr/bin/env python3
"""
simple_rag.py 改进示例
展示关键模块的优化实现
"""

import os
import re
import json
import time
import logging
import hashlib
import threading
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq

import chromadb
import jieba
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import requests

# 加载 .env 文件
load_dotenv()

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
)
logger = logging.getLogger(__name__)


# ========== 改进的数据结构 ==========

@dataclass
class Document:
    """文档块（改进版）"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = str(uuid.uuid4())


@dataclass
class QueryResult:
    """查询结果（改进版）"""
    answer: str
    sources: List[str]
    success: bool
    processing_time: float
    error_message: Optional[str] = None
    query_id: Optional[str] = None


# ========== 改进的配置类 ==========

class Config:
    """系统配置（改进版）"""
    
    # API 配置
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.stepfun.com/step_plan/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "step-3.7-flash")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")

    # Embedding 配置
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))
    EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "10000"))

    # Rerank 配置
    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANK_API_URL = os.getenv("RERANK_API_URL", "https://api.siliconflow.cn/v1/rerank")
    RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "30"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
    RERANK_CACHE_SIZE = int(os.getenv("RERANK_CACHE_SIZE", "1000"))

    # 向量库配置
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain_collection")

    # 切分配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
    CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "100"))
    CHUNK_SEPARATORS = ["\n\n", "###", "####", "#####", "\n", "。", "！", "？", "；", "，", " "]

    # 分类型切分配置
    CHUNK_PROFILES = {
        "small": {"chunk_size": 250, "chunk_overlap": 50},
        "large": {"chunk_size": 500, "chunk_overlap": 100},
        "default": {"chunk_size": 300, "chunk_overlap": 60},
    }
    CHUNK_FILE_RULES = {
        "small": ["Q&A", "qa_plans", "transfer_faq", "搭配，Q&A"],
        "large": ["套餐详情", "搭配表_重构", "plan_details", "套餐搭配表.csv"],
    }

    # 检索配置
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
    VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "15"))
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))

    # 数据目录
    DATA_DIR = os.getenv("DATA_DIR", "./data")

    # 查询配置
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
    QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", "1000"))

    @classmethod
    def validate(cls):
        """验证配置（改进版）"""
        errors = []
        
        # 验证API配置
        if not cls.LLM_API_KEY and not cls.SILICONFLOW_API_KEY:
            errors.append("请配置 LLM_API_KEY 或 SILICONFLOW_API_KEY")
        
        # 验证数值范围
        if cls.LLM_TEMPERATURE < 0 or cls.LLM_TEMPERATURE > 2:
            errors.append(f"LLM_TEMPERATURE 必须在 0-2 之间，当前值: {cls.LLM_TEMPERATURE}")
        
        if cls.LLM_MAX_TOKENS < 1 or cls.LLM_MAX_TOKENS > 4096:
            errors.append(f"LLM_MAX_TOKENS 必须在 1-4096 之间，当前值: {cls.LLM_MAX_TOKENS}")
        
        if cls.CHUNK_SIZE < 50 or cls.CHUNK_SIZE > 1000:
            errors.append(f"CHUNK_SIZE 必须在 50-1000 之间，当前值: {cls.CHUNK_SIZE}")
        
        if cls.CHUNK_OVERLAP < 0 or cls.CHUNK_OVERLAP > cls.CHUNK_SIZE:
            errors.append(f"CHUNK_OVERLAP 必须在 0-{cls.CHUNK_SIZE} 之间，当前值: {cls.CHUNK_OVERLAP}")
        
        # 创建必要目录
        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(errors))
        
        return True


# ========== 改进的文本切分器 ==========

class TextSplitter:
    """递归字符文本切分器（改进版）"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 60,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", "，", " "]

    def split_text(self, text: str) -> List[str]:
        """切分文本（改进版）"""
        if len(text) <= self.chunk_size:
            return [text]

        # 选择分隔符
        separator = self._choose_separator(text)
        splits = text.split(separator)

        # 合并小块
        chunks = []
        current_chunk = ""

        for split in splits:
            if not split.strip():
                continue

            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        # 添加重叠（改进版：在句子边界截断）
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _choose_separator(self, text: str) -> str:
        """选择分隔符（改进版：优先选择能保持语义完整的分隔符）"""
        # 优先选择段落分隔符
        for sep in ["\n\n", "###", "####"]:
            if sep in text:
                return sep
        
        # 其次选择句子分隔符
        for sep in ["。", "！", "？", "；"]:
            if sep in text:
                return sep
        
        # 最后使用默认分隔符
        return self.separators[-1]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠（改进版：智能在句子边界截断）"""
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            
            # 寻找句子边界
            overlap_end = len(prev)
            for sep in ["。", "！", "？", "；", "\n"]:
                idx = prev.rfind(sep, -self.chunk_overlap * 2)
                if idx != -1:
                    overlap_end = idx + 1
                    break
            
            # 计算重叠文本
            overlap_text = prev[-min(overlap_end, self.chunk_overlap):]
            result.append(overlap_text + chunks[i])
        
        return result


# ========== 改进的文档处理器 ==========

class DocumentProcessor:
    """文档处理器（改进版）"""

    def __init__(self):
        self.splitter = TextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=Config.CHUNK_SEPARATORS
        )

    def load_documents(self, data_dir: str = None) -> List[Document]:
        """加载文档（改进版：支持大小限制和错误恢复）"""
        data_dir = data_dir or Config.DATA_DIR
        documents = []

        if not os.path.exists(data_dir):
            logger.warning(f"数据目录不存在: {data_dir}")
            return documents

        # 文件大小限制
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

        for file_path in Path(data_dir).rglob("*"):
            if not file_path.is_file():
                continue

            # 检查文件大小
            if file_path.stat().st_size > MAX_FILE_SIZE:
                logger.warning(f"文件过大，跳过: {file_path}")
                continue

            try:
                # 根据文件后缀调用不同加载器
                if file_path.suffix in [".txt", ".md"]:
                    documents.extend(self._load_text_file(file_path))
                elif file_path.suffix == ".jsonl":
                    documents.extend(self._load_jsonl_file(file_path))
                elif file_path.suffix == ".csv":
                    documents.extend(self._load_csv_file(file_path))
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")
                continue

        logger.info(f"共加载 {len(documents)} 个文档")
        return documents

    def _load_text_file(self, file_path: Path) -> List[Document]:
        """加载文本文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(
                page_content=content,
                metadata={"source": str(file_path), "type": file_path.suffix}
            )]
        except Exception as e:
            logger.warning(f"加载文本文件失败 {file_path}: {e}")
            return []

    def _load_jsonl_file(self, file_path: Path) -> List[Document]:
        """加载JSONL文件"""
        documents = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "q" in obj and "a" in obj:
                            text = f"问：{obj['q']}\n答：{obj['a']}"
                        else:
                            text = "\n".join(f"{k}：{v}" for k, v in obj.items() if not k.startswith("_"))
                        
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": str(file_path), "line": line_num, "type": "jsonl"}
                        ))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"加载JSONL文件失败 {file_path}: {e}")
        
        return documents

    def _load_csv_file(self, file_path: Path) -> List[Document]:
        """加载CSV文件"""
        documents = []
        try:
            import csv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for line_num, row in enumerate(reader, 2):
                    text = "，".join(f"{k}：{v}" for k, v in row.items() if v and v != "-")
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": str(file_path), "line": line_num, "type": "csv"}
                    ))
        except Exception as e:
            logger.warning(f"加载CSV文件失败 {file_path}: {e}")
        
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档（改进版：动态调整切分策略）"""
        # 分离原子文档（jsonl/csv）和需要切分的文档
        atomic_docs = [d for d in documents if d.metadata.get("type") in ("jsonl", "csv")]
        other_docs = [d for d in documents if d.metadata.get("type") not in ("jsonl", "csv")]

        # 按文件名分组，应用不同的切分策略
        grouped: Dict[str, List[Document]] = {}
        for doc in other_docs:
            profile = self._get_chunk_profile(doc.metadata.get("source", ""))
            grouped.setdefault(profile, []).append(doc)

        # 切分
        all_chunks = []
        for profile, docs in grouped.items():
            params = Config.CHUNK_PROFILES.get(profile, Config.CHUNK_PROFILES["default"])
            splitter = TextSplitter(
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
                separators=Config.CHUNK_SEPARATORS
            )

            for doc in docs:
                # 动态调整切分策略：根据文档大小
                doc_size = len(doc.page_content)
                if doc_size < params["chunk_size"] * 0.5:
                    # 文档很小，不切分
                    all_chunks.append(doc)
                else:
                    # 正常切分
                    chunks = splitter.split_text(doc.page_content)
                    for chunk in chunks:
                        all_chunks.append(Document(
                            page_content=chunk,
                            metadata=doc.metadata
                        ))

            logger.info(f"[{profile}] {len(docs)} 个文档 → {len(all_chunks)} 个 chunk")

        # 合并超短 chunk
        all_chunks = self._merge_short_chunks(all_chunks)

        # 添加原子文档
        all_chunks.extend(atomic_docs)

        logger.info(f"切分完成，共 {len(all_chunks)} 个文本块")
        return all_chunks

    def _get_chunk_profile(self, source: str) -> str:
        """根据文件名确定切分策略"""
        filename = os.path.basename(source)
        for profile, keywords in Config.CHUNK_FILE_RULES.items():
            if any(kw in filename for kw in keywords):
                return profile
        return "default"

    def _merge_short_chunks(self, chunks: List[Document]) -> List[Document]:
        """合并超短 chunk（改进版：更智能的合并逻辑）"""
        if not chunks:
            return chunks

        min_size = Config.CHUNK_MIN_SIZE
        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            if len(current.page_content) < min_size:
                # 合并到前一个块
                current = Document(
                    page_content=current.page_content + "\n" + chunk.page_content,
                    metadata=current.metadata
                )
            else:
                merged.append(current)
                current = chunk

        merged.append(current)
        return merged


# ========== 改进的嵌入模型 ==========

class EmbeddingModel:
    """嵌入模型（改进版：支持缓存管理和重试）"""

    def __init__(self):
        # Embedding 使用 SiliconFlow API
        api_key = Config.SILICONFLOW_API_KEY
        api_base = Config.SILICONFLOW_API_BASE
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = Config.EMBED_MODEL
        self._cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = Config.EMBED_CACHE_SIZE

    def _clean_cache(self):
        """清理缓存，保持最大大小"""
        with self._cache_lock:
            if len(self._cache) > self._max_cache_size:
                # 删除一半缓存
                keys_to_delete = list(self._cache.keys())[:len(self._cache) // 2]
                for key in keys_to_delete:
                    del self._cache[key]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档（改进版：支持重试和缓存清理）"""
        # 检查缓存
        uncached = []
        uncached_indices = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached.append(text)
                uncached_indices.append(i)

        # 调用 API 嵌入未缓存的文本（截断超长文本，避免API报错）
        if uncached:
            max_retries = Config.LLM_MAX_RETRIES
            
            for attempt in range(max_retries):
                try:
                    safe_texts = [t[:512] if len(t) > 512 else t for t in uncached]
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=safe_texts
                    )
                    
                    for j, item in enumerate(response.data):
                        idx = uncached_indices[j]
                        results[idx] = item.embedding
                        # 缓存
                        cache_key = hashlib.md5(texts[idx].encode()).hexdigest()
                        self._cache[cache_key] = item.embedding
                    
                    # 成功，跳出重试循环
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"嵌入失败，重试 {attempt + 1}/{max_retries}: {e}")
                        time.sleep(2 ** attempt)  # 指数退避
                    else:
                        logger.error(f"嵌入最终失败: {e}")
                        raise
            
            # 清理缓存
            self._clean_cache()

        return results

    def embed_query(self, query: str) -> List[float]:
        """嵌入查询（改进版：支持查询缓存）"""
        return self.embed_documents([query])[0]


# ========== 改进的BM25检索器 ==========

class BM25Retriever:
    """BM25 关键词检索器（改进版：支持延迟初始化和堆排序）"""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._query_cache: Dict[str, List[str]] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = Config.QUERY_CACHE_SIZE
        
        # 延迟初始化
        self.tokenized_docs = None
        self.bm25 = None
        
        logger.info(f"BM25 检索器初始化完成，文档数: {len(documents)}")

    def _init_index(self):
        """延迟初始化索引"""
        if self.tokenized_docs is None:
            self.tokenized_docs = [list(jieba.cut(doc.page_content)) for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """搜索（改进版：使用堆排序）"""
        self._init_index()
        
        # 缓存查询分词
        with self._cache_lock:
            if query not in self._query_cache:
                self._query_cache[query] = list(jieba.cut(query))
                
                # 清理缓存
                if len(self._query_cache) > self._max_cache_size:
                    keys_to_delete = list(self._query_cache.keys())[:len(self._query_cache) // 2]
                    for k in keys_to_delete:
                        del self._query_cache[k]
            
            tokenized_query = self._query_cache[query]
        
        # 计算分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 使用堆获取top_k（更高效）
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])
        
        # 过滤低分结果
        threshold = 0.1
        return [self.documents[i] for i in top_indices if scores[i] > threshold]


# ========== 改进的向量存储 ==========

class VectorStore:
    """向量存储（改进版：支持去重和阈值过滤）"""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        self.collection = None
        self._search_cache: Dict[str, List[Document]] = {}
        self._search_cache_lock = threading.Lock()
        self._max_cache_size = Config.QUERY_CACHE_SIZE

    def create_or_load(self, collection_name: str = None):
        """创建或加载集合"""
        collection_name = collection_name or Config.COLLECTION_NAME
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"向量库加载成功，文档数: {self.collection.count()}")
        return self

    def add_documents(self, documents: List[Document]):
        """添加文档（改进版：支持去重）"""
        if not self.collection:
            self.create_or_load()

        # 生成唯一ID
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # 检查是否已存在（基于内容哈希）
        existing_hashes = set()
        unique_documents = []
        unique_ids = []
        
        for i, doc in enumerate(documents):
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in existing_hashes:
                existing_hashes.add(content_hash)
                unique_documents.append(doc)
                unique_ids.append(ids[i])

        # 准备数据
        texts = [doc.page_content for doc in unique_documents]
        metadatas = [doc.metadata for doc in unique_documents]

        # 批量嵌入并添加
        batch_size = Config.EMBED_BATCH_SIZE
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = unique_ids[i:i + batch_size]

            embeddings = self.embedding_model.embed_documents(batch_texts)

            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )

        logger.info(f"添加了 {len(unique_documents)} 个文档到向量库（去重后）")

    def search(self, query: str, top_k: int = None) -> List[Document]:
        """向量检索（改进版：支持缓存和阈值过滤）"""
        if not self.collection:
            return []

        top_k = top_k or Config.VECTOR_TOP_K

        # 检查缓存
        cache_key = f"{query}:{top_k}"
        with self._search_cache_lock:
            if cache_key in self._search_cache:
                return self._search_cache[cache_key]

        # 嵌入查询
        query_embedding = self.embedding_model.embed_query(query)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 转换为 Document
        documents = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # 过滤低质量结果（距离阈值）
                if dist <= Config.SIMILARITY_THRESHOLD:
                    documents.append(Document(
                        page_content=doc,
                        metadata=meta or {}
                    ))

        # 缓存结果
        with self._search_cache_lock:
            self._search_cache[cache_key] = documents
            if len(self._search_cache) > self._max_cache_size:
                # 清理一半缓存
                keys_to_delete = list(self._search_cache.keys())[:len(self._search_cache) // 2]
                for k in keys_to_delete:
                    del self._search_cache[k]

        return documents

    def get_count(self) -> int:
        """获取文档数量"""
        if self.collection:
            return self.collection.count()
        return 0


# ========== 改进的Rerank重排序 ==========

class Reranker:
    """Rerank 重排序器（改进版：支持重试和缓存管理）"""

    def __init__(self):
        self.api_url = Config.RERANK_API_URL
        self.api_key = Config.SILICONFLOW_API_KEY
        self.model = Config.RERANK_MODEL
        self.timeout = Config.RERANK_TIMEOUT
        self._cache: Dict[str, List[int]] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = Config.RERANK_CACHE_SIZE

    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """重排序（改进版：支持重试和缓存管理）"""
        if not documents:
            return []

        top_k = top_k or Config.RETRIEVAL_K

        # 检查缓存
        cache_key = self._make_cache_key(query, documents)
        with self._cache_lock:
            if cache_key in self._cache:
                indices = self._cache[cache_key]
                return [documents[i] for i in indices if i < len(documents)]

        # 调用 API，支持重试
        max_retries = Config.LLM_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": [doc.page_content for doc in documents],
                    "top_n": top_k,
                    "return_documents": False
                }

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()

                # 提取排序后的索引
                indices = [item["index"] for item in result.get("results", [])]
                reranked = [documents[i] for i in indices if i < len(documents)]

                # 缓存结果
                with self._cache_lock:
                    self._cache[cache_key] = indices
                    if len(self._cache) > self._max_cache_size:
                        # 清理一半缓存
                        keys_to_delete = list(self._cache.keys())[:len(self._cache) // 2]
                        for k in keys_to_delete:
                            del self._cache[k]

                return reranked
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Rerank超时，重试 {attempt + 1}/{max_retries}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Rerank最终超时")
                    return documents[:top_k]
                    
            except Exception as e:
                logger.warning(f"Rerank失败: {e}")
                return documents[:top_k]

        return documents[:top_k]

    def _make_cache_key(self, query: str, documents: List[Document]) -> str:
        """生成缓存键（改进版：使用更可靠的哈希）"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        doc_contents = "".join(d.page_content[:100] for d in documents[:10])  # 限制文档数量
        doc_hash = hashlib.sha256(doc_contents.encode()).hexdigest()[:16]
        return f"{query_hash}:{doc_hash}"


# ========== 改进的LLM生成器 ==========

class LLMGenerator:
    """LLM 生成器（改进版：支持重试和通用提示词）"""

    def __init__(self):
        api_key = Config.LLM_API_KEY or Config.SILICONFLOW_API_KEY
        api_base = Config.LLM_API_BASE or Config.SILICONFLOW_API_BASE
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = Config.LLM_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.max_tokens = Config.LLM_MAX_TOKENS
        self.timeout = Config.LLM_TIMEOUT
        
        # 通用提示词模板
        self._default_prompt = """你是一个智能助手，请根据以下上下文回答用户问题。

【上下文信息】
{context}

【用户问题】
{query}

【回答】"""

    def generate(self, query: str, context: str, history: str = "") -> str:
        """生成回答（改进版：支持重试）"""
        prompt = self._default_prompt.format(context=context, query=query)
        
        max_retries = Config.LLM_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM生成失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"LLM生成最终失败: {e}")
                    return "抱歉，系统暂时无法回答您的问题，请稍后重试。"


# ========== 改进的混合检索器 ==========

class HybridRetriever:
    """混合检索器（改进版：支持配置化参数和重试）"""

    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25Retriever,
                 reranker: Reranker):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

    def search(self, query: str, top_k: int = None,
               vector_top_k: int = None, bm25_top_k: int = None) -> List[Document]:
        """混合检索（改进版：支持配置化参数）"""
        top_k = top_k or Config.RETRIEVAL_K
        vector_top_k = vector_top_k or Config.VECTOR_TOP_K
        bm25_top_k = bm25_top_k or Config.BM25_TOP_K

        # 并行执行向量检索和 BM25 检索
        vector_docs = []
        bm25_docs = []

        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    vector_future = executor.submit(
                        self.vector_store.search, query, vector_top_k
                    )
                    bm25_future = executor.submit(
                        self.bm25_retriever.search, query, bm25_top_k
                    )

                    # 统一超时时间
                    timeout = 30

                    try:
                        vector_docs = vector_future.result(timeout=timeout)
                        logger.info(f"向量检索命中: {len(vector_docs)} 个")
                    except Exception as e:
                        logger.warning(f"向量检索失败: {e}")

                    try:
                        bm25_docs = bm25_future.result(timeout=timeout)
                        logger.info(f"BM25 检索命中: {len(bm25_docs)} 个")
                    except Exception as e:
                        logger.warning(f"BM25 检索失败: {e}")

                # 成功，跳出重试循环
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"混合检索失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"混合检索最终失败: {e}")

        # 合并去重
        seen = set()
        merged = []
        for doc in vector_docs + bm25_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        if not merged:
            return []

        # Rerank 重排序
        reranked = self.reranker.rerank(query, merged, top_k)

        logger.info(f"检索完成: 向量 {len(vector_docs)} + BM25 {len(bm25_docs)} → {len(reranked)} 个")
        return reranked

    def bare_search(self, query: str, top_k: int = None) -> List[Document]:
        """混合检索（不带 Rerank），用于多路子查询场景"""
        top_k = top_k or Config.VECTOR_TOP_K

        # 并行执行向量检索和 BM25 检索
        vector_docs = []
        bm25_docs = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self.vector_store.search, query, top_k)
            bm25_future = executor.submit(self.bm25_retriever.search, query, Config.BM25_TOP_K)

            try:
                vector_docs = vector_future.result(timeout=30)
            except Exception as e:
                logger.warning(f"bare_search 向量检索失败: {e}")

            try:
                bm25_docs = bm25_future.result(timeout=10)
            except Exception as e:
                logger.warning(f"bare_search BM25 检索失败: {e}")

        # 合并去重
        seen = set()
        merged = []
        for doc in vector_docs + bm25_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        logger.info(f"bare_search: 向量 {len(vector_docs)} + BM25 {len(bm25_docs)} → 合并 {len(merged)}")
        return merged


# ========== 改进的RAG系统 ==========

class SimpleRAG:
    """纯 Python RAG 系统（改进版：支持懒加载）"""

    def __init__(self, lazy_load: bool = False):
        logger.info("初始化 RAG 系统...")

        # 初始化基础组件
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.reranker = Reranker()
        self.llm = LLMGenerator()

        # 懒加载或立即加载
        if not lazy_load:
            self._init_full()
        else:
            self.bm25_retriever = None
            self.hybrid_retriever = None

        logger.info("RAG 系统初始化完成")

    def _init_full(self):
        """完整初始化"""
        # 加载向量库
        self.vector_store.create_or_load()

        # 加载 BM25 索引
        self._init_bm25()

        # 初始化混合检索器
        if self.bm25_retriever:
            self.hybrid_retriever = HybridRetriever(
                self.vector_store,
                self.bm25_retriever,
                self.reranker
            )

    def _init_bm25(self):
        """初始化 BM25 索引"""
        count = self.vector_store.get_count()
        if count == 0:
            logger.warning("向量库为空，跳过 BM25 初始化")
            return

        try:
            # 从向量库获取所有文档
            collection = self.vector_store.collection
            all_data = collection.get(include=["documents", "metadatas"])

            documents = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(all_data["documents"], all_data["metadatas"])
            ]

            self.bm25_retriever = BM25Retriever(documents)
            logger.info(f"BM25 索引初始化完成，文档数: {len(documents)}")
        except Exception as e:
            logger.error(f"BM25 初始化失败: {e}")

    def load_knowledge_base(self, data_dir: str = None):
        """加载知识库"""
        processor = DocumentProcessor()

        # 加载文档
        documents = processor.load_documents(data_dir)
        if not documents:
            logger.warning("没有找到文档")
            return 0

        # 切分文档
        chunks = processor.split_documents(documents)

        # 添加到向量库
        self.vector_store.add_documents(chunks)

        # 重新初始化 BM25
        self._init_bm25()

        # 重新初始化混合检索器
        if self.bm25_retriever:
            self.hybrid_retriever = HybridRetriever(
                self.vector_store,
                self.bm25_retriever,
                self.reranker
            )

        return len(chunks)

    def query(self, question: str, history: str = "") -> QueryResult:
        """查询（改进版：支持错误恢复）"""
        start_time = time.time()
        query_id = str(uuid.uuid4())

        if not self.hybrid_retriever:
            return QueryResult(
                answer="知识库未初始化，请先加载知识库",
                sources=[],
                success=False,
                processing_time=0,
                error_message="知识库未初始化",
                query_id=query_id
            )

        # 查询改写
        rewritten = self._rewrite_query(question)

        # 意图分类 + 查询扩展
        intent = self._classify_intent(rewritten)
        queries = self._expand_queries(rewritten, intent)

        if len(queries) > 1:
            logger.info(f"查询扩展: intent={intent}, {len(queries)} 个子查询: {queries}")
            docs = self._multi_query_retrieve(queries)
        else:
            docs = self.hybrid_retriever.search(rewritten, Config.RETRIEVAL_K)

        if not docs:
            return QueryResult(
                answer="抱歉，未找到相关信息",
                sources=[],
                success=True,
                processing_time=time.time() - start_time,
                query_id=query_id
            )

        # 拼接上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.page_content[:100] for doc in docs]

        # LLM 生成
        answer = self.llm.generate(question, context, history)

        elapsed = time.time() - start_time

        return QueryResult(
            answer=answer,
            sources=sources,
            success=True,
            processing_time=elapsed,
            query_id=query_id
        )

    def _rewrite_query(self, query: str) -> str:
        """查询改写（改进版：支持同义词扩展）"""
        rewritten = query

        # 单位转换
        rewritten = re.sub(r'(?<!\w)(\d+)G(?!\w)', r'\1GB', rewritten)

        # 价格提取
        price_match = re.search(r'(\d+)\s*元', query)
        if price_match:
            rewritten += f" {price_match.group(1)}元"

        # 同义词扩展
        synonyms = {
            "话费": ["月租", "月费", "资费"],
            "流量": ["数据流量", "上网流量"],
            "通话": ["语音", "电话"],
        }

        for word, syns in synonyms.items():
            if word in rewritten:
                # 添加同义词
                rewritten += " " + " ".join(syns)

        return rewritten

    def _classify_intent(self, query: str) -> str:
        """意图分类（改进版：支持更复杂的逻辑）"""
        # 使用规则+关键词的组合判断

        # 推荐意图
        recommend_patterns = [
            r"推荐.*套餐",
            r"适合.*套餐",
            r"怎么选",
            r"选哪个",
        ]

        # 对比意图
        compare_patterns = [
            r".*和.*对比",
            r".*与.*比较",
            r"区别.*是什么",
        ]

        # 检查推荐意图
        for pattern in recommend_patterns:
            if re.search(pattern, query):
                return "recommend"

        # 检查对比意图
        for pattern in compare_patterns:
            if re.search(pattern, query):
                return "compare"

        # 默认返回fact
        return "fact"

    def _expand_queries(self, query: str, intent: str) -> List[str]:
        """查询扩展（改进版：支持配置化扩展规则）"""
        queries = [query]  # 始终包含原始查询

        if intent == "recommend":
            # 学生/低价场景
            if any(kw in query for kw in ["学生", "便宜", "低价", "预算"]):
                queries.extend(["29元套餐", "39元套餐 星卡", "59元套餐 流量"])
            # 大流量/视频场景
            if any(kw in query for kw in ["流量多", "大流量", "视频", "看视频"]):
                queries.extend(["大流量套餐", "129元套餐 流量", "199元套餐 流量", "299元套餐 流量"])
            # 全家/家庭场景
            if any(kw in query for kw in ["全家", "老人", "小孩", "家庭"]):
                queries.extend(["副卡 套餐", "129元套餐 副卡", "宽带 融合套餐"])
            # 购机场景
            if any(kw in query for kw in ["买手机", "购机", "新手机"]):
                queries.extend(["橙分期 补贴", "购机优惠 直降"])

        elif intent == "compare":
            # 提取套餐数字，为每个套餐生成独立查询
            tiers = re.findall(r'(\d+)\s*元', query)
            for tier in tiers:
                queries.append(f"{tier}元套餐 流量 通话 宽带")

        # 去重
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def _multi_query_retrieve(self, queries: List[str]) -> List[Document]:
        """多查询并行检索（改进版：支持超时控制）"""
        all_docs = []
        
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            futures = {
                executor.submit(self.hybrid_retriever.bare_search, q, Config.VECTOR_TOP_K): q
                for q in queries
            }
            
            # 等待所有任务完成，支持超时
            for future in as_completed(futures, timeout=60):
                try:
                    docs = future.result(timeout=30)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"多路检索子查询失败: {e}")

        # 合并去重
        seen = set()
        merged = []
        for doc in all_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        if not merged:
            return []

        # 只做一次 Rerank
        reranked = self.reranker.rerank(queries[0], merged, Config.RETRIEVAL_K)
        logger.info(f"多路检索: {len(queries)} 子查询 → {len(all_docs)} 条 → 去重 {len(merged)} → Rerank {len(reranked)}")
        return reranked

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "status": "ready",
            "document_count": self.vector_store.get_count(),
            "llm_model": Config.LLM_MODEL,
            "embedding_model": Config.EMBED_MODEL,
            "rerank_model": Config.RERANK_MODEL,
            "vector_db_path": Config.VECTOR_DB_PATH,
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 验证配置
    Config.validate()

    # 创建 RAG 实例（支持懒加载）
    rag = SimpleRAG(lazy_load=True)

    # 如果向量库为空，加载知识库
    if rag.get_stats()["document_count"] == 0:
        print("知识库为空，开始加载...")
        count = rag.load_knowledge_base(Config.DATA_DIR)
        print(f"加载完成，共 {count} 个文本块")

    # 打印统计信息
    stats = rag.get_stats()
    print(f"\n知识库状态:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 交互式查询
    print("\n" + "=" * 50)
    print("RAG 查询系统（输入 quit 退出）")
    print("=" * 50)

    while True:
        try:
            question = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        result = rag.query(question)
        print(f"\n💬 回答:\n{result.answer}")
        print(f"\n⏱  耗时: {result.processing_time:.2f}s")
