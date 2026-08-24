"""
向量库构建脚本
从文档构建ChromaDB向量数据库
"""

import os
import re
import sys
import shutil
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import Config

# 已知套餐档位（用于从chunk内容中提取plan_tier元数据）
_PLAN_TIERS = ['29', '39', '59', '79', '99', '129', '169', '199', '229', '299']


def _extract_plan_tiers(content: str) -> str:
    """从chunk内容中提取涉及的套餐档位，返回逗号分隔字符串。
    用于在检索时按套餐档位过滤，防止跨套餐数据混淆。
    """
    found = set()
    # 匹配 "59元"、"畅享59"、"59套餐"、"59元/月" 等模式
    for tier in _PLAN_TIERS:
        patterns = [
            rf'(?<!\d){tier}\s*元',           # 59元
            rf'畅享\s*{tier}(?!\d)',           # 畅享59
            rf'{tier}\s*套餐',                 # 59套餐
            rf'(?<!\d){tier}(?!\d)\s*元/月',   # 59元/月
        ]
        for p in patterns:
            if re.search(p, content):
                found.add(tier)
                break
    return ",".join(sorted(found, key=int)) if found else ""


def _extract_plan_tier_from_header(text_before: str) -> str:
    """从 chunk 前面的上级标题中提取套餐档位。
    例如 "### 3. 5G畅享99元套餐" → "99"
    用于在 chunk 内容本身不含档位关键词时，继承父级标题的档位信息。
    """
    # 匹配常见标题格式：### N. 5G畅享XX元套餐 / 5G畅享XX元套餐
    m = re.search(r'畅享\s*(\d+)\s*元?\s*套餐', text_before)
    if m and m.group(1) in _PLAN_TIERS:
        return m.group(1)
    m = re.search(r'(\d+)\s*元\s*套餐', text_before)
    if m and m.group(1) in _PLAN_TIERS:
        return m.group(1)
    return ""


def _plan_level_split(file_path: str, content: str) -> List[Document]:
    """套餐级切分：先按 PLAN_LEVEL_CHUNK_SEPARATOR 切分成大块（每个套餐），
    内容超过 max_size 的再按子标题二次切分。
    每个 chunk 的 metadata.plan_tier 从其所属套餐标题中提取。
    """
    separator = Config.PLAN_LEVEL_CHUNK_SEPARATOR
    max_size = Config.PLAN_LEVEL_MAX_CHUNK_SIZE
    source = str(file_path)

    # 按正则分隔符切分
    parts = re.split(separator, content)

    docs = []
    current_tier = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # 尝试从这部分内容的开头提取套餐档位（标题行）
        tier = _extract_plan_tier_from_header(part[:200])
        if tier:
            current_tier = tier

        # 如果这部分不超过 max_size，作为一个完整 chunk
        if len(part) <= max_size:
            tier_str = current_tier or _extract_plan_tiers(part)
            docs.append(Document(
                page_content=part,
                metadata={"source": source, "plan_tier": tier_str}
            ))
        else:
            # 超过 max_size，按子标题二次切分
            # 在套餐内部保持同一个 current_tier
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=Config.CHUNK_PROFILES["large"]["chunk_overlap"],
                separators=["\n\n", "###", "####", "\n", "。", "；"]
            )
            sub_docs = sub_splitter.create_documents([part])
            for sd in sub_docs:
                # 子 chunk 优先继承套餐标题的 tier，fallback 到内容匹配
                tier_str = current_tier or _extract_plan_tiers(sd.page_content)
                sd.metadata = {"source": source, "plan_tier": tier_str}
            docs.extend(sub_docs)

    return docs


def _should_use_plan_level(file_path: str) -> bool:
    """判断文件是否应使用套餐级切分"""
    filename = os.path.basename(file_path)
    return any(kw in filename for kw in Config.PLAN_LEVEL_CHUNK_FILES)


def build_vectorstore(
    data_dir: str = None,
    persist_directory: str = None,
    force_rebuild: bool = False
) -> int:
    print("=" * 50)
    print("向量库构建工具")
    print("=" * 50)

    if data_dir is None:
        data_dir = Config.DATA_DIR
    if persist_directory is None:
        persist_directory = Config.VECTOR_DB_PATH

    print(f"数据目录: {data_dir}")
    print(f"向量库路径: {persist_directory}")
    print(f"强制重建: {force_rebuild}")

    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return 0

    txt_files = list(Path(data_dir).glob("**/*.txt")) + list(Path(data_dir).glob("**/*.md"))
    jsonl_files = list(Path(data_dir).glob("**/*.jsonl"))
    if not txt_files and not jsonl_files:
        print(f"在 {data_dir} 中未找到 .txt / .md / .jsonl 文件")
        return 0

    all_files = txt_files + jsonl_files
    print(f"找到 {len(txt_files)} 个文档文件 + {len(jsonl_files)} 个 JSONL 文件:")
    for file in all_files[:5]:
        print(f"   - {file.name}")
    if len(all_files) > 5:
        print(f"   ... 还有 {len(all_files) - 5} 个文件")

    if os.path.exists(persist_directory) and force_rebuild:
        print(f"清理旧向量库: {persist_directory}")
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)

    print("初始化嵌入模型...")
    try:
        from local_embeddings import LocalBGEEmbeddings
        embedding = LocalBGEEmbeddings(model_name=Config.EMBED_MODEL)
        print("使用本地嵌入模型")
    except ImportError as e:
        raise ImportError(f"本地嵌入模型不可用: {e}。请安装sentence-transformers库。")
    except Exception as e:
        raise RuntimeError(f"本地嵌入模型加载失败: {e}")

    print("加载文档...")
    documents = []
    atomic_documents = []

    # 加载 .txt 和 .md 文件
    files = list(Path(data_dir).glob("**/*.txt")) + list(Path(data_dir).glob("**/*.md"))
    for file in files:
        try:
            loader = TextLoader(str(file), encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"加载 {file.name} 失败: {e}")

    # 加载 .jsonl 文件（每行作为一个独立文档，不二次切分）
    import json as json_mod
    for jsonl_path in Path(data_dir).glob("**/*.jsonl"):
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json_mod.loads(line)
                        if "q" in obj and "a" in obj:
                            text = f"问：{obj['q']}\n答：{obj['a']}"
                        else:
                            text = "\n".join(f"{k}：{v}" for k, v in obj.items() if not k.startswith("_"))
                        atomic_documents.append(Document(
                            page_content=text,
                            metadata={"source": str(jsonl_path), "line": line_num, "plan_tier": _extract_plan_tiers(text)}
                        ))
                    except json_mod.JSONDecodeError:
                        print(f"  JSONL 解析失败: {jsonl_path} 第{line_num}行")
            print(f"  加载 JSONL: {jsonl_path.name}")
        except Exception as e:
            print(f"  加载 JSONL 失败: {jsonl_path}: {e}")

    # 加载 .csv 文件（首行为表头，每行转为可读文本）
    import csv as csv_mod
    for csv_path in Path(data_dir).glob("**/*.csv"):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for line_num, row in enumerate(reader, 2):
                    text = "，".join(f"{k}：{v}" for k, v in row.items() if v and v != "-")
                    atomic_documents.append(Document(
                        page_content=text,
                        metadata={"source": str(csv_path), "line": line_num, "plan_tier": _extract_plan_tiers(text)}
                    ))
            print(f"  加载 CSV: {csv_path.name}")
        except Exception as e:
            print(f"  加载 CSV 失败: {csv_path}: {e}")

    print(f"加载了 {len(documents)} 个文档文件 + {len(atomic_documents)} 条原子记录")

    print("分割文档 (差异化切分)...")

    def _get_chunk_profile(source: str) -> str:
        filename = os.path.basename(source)
        for profile, keywords in Config.CHUNK_FILE_RULES.items():
            if any(kw in filename for kw in keywords):
                return profile
        return "default"

    from collections import defaultdict

    # 先分出套餐级切分的文档和普通文档
    plan_level_docs = []
    normal_docs = []
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        if _should_use_plan_level(source):
            plan_level_docs.append(doc)
        else:
            normal_docs.append(doc)

    all_texts = []

    # 套餐级切分
    if plan_level_docs:
        print(f"  [套餐级切分] {len(plan_level_docs)} 个文档, 分隔符={Config.PLAN_LEVEL_CHUNK_SEPARATOR}, 最大块={Config.PLAN_LEVEL_MAX_CHUNK_SIZE}")
        for doc in plan_level_docs:
            source = doc.metadata.get('source', 'unknown')
            chunks = _plan_level_split(source, doc.page_content)
            print(f"    {os.path.basename(source)}: {len(doc.page_content)}字 → {len(chunks)} 个chunk")
            all_texts.extend(chunks)

    # 普通文档按 profile 切分
    grouped_docs = defaultdict(list)
    for doc in normal_docs:
        source = doc.metadata.get('source', 'unknown')
        profile = _get_chunk_profile(source)
        grouped_docs[profile].append(doc)

    for profile, docs in grouped_docs.items():
        params = Config.CHUNK_PROFILES[profile]
        print(f"  [{profile}] {len(docs)} 个文档, 块大小={params['chunk_size']}, 重叠={params['chunk_overlap']}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            separators=Config.CHUNK_SEPARATORS
        )
        chunks = splitter.split_documents(docs)
        all_texts.extend(chunks)

    # 合并超短 chunk：低于 CHUNK_MIN_SIZE 的 chunk 与前一个 chunk 合并
    min_size = Config.CHUNK_MIN_SIZE
    merged_count = 0
    final_texts = []
    for chunk in all_texts:
        if final_texts and len(chunk.page_content) < min_size:
            final_texts[-1] = Document(
                page_content=final_texts[-1].page_content + "\n" + chunk.page_content,
                metadata=final_texts[-1].metadata
            )
            merged_count += 1
        else:
            final_texts.append(chunk)
    if merged_count:
        print(f"  合并超短 chunk: {merged_count} 个 (< {min_size} 字)")
    all_texts = final_texts

    # 为切分后的 chunk 添加 plan_tier 元数据（套餐级切分的已有 metadata，跳过）
    for chunk in all_texts:
        if "plan_tier" not in chunk.metadata or not chunk.metadata["plan_tier"]:
            chunk.metadata["plan_tier"] = _extract_plan_tiers(chunk.page_content)

    # JSONL/CSV 文档直接加入，不经过切分器
    if atomic_documents:
        print(f"  [atomic] {len(atomic_documents)} 条记录直接加入（跳过切分）")
        all_texts.extend(atomic_documents)

    texts = all_texts
    print(f"切分完成，共 {len(texts)} 个文本块")

    if texts:
        print(f"分割示例（前2个块）:")
        for i, doc in enumerate(texts[:2]):
            print(f"   块{i+1}: {doc.page_content[:100]}...")

    print("创建向量数据库...")
    import time
    batch_size = 10  # 减小 batch 避免 API 限流
    vectorstore = None
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  处理批次 {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}: {len(batch)} 个文档块")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory=persist_directory,
                collection_name=Config.COLLECTION_NAME
            )
        else:
            vectorstore.add_documents(batch)
        # 批次间短暂等待，避免 API 限流
        if i + batch_size < len(texts):
            time.sleep(0.5)

    print("向量数据库已保存")

    print("验证向量数据库...")
    collection = vectorstore._collection
    doc_count = collection.count()

    print("=" * 50)
    print(f"向量库构建完成!")
    print(f"文档块数量: {doc_count}")
    print(f"保存位置: {persist_directory}")
    print(f"嵌入模型: {Config.EMBED_MODEL}")
    print("=" * 50)

    metadata_file = os.path.join(persist_directory, "metadata.txt")
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(f"构建时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"文档数量: {len(documents) + len(atomic_documents)}\n")
        f.write(f"文档块数量: {doc_count}\n")
        f.write(f"切分配置: {Config.CHUNK_PROFILES}\n")
        f.write(f"嵌入模型: {Config.EMBED_MODEL}\n")
        f.write(f"数据目录: {data_dir}\n")

    return doc_count


def incremental_update(data_dir: str = None, persist_directory: str = None) -> int:
    print("=" * 50)
    print("增量更新向量库")
    print("=" * 50)

    if data_dir is None:
        data_dir = Config.DATA_DIR
    if persist_directory is None:
        persist_directory = Config.VECTOR_DB_PATH

    if not os.path.exists(persist_directory):
        print("向量库不存在，执行完整构建")
        return build_vectorstore(data_dir, persist_directory)

    print("增量更新功能需要更复杂的文档哈希比较逻辑")
    print("建议使用 force_rebuild=True 重新构建")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="向量库构建工具")
    parser.add_argument("--data-dir", type=str, help="数据目录路径")
    parser.add_argument("--persist-dir", type=str, help="向量库保存路径")
    parser.add_argument("--force", action="store_true", help="强制重建")
    parser.add_argument("--incremental", action="store_true", help="增量更新")

    args = parser.parse_args()

    try:
        Config.validate()

        if args.incremental:
            count = incremental_update(args.data_dir, args.persist_dir)
        else:
            count = build_vectorstore(
                args.data_dir,
                args.persist_dir,
                force_rebuild=args.force
            )

        if count > 0:
            print(f"\n下一步: 启动API服务")
            print(f"   python api.py")
            print(f"\n测试查询:")
            print(f"   python workflow_langchain.py '你的问题'")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
