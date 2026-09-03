"""
RecursiveCharacterTextSplitter 纯 Python 等价实现

1:1 移植自 langchain-text-splitters 0.2.x（base.py + character.py），
默认参数行为与 LangChain 完全一致：
    keep_separator=True, strip_whitespace=True, length_function=len

本项目用它替代 build_vectors.py 对 LangChain 的最后一处依赖。
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class Document:
    """与 LangChain Document 等价的轻量文档结构。"""
    page_content: str
    metadata: Dict = field(default_factory=dict)


def _split_text_with_regex(
    text: str, separator: str, keep_separator=True
) -> List[str]:
    """按分隔符拆分文本；keep_separator=True 时分隔符保留在后续片段开头。"""
    if separator:
        # 捕获组让分隔符保留在结果中
        _splits = re.split(f"({separator})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
        if len(_splits) % 2 == 0:
            splits += _splits[-1:]
        splits = [_splits[0]] + splits
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter:
    """递归字符切分器：按分隔符优先级递归切分，合并出不超过 chunk_size 的块。"""

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 不能大于 chunk_size ({chunk_size})"
            )
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._keep_separator = keep_separator
        self._strip_whitespace = strip_whitespace

    # ── 内部方法：与 langchain 实现逐行对应 ──

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """把碎片合并成不超过 chunk_size 的块，块间保留 chunk_overlap 重叠。"""
        separator_len = len(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = len(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # 持续弹出直到：剩余部分小于 overlap，或能容纳下一个碎片
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= len(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归切分：选择文本中存在的最高优先级分隔符，超长片段继续向下切。"""
        final_chunks = []
        # 选择合适的分隔符
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if re.search(re.escape(_s), text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # 合并碎片，超长片段递归切分
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if len(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    # ── 公开接口 ──

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> List[Document]:
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                documents.append(Document(page_content=chunk, metadata=metadata))
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)
