# simple_rag.py Review 精简版

> 日期：2026-08-24 | 字符数：约3000

---

## 一、dx_agent.py 检索路由（已确认）

三种路由底层都是混合检索（向量+BM25）+ Rerank + LLM生成，区别在于**召回策略**：

| 路由 | 触发条件 | 召回策略 |
|------|----------|----------|
| `simple` | 无对比/计算/推荐关键词 | 单次查询，rag.query() |
| `comparison` | 对比关键词 + 隐含对比模式 | 按档位分别检索，合并去重 |
| `complex` | 计算/推荐/多档位 | Agent多轮调用工具 |

路由划分本质上是**关键词+正则的规则引擎**，不是语义理解。

---

## 二、simple_rag.py 各模块实际问题

### TextSplitter（第132-189行）
- **重叠截断问题**：`_add_overlap` 直接按字符数截取，可能断在词中间
- **分隔符选择**：`_choose_separator` 选第一个命中的，不是最优的

### EmbeddingModel（第341-389行）
- **缓存无上限**：`self._cache` 字典没有大小限制，长时间运行会OOM
- **异常直接抛出**：embed失败没有重试，上层调用者全崩

### BM25Retriever（第394-419行）
- **初始化同步分词**：`__init__` 对全部文档jieba分词，文档多时启动慢
- **排序效率**：用`sorted`对全量分数排序取top_k，应该用`heapq.nlargest`

### VectorStore（第424-506行）
- **ID不唯一**：`ids = [f"doc_{i}" for i in range(len(documents))]`，重新加载会覆盖
- **无距离阈值**：search返回top_k，不管距离多远都返回，可能召回不相关内容

### Reranker（第511-578行）
- **缓存键易冲突**：只取文档前50字符做hash，不同文档可能hash相同
- **失败静默降级**：rerank失败直接返回前top_k个，没有重试，没有告警

### HybridRetriever（第637-720行）
- **超时不一致**：向量检索timeout=30s，BM25 timeout=10s
- **bare_search命名误导**：参数名top_k实际用于向量检索，BM25用的是Config.BM25_TOP_K

### SimpleRAG（第725-953行）
- **初始化一次性加载所有组件**，没有懒加载
- **_rewrite_query 过于简单**：只做"G→GB"和价格提取，没有同义词扩展
- **_classify_intent 只靠关键词**："推荐"→recommend，"对比"→compare，其余→fact

---

## 三、真正需要修的3个问题（按优先级）

### P0：缓存无上限
```python
# EmbeddingModel._cache、BM25Retriever._query_cache 都没有大小限制
# 长时间运行的服务会OOM
# 修复：加LRU或定期清理
```

### P1：Rerank失败静默降级
```python
# 当前：失败直接返回未经rerank的top_k
# 风险：返回质量差的结果，用户无感知
# 修复：至少记录error日志，考虑重试1次
```

### P2：向量检索无距离阈值
```python
# 当前：只要top_k个，不管相似度多低都返回
# 风险：召回大量不相关内容，污染LLM上下文
# 修复：加 Config.SIMILARITY_THRESHOLD 过滤
```

---

## 四、不建议改的部分

- **Config类**：当前用环境变量的方式够用，不需要引入YAML
- **LLMGenerator的prompt**：特定业务场景的prompt改了反而影响效果
- **分类型切分策略**：small/large/default的规则是业务经验，不是代码问题

---

*详细版见同目录下其他3个review文档（共约8万字符，仅供存档）*
