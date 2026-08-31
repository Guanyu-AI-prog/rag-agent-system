# RAG Agent System

纯Python RAG + Agent系统，支持知识库问答、工具调用、费用计算。正在逐步去除LangChain依赖。

## 📁 项目结构

```
rag-agent-system/
├── simple_rag.py              # 纯RAG实现（向量检索 + BM25 + Rerank）
├── dx_agent.py                # 纯Python Agent（正则匹配工具调用，无LangChain）
├── taocan_agent.py            # LangChain Agent（Tool Calling，逐步废弃）
├── api.py                     # FastAPI服务（RAG模式）
├── api_agent.py               # FastAPI服务（Agent模式）
├── dx_agent_api.py            # FastAPI服务（纯Python Agent模式）
├── config.py                  # 系统配置
├── config_manager.py          # 配置管理器
├── build_vectors.py           # 向量库构建脚本
├── local_embeddings.py        # 本地Embedding（优先本地，API兜底）
├── conversation_history.py    # 对话历史管理
├── cache_manager.py           # 缓存管理
├── circuit_breaker.py         # 熔断器
├── rate_limiter.py            # 限流器
├── metrics.py                 # 指标收集
├── query_logger.py            # 查询日志
├── structured_logging.py      # 结构化日志
├── voice_api.py               # 语音API
├── voice_module.py            # 语音模块
├── workflow_langchain.py      # LangChain RAG工作流（旧）
├── gunicorn.conf.py           # Gunicorn配置
├── start.sh                   # 启动脚本
├── requirements.txt           # 完整依赖
├── requirements_pure.txt      # 精简依赖（无LangChain）
├── .env.example               # 环境变量示例
├── visualize_eval.py          # 评测数据可视化脚本
├── data/                      # 知识库源数据
├── docs/                      # 文档
│   └── images/                # 评测图表
├── static/                    # Web UI
├── eval_*.py                  # 评测脚本
├── eval_*.json                # 评测结果
└── test_*.py                  # 测试脚本
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖（二选一）
pip install -r requirements.txt        # 完整版（含LangChain）
pip install -r requirements_pure.txt   # 精简版（纯Python Agent用）
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入API密钥
```

### 3. 构建向量库

```bash
python build_vectors.py --force
```

### 4. 启动服务

```bash
# RAG模式
python api.py

# Agent模式
python api_agent.py

# 纯Python Agent模式
python dx_agent_api.py
```

### 5. 测试

```bash
# 命令行查询
python simple_rag.py "有哪些套餐？"
python dx_agent.py "59元套餐多少流量？"

# 运行测试脚本
python test_system.py
```

## 🔧 核心组件

### 纯RAG（simple_rag.py）

- 向量检索（Chroma） + BM25 混合检索 + Rerank重排序
- 本地Embedding优先，API兜底
- 缓存、熔断、限流、指标收集

### 纯Python Agent（dx_agent.py）

- 纯Python实现，正则匹配LLM输出解析工具调用
- 3个工具：套餐知识查询、费用计算器、套餐统计
- 无LangChain依赖

### LangChain Agent（taocan_agent.py）

- LangChain AgentExecutor + Tool Calling
- 正在逐步被dx_agent.py替代

## 📊 评测数据可视化

> 由 `visualize_eval.py` 自动生成，数据来自8份迭代评测记录。

### 各题型表现
![题型得分](docs/images/01_question_type_scores.png)

### 综合能力雷达图
![雷达图](docs/images/02_capability_radar.png)

### 通过率 & 响应时间趋势
![趋势图](docs/images/03_trends.png)

### 调优前后效果对比
![调优对比](docs/images/04_before_after.png)

### 纯RAG vs 纯Python Agent 对比
![方案对比](docs/images/05_scheme_comparison.png)

### 评测文件说明

| 文件 | 说明 |
|------|------|
| `eval_simple_rag.py` | 纯RAG评测脚本（5维度打分） |
| `eval_dx_agent.py` | 纯Python Agent评测脚本 |
| `eval_simple_rag_*.json` | RAG评测结果（4次迭代） |
| `eval_dx_agent_*.json` | Agent评测结果（4次迭代） |
| `test_questions.py` | 30道标准测试题库 |
| `benchmark.py` | RAG API并发压测脚本 |
| `visualize_eval.py` | 评测数据可视化脚本 |

## 🐛 常见问题

**Q: 启动时报错"SILICONFLOW_API_KEY未设置"**
A: 请确保已创建`.env`文件并填入有效的API密钥。

**Q: 查询返回"向量库未初始化"**
A: 请先运行`python build_vectors.py --force`构建向量库。

**Q: 响应时间太长（超过30秒）**
A: 1. 检查LLM模型配置 2. 启用缓存（默认已启用） 3. 检查网络连接

**Q: 如何添加新文档？**
A: 将文件放入`data/`目录，运行`python build_vectors.py`，重启服务。

## 📈 监控

```bash
# 缓存状态
curl http://localhost:8000/cache/stats

# 系统统计
curl http://localhost:8000/stats

# 日志级别（在.env中设置）
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## 📜 许可证

MIT

---

*最后更新：2026年8月31日*
