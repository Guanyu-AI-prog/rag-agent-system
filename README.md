# RAG Agent System

纯Python RAG + Agent系统，支持知识库问答、工具调用、费用计算。正在逐步去除LangChain依赖。

> ⚠️ **评测说明**：本项目评测题目来自向量数据库中的已有内容（开卷考试），系统从知识库中检索到相关文档后作答。因此准确率、召回率等指标较高，不代表系统在面对未知问题时也能达到同样效果。实际生产环境中，用户问题与知识库内容的匹配度会有所下降。

## 📁 项目结构

```
rag-agent-system/
│
├── core/                        # 核心业务（RAG/Agent引擎）
│   ├── simple_rag.py            #   纯RAG（向量检索 + BM25 + Rerank）
│   ├── dx_agent.py              #   纯Python Agent（正则匹配工具调用）
│   ├── taocan_agent.py          #   LangChain Agent（逐步废弃）
│   ├── simple_rag_lite.py       #   RAG精简版
│   ├── simple_rag_improvements.py #  RAG改进版
│   └── workflow_langchain.py    #   LangChain工作流（旧）
│
├── api/                         # API服务层
│   ├── api.py                   #   FastAPI（RAG模式）
│   ├── api_agent.py             #   FastAPI（Agent模式）
│   ├── dx_agent_api.py          #   FastAPI（纯Python Agent）
│   ├── gunicorn.conf.py         #   Gunicorn配置
│   └── start.sh                 #   启动脚本
│
├── vector/                      # 向量库/Embedding
│   ├── build_vectors.py         #   向量库构建
│   └── local_embeddings.py      #   本地Embedding
│
├── infra/                       # 基础设施（通用组件）
│   ├── config.py                #   系统配置
│   ├── config_manager.py        #   配置管理器
│   ├── cache_manager.py         #   缓存管理
│   ├── conversation_history.py  #   对话历史
│   ├── circuit_breaker.py       #   熔断器
│   ├── rate_limiter.py          #   限流器
│   ├── metrics.py               #   指标收集
│   ├── query_logger.py          #   查询日志
│   └── structured_logging.py    #   结构化日志
│
├── extensions/                  # 扩展功能
│   └── voice/                   #   语音模块
│       ├── voice_api.py
│       └── voice_module.py
│
├── evaluation/                  # 评测
│   ├── scripts/                 #   评测脚本
│   ├── results/                 #   评测结果JSON
│   ├── reports/                 #   评测报告
│   ├── tests/                   #   测试脚本 & 题库
│   └── tools/                   #   压测/调试工具
│
├── static/                      # Web UI
├── docs/                        # 文档 & 评测图表
│
├── requirements.txt             # 完整依赖
├── requirements_pure.txt        # 精简依赖（无LangChain）
├── .env.example                 # 环境变量示例
└── visualize_eval.py            # 评测数据可视化
```

## 🚀 快速开始

```bash
# 安装依赖
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # 完整版
pip install -r requirements_pure.txt   # 精简版（纯Python Agent）

# 配置
cp .env.example .env
# 编辑 .env 填入API密钥

# 构建向量库
python vector/build_vectors.py --force

# 启动
python api/api.py              # RAG模式
python api/api_agent.py        # Agent模式
python api/dx_agent_api.py     # 纯Python Agent模式

# 命令行测试
python core/simple_rag.py "有哪些套餐？"
python core/dx_agent.py "59元套餐多少流量？"
```

## 🔧 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 纯RAG | `core/simple_rag.py` | 向量检索 + BM25 + Rerank，有缓存/熔断/限流 |
| 纯Python Agent | `core/dx_agent.py` | 正则匹配工具调用，3个工具，无LangChain依赖 |
| LangChain Agent | `core/taocan_agent.py` | AgentExecutor + Tool Calling，逐步废弃 |

## 📊 评测数据可视化

> ⚠️ 以下评测数据基于向量库中已有内容出题（开卷），系统检索到相关文档后作答。
> 高准确率反映的是"检索 + 生成"链路的工程质量，不代表对未知问题的泛化能力。

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

## 🐛 常见问题

**Q: 启动时报错"SILICONFLOW_API_KEY未设置"**
A: 确保`.env`文件已填入有效的API密钥。

**Q: 查询返回"向量库未初始化"**
A: 运行`python vector/build_vectors.py --force`构建向量库。

**Q: 响应时间太长**
A: 检查LLM模型配置，启用缓存，检查网络连接。

**Q: 如何添加新文档？**
A: 放入`data/`目录，运行`python vector/build_vectors.py`，重启服务。

## 📈 监控

```bash
curl http://localhost:8000/cache/stats    # 缓存状态
curl http://localhost:8000/stats          # 系统统计
# 日志级别在 .env 中设置: LOG_LEVEL=INFO
```

## 📜 许可证

MIT

---

*最后更新：2026年8月31日*
