# LangChain RAG代码实现

本目录包含基于LangChain的RAG系统完整实现，可直接用于生产环境。

## 📁 目录结构

```
code/
├── config.py                # 系统配置管理
├── workflow_langchain.py    # 核心RAG工作流
├── api.py                   # FastAPI服务接口
├── build_vectors.py         # 向量库构建脚本
├── test_system.py           # 系统测试脚本
├── start.sh                 # 快速启动脚本
├── requirements.txt         # Python依赖
├── .env.example             # 环境变量示例
├── README.md                # 本文件
└── data/                    # 示例知识库数据
    └── sample_telecom.txt   # 电信套餐示例数据
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入代码目录
cd code/

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置文件，填入你的SiliconFlow API密钥
# Linux/Mac: nano .env
# Windows: notepad .env
```

**必需配置：**
```env
SILICONFLOW_API_KEY=sk-your-siliconflow-api-key-here
LLM_MODEL=THUDM/GLM-4-9B-0414
EMBED_MODEL=BAAI/bge-large-zh-v1.5
```

### 3. 构建向量数据库

```bash
# 方法1：使用启动脚本
./start.sh build

# 方法2：直接运行脚本
python build_vectors.py --force
```

### 4. 启动API服务

```bash
# 方法1：使用启动脚本
./start.sh run

# 方法2：直接运行
python api.py
```

### 5. 测试系统

```bash
# 方法1：使用启动脚本测试
./start.sh test

# 方法2：运行测试脚本
python test_system.py

# 方法3：命令行查询
python workflow_langchain.py "有哪些套餐？"
```

## 🔧 核心组件说明

### 1. config.py - 配置管理
- 统一管理所有配置项
- 支持环境变量和默认值
- 包含配置验证功能

### 2. workflow_langchain.py - RAG工作流
- 文档加载和分割
- 向量化和存储
- 检索和生成
- 支持批量查询

### 3. api.py - API服务
- RESTful接口
- 内存缓存（5分钟TTL）
- 跨域支持
- 异步处理

### 4. build_vectors.py - 向量库构建
- 支持增量更新
- 文档分割配置
- 元数据生成

## 📊 性能优化建议

### 4核8GB服务器配置
```env
MAX_WORKERS=2           # 线程池大小
CACHE_TTL=300          # 缓存时间（秒）
CHUNK_SIZE=500         # 文本分块大小
RETRIEVAL_K=3          # 检索文档数量
```

### 模型选择策略
```env
# 快速响应（推荐客服场景）
LLM_MODEL=THUDM/GLM-4-9B-0414  # 响应时间：4秒

# 高质量响应（复杂分析）
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash  # 响应时间：25-33秒
```

## 🧪 API接口文档

启动服务后访问：http://localhost:8000/docs

### 主要接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 系统统计 |
| `/query` | POST | 单个查询 |
| `/batch_query` | POST | 批量查询 |
| `/cache/clear` | POST | 清空缓存 |

### 查询示例：

```bash
# 单个查询
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "有哪些套餐？"}'

# 批量查询
curl -X POST "http://localhost:8000/batch_query" \
  -H "Content-Type: application/json" \
  -d '["有哪些套餐？", "流量怎么算？"]'
```

## 🐛 常见问题

### Q: 启动时报错"SILICONFLOW_API_KEY未设置"
**A:** 请确保已创建`.env`文件并填入有效的API密钥。

### Q: 查询返回"向量库未初始化"
**A:** 请先运行`python build_vectors.py --force`构建向量库。

### Q: 响应时间太长（超过30秒）
**A:** 1. 检查是否使用了正确的LLM模型
   2. 考虑启用缓存（默认已启用）
   3. 检查服务器网络连接

### Q: 如何添加新文档？
**A:** 1. 将.txt文件放入`data/`目录
   2. 运行`python build_vectors.py`
   3. 重启API服务

### Q: 如何集成到钉钉机器人？
**A:** 参考`chapters/08-dingtalk-deployment.md`中的部署指南。

## 📈 监控和日志

### 查看系统状态
```bash
# 查看缓存状态
curl http://localhost:8000/cache/stats

# 查看系统统计
curl http://localhost:8000/stats
```

### 日志配置
在`.env`中设置：
```env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## 🔄 更新和维护

### 更新代码
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### 重建向量库
```bash
python build_vectors.py --force
```

### 清理缓存
```bash
curl -X POST "http://localhost:8000/cache/clear"
```

## 📞 技术支持

遇到问题时：
1. 查看本文档的常见问题
2. 检查`chapters/`目录中的相关文档
3. 查看API文档：http://localhost:8000/docs
4. 检查日志文件：`logs/`目录

## 📜 许可证

本代码基于MIT许可证开源。

---

*最后更新：2026年5月20日*