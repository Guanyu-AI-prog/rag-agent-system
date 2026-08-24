# 纯 Python Agent 迁移指南

## 概述

`taocan_agent_pure.py` 是 `taocan_agent.py` 的纯 Python 实现，移除了 LangChain 依赖，保留了所有核心功能。

## 主要变化

### 1. 移除的依赖
- `langchain-classic`
- `langchain-core`
- `langchain-openai`

### 2. 保留的依赖
- `openai` - 直接调用 OpenAI 兼容 API
- `simpleeval` - 安全数学计算
- 其他原有依赖

### 3. 架构变化

| 原版 (LangChain) | 纯 Python 版 |
|------------------|--------------|
| `AgentExecutor` | `PurePythonAgent` 类 |
| `@tool` 装饰器 | `ToolRegistry` 注册系统 |
| `ChatOpenAI` | 直接使用 `OpenAI` 客户端 |
| `ChatPromptTemplate` | 手动构建消息列表 |
| `MessagesPlaceholder` | 手动插入历史消息 |

## 使用方法

### 安装依赖

```bash
pip install -r requirements_pure.txt
```

### 运行

```bash
# 交互模式
python taocan_agent_pure.py

# 单次查询
python taocan_agent_pure.py "59套餐多少流量"
```

## 工具调用机制

### 原版 (LangChain)
使用 OpenAI 原生的 function calling，由 LangChain 自动处理。

### 纯 Python 版
支持两种方式：

1. **OpenAI 原生 tool_calls**（推荐）
   - 如果模型支持 tool_calls，自动使用原生方式

2. **正则匹配**（兼容模式）
   - 对于不支持 tool_calls 的模型
   - 通过正则匹配 `<tool_call>` 标签解析工具调用

示例输出格式：
```
<tool_call>
{"name": "套餐知识查询", "arguments": {"query": "59套餐包含多少流量"}}
</tool_call>
```

## 核心组件

### ToolRegistry
工具注册表，管理所有可用工具：

```python
_registry = ToolRegistry()
_registry.register(
    name="工具名",
    description="工具描述",
    func=工具函数,
    parameters={...}
)
```

### PurePythonAgent
纯 Python Agent 实现：

```python
agent = PurePythonAgent(max_iterations=5)
answer = agent.run("用户问题", verbose=True)
```

## 功能对比

| 功能 | 原版 | 纯 Python 版 |
|------|------|--------------|
| 套餐知识查询 | ✅ | ✅ |
| 套餐对比直达 | ✅ | ✅ |
| 费用计算器 | ✅ | ✅ |
| 套餐统计 | ✅ | ✅ |
| 对话历史 | ✅ | ✅ |
| 结果缓存 | ✅ | ✅ |
| 限流重试 | ✅ | ✅ |
| fast-path 优化 | ✅ | ✅ |
| 降级机制 | ✅ | ✅ |

## 注意事项

1. **模型兼容性**
   - 推荐使用支持 function calling 的模型
   - 对于不支持的模型，会自动降级到正则匹配模式

2. **性能**
   - 纯 Python 版可能比 LangChain 版更轻量
   - 但需要手动处理更多细节

3. **错误处理**
   - 保留了原有的错误处理和降级机制
   - 日志格式保持一致

## 配置

使用原有的 `config.py`，所有配置项保持不变：

```python
# .env 文件
LLM_API_KEY=your_api_key
LLM_API_BASE=https://api.example.com/v1
LLM_MODEL=model_name
```

## 测试

```bash
# 测试单次查询
python taocan_agent_pure.py "59套餐包含多少流量"

# 测试对比功能
python taocan_agent_pure.py "对比59和129套餐"

# 测试计算功能
python taocan_agent_pure.py "59套餐一年多少钱"
```

## 故障排除

### 问题：工具调用失败
- 检查模型是否支持 function calling
- 查看日志中的错误信息

### 问题：正则匹配失败
- 确保 LLM 输出包含 `<tool_call>` 标签
- 检查 JSON 格式是否正确

### 问题：API 调用失败
- 检查 API Key 和 API Base 配置
- 查看限流重试日志
