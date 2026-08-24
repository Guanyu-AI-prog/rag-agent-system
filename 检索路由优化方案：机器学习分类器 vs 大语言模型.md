# 检索路由优化方案：机器学习分类器 vs 大语言模型

## 📋 文档信息
- **项目**：dx_agent.py 检索路由优化
- **创建日期**：2026年8月24日
- **目的**：对比分析两种检索路由优化方案，为技术选型提供参考

---

## 1. 项目背景与当前方案分析

### 1.1 当前检索路由机制

```python
def _classify_query(query: str) -> str:
    """基于规则的查询分类"""
    # 对比关键词检测
    comparison_keywords = ["对比", "比较", "区别", "差异", "分别", "各是"]
    is_comparison = any(kw in query for kw in comparison_keywords)
    
    # 隐含对比模式检测
    if not is_comparison and re.search(r'(\d+).+?和.+?(\d+)', query):
        is_comparison = True
    
    # 套餐档位提取
    numbers = re.findall(r'(\d+)\s*(?:元|套餐)', query)
    
    # 规则决策树
    if is_comparison and len(unique) >= 2:
        return "comparison"
    elif re.search(r'[\+\-\*\/]', query):
        return "complex"
    elif any(kw in query for kw in ["推荐", "合适", "怎么选"]):
        return "complex"
    else:
        return "simple"
```

### 1.2 当前方案局限性

| 问题类型 | 具体表现 | 影响范围 |
|---------|---------|---------|
| **规则覆盖不全** | 无法识别"这两个套餐哪个对打游戏更好？"这类隐含对比 | 约15%的查询 |
| **关键词硬编码** | 需要人工维护关键词列表，难以适应新表达 | 持续维护成本 |
| **语义理解不足** | 无法理解"套餐A的流量是套餐B的两倍"的对比含义 | 复杂查询场景 |
| **无法处理歧义** | "哪个套餐更好"可能是推荐也可能是对比 | 约10%的查询 |

---

## 2. 方案一：传统机器学习分类器

### 2.1 技术原理

**核心思想**：通过标注数据训练模型，自动学习查询模式与路由类别的映射关系。

```
训练阶段：
标注数据 → 特征提取 → 模型训练 → 学习规律
   ↓           ↓           ↓
查询文本     TF-IDF向量   分类模型

预测阶段：
新查询 → 特征提取 → 模型预测 → 路由决策
   ↓           ↓           ↓
原始文本     向量表示     simple/comparison/complex
```

### 2.2 实现步骤

#### **步骤1：数据收集与标注**
```python
# 收集历史查询日志
historical_queries = [
    "59套餐包含多少流量？",
    "对比59和99套餐",
    "推荐一个合适的套餐",
    # ... 成百上千条
]

# 人工标注（可使用众包平台）
labels = [
    "simple",      # 事实查询
    "comparison",  # 对比查询
    "complex",     # 推荐查询
    # ...
]
```

#### **步骤2：特征工程**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 中文分词
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# TF-IDF特征提取
vectorizer = TfidfVectorizer(
    tokenizer=chinese_tokenizer,
    max_features=1000,      # 最多1000个特征词
    ngram_range=(1, 2),     # 一元和二元词组
    min_df=2,               # 至少出现2次
    max_df=0.95             # 最多95%的文档中出现
)

X = vectorizer.fit_transform(historical_queries)
```

#### **步骤3：模型训练与评估**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# 训练模型
classifier = RandomForestClassifier(
    n_estimators=100,       # 100棵决策树
    max_depth=10,           # 最大深度
    random_state=42,
    n_jobs=-1               # 使用所有CPU核心
)
classifier.fit(X_train, y_train)

# 评估模型
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# 交叉验证
cv_scores = cross_val_score(classifier, X, labels, cv=5)
print(f"交叉验证准确率: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
```

#### **步骤4：部署与预测**
```python
import joblib

# 保存模型
joblib.dump(classifier, 'query_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# 加载模型（生产环境）
classifier = joblib.load('query_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_route(query: str) -> tuple:
    """预测查询路由"""
    # 特征提取
    X = vectorizer.transform([query])
    
    # 预测类别
    prediction = classifier.predict(X)[0]
    
    # 预测概率（置信度）
    probabilities = classifier.predict_proba(X)[0]
    confidence = max(probabilities)
    
    return prediction, confidence

# 使用示例
query = "这两个套餐哪个更划算？"
route, confidence = predict_route(query)
print(f"查询: {query}")
print(f"预测路由: {route}, 置信度: {confidence:.2%}")
```

### 2.3 优化策略

#### **特征工程优化**
```python
# 添加额外特征
def extract_enhanced_features(query: str) -> dict:
    """提取增强特征"""
    features = {
        # 基础统计特征
        'query_length': len(query),
        'word_count': len(query.split()),
        'number_count': len(re.findall(r'\d+', query)),
        
        # 关键词特征
        'has_comparison_keyword': int(any(kw in query for kw in comparison_keywords)),
        'has_recommend_keyword': int(any(kw in query for kw in recommend_keywords)),
        'has_calculation_keyword': int(any(kw in query for kw in ['计算', '年费', '差价'])),
        
        # 模式特征
        'has_number_and_keyword': int(bool(re.search(r'\d+\s*(元|套餐)', query))),
        'has_comparison_pattern': int(bool(re.search(r'(\d+).+?和.+?(\d+)', query))),
        
        # 句法特征
        'has_question_mark': int('?' in query or '？' in query),
        'has_exclamation': int('!' in query or '！' in query),
    }
    return features

# 组合TF-IDF特征和手工特征
from scipy.sparse import hstack

tfidf_features = vectorizer.transform(queries)
manual_features = [extract_enhanced_features(q) for q in queries]
manual_features_array = np.array([list(f.values()) for f in manual_features])

combined_features = hstack([tfidf_features, manual_features_array])
```

#### **模型调优**
```python
from sklearn.model_selection import GridSearchCV

# 参数网格搜索
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳准确率: {grid_search.best_score_:.2%}")

# 使用最佳模型
best_classifier = grid_search.best_estimator_
```

---

## 3. 方案二：大语言模型分类器

### 3.1 技术原理

**核心思想**：利用大语言模型的语义理解能力，通过提示工程实现零样本或少样本分类。

```
传统机器学习：
输入 → 特征工程 → 模型 → 输出
       (人工设计)

大语言模型：
输入 → 预训练模型 → 输出
       (自动理解语义)
```

### 3.2 实现方式

#### **方式1：直接提示分类（零样本）**
```python
import openai

def classify_with_llm_zero_shot(query: str) -> tuple:
    """使用大语言模型进行零样本分类"""
    prompt = f"""请判断以下用户查询属于哪一类，只输出类别名称。

查询：{query}

类别说明：
- simple：简单事实查询，如"59套餐包含多少流量"、"价格是多少"
- comparison：对比查询，如"对比59和99套餐"、"哪个更划算"、"A和B的区别"
- complex：复杂推理查询，如"推荐一个套餐"、"怎么选择"、"年费计算"

类别："""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 低温度，输出更确定
            max_tokens=10
        )
        
        category = response.choices[0].message.content.strip().lower()
        
        # 验证类别有效性
        valid_categories = ['simple', 'comparison', 'complex']
        if category not in valid_categories:
            category = 'simple'  # 默认回退
        
        return category, 0.9  # 大语言模型通常置信度较高
        
    except Exception as e:
        logger.error(f"LLM分类失败: {e}")
        return 'simple', 0.5  # 失败时回退到规则方法
```

#### **方式2：少样本学习（Few-shot）**
```python
def classify_with_llm_few_shot(query: str, examples: list = None) -> tuple:
    """使用大语言模型进行少样本分类"""
    if examples is None:
        examples = [
            ("59套餐包含多少流量？", "simple"),
            ("对比59和99套餐", "comparison"),
            ("推荐一个合适的套餐", "complex"),
            ("实付39和实付89区别", "comparison"),
            ("年费怎么计算？", "complex"),
            ("哪个套餐性价比最高？", "complex"),
        ]
    
    # 构建少样本提示
    examples_text = "\n".join([f"查询：{q}\n类别：{l}" for q, l in examples])
    
    prompt = f"""请判断以下用户查询属于哪一类。

类别说明：
- simple：简单事实查询，获取具体信息
- comparison：对比查询，比较多个选项
- complex：复杂推理查询，需要推荐或计算

示例：
{examples_text}

现在请判断：
查询：{query}
类别："""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        category = response.choices[0].message.content.strip().lower()
        
        # 解析可能的额外说明
        if 'simple' in category:
            category = 'simple'
        elif 'comparison' in category:
            category = 'comparison'
        elif 'complex' in category:
            category = 'complex'
        else:
            category = 'simple'
        
        return category, 0.95
        
    except Exception as e:
        logger.error(f"LLM少样本分类失败: {e}")
        return 'simple', 0.5
```

#### **方式3：结构化输出分类**
```python
def classify_with_structured_output(query: str) -> dict:
    """使用大语言模型进行结构化分类"""
    prompt = f"""分析以下用户查询，返回JSON格式的分类结果。

查询：{query}

请严格按照以下JSON格式输出：
{{
    "category": "simple" 或 "comparison" 或 "complex",
    "confidence": 0.0到1.0的置信度,
    "reasoning": "判断理由",
    "keywords": ["识别到的关键词"],
    "entities": ["识别到的实体，如套餐档位数字"]
}}

JSON输出："""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        # 解析JSON响应
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        return {
            'category': result.get('category', 'simple'),
            'confidence': result.get('confidence', 0.8),
            'reasoning': result.get('reasoning', ''),
            'keywords': result.get('keywords', []),
            'entities': result.get('entities', [])
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        # 尝试从文本中提取类别
        if 'comparison' in result_text:
            return {'category': 'comparison', 'confidence': 0.7}
        elif 'complex' in result_text:
            return {'category': 'complex', 'confidence': 0.7}
        else:
            return {'category': 'simple', 'confidence': 0.6}
            
    except Exception as e:
        logger.error(f"结构化分类失败: {e}")
        return {'category': 'simple', 'confidence': 0.5}
```

### 3.3 成本优化策略

#### **缓存机制**
```python
import hashlib
from functools import lru_cache

# 内存缓存
_query_cache = {}

def classify_with_cache(query: str) -> tuple:
    """带缓存的LLM分类"""
    # 生成缓存键
    cache_key = hashlib.md5(query.encode()).hexdigest()
    
    # 检查缓存
    if cache_key in _query_cache:
        return _query_cache[cache_key]
    
    # 调用LLM
    result = classify_with_llm_zero_shot(query)
    
    # 存储缓存（限制大小）
    if len(_query_cache) > 1000:
        # 清理最旧的缓存
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
    
    _query_cache[cache_key] = result
    return result
```

#### **批处理优化**
```python
def classify_batch_with_llm(queries: list) -> list:
    """批量分类优化"""
    # 将多个查询合并到一个提示中
    queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    
    prompt = f"""请判断以下每个查询的类别，返回JSON数组。

查询列表：
{queries_text}

类别：simple（事实查询）、comparison（对比查询）、complex（复杂推理）

返回格式：
[
    {{"query": "查询1", "category": "simple"}},
    {{"query": "查询2", "category": "comparison"}},
    ...
]"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        results = json.loads(response.choices[0].message.content)
        return [(r['query'], r['category'], 0.9) for r in results]
        
    except Exception as e:
        logger.error(f"批量分类失败: {e}")
        # 回退到单个分类
        return [(q, 'simple', 0.5) for q in queries]
```

---

## 4. 详细对比分析

### 4.1 核心维度对比

| 维度 | 传统机器学习分类器 | 大语言模型分类器 |
|------|-------------------|-----------------|
| **技术成熟度** | ⭐⭐⭐⭐⭐ 高度成熟 | ⭐⭐⭐⭐ 快速发展中 |
| **实现复杂度** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 较高 |
| **数据需求** | ⭐⭐⭐⭐⭐ 需要标注数据 | ⭐ 零样本/少样本 |
| **计算资源** | ⭐ CPU即可 | ⭐⭐⭐⭐⭐ 需要GPU或API |
| **响应速度** | ⭐⭐⭐⭐⭐ 毫秒级 | ⭐⭐ 秒级 |
| **语义理解** | ⭐⭐ 有限 | ⭐⭐⭐⭐⭐ 强大 |
| **可解释性** | ⭐⭐⭐⭐⭐ 高 | ⭐⭐ 较低 |
| **维护成本** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 较高 |
| **扩展能力** | ⭐⭐ 有限 | ⭐⭐⭐⭐⭐ 强大 |

### 4.2 性能对比实验设计

```python
# 实验设置
test_queries = [
    # simple类
    ("59套餐包含多少流量？", "simple"),
    ("129有宽带吗？", "simple"),
    ("价格是多少？", "simple"),
    
    # comparison类  
    ("对比59和99套餐", "comparison"),
    ("实付39和实付89区别", "comparison"),
    ("哪个套餐流量更多？", "comparison"),
    
    # complex类
    ("推荐一个合适的套餐", "complex"),
    ("年费怎么计算？", "complex"),
    ("性价比最高的是哪个？", "complex"),
    
    # 边界案例
    ("这两个套餐哪个对打游戏更好？", "comparison"),  # 隐含对比
    ("如果我经常看视频，选哪个？", "complex"),      # 隐含推荐
    ("套餐A的流量是套餐B的两倍", "comparison"),     # 数量对比
]

def evaluate_classifier(classifier_func, test_data):
    """评估分类器性能"""
    correct = 0
    total = len(test_data)
    
    for query, true_label in test_data:
        predicted_label, _ = classifier_func(query)
        if predicted_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# 评估规则方法
rule_accuracy = evaluate_classifier(rule_based_classify, test_queries)

# 评估机器学习方法
ml_accuracy = evaluate_classifier(ml_classify, test_queries)

# 评估大语言模型方法
llm_accuracy = evaluate_classifier(llm_classify, test_queries)

print(f"规则方法准确率: {rule_accuracy:.2%}")
print(f"机器学习准确率: {ml_accuracy:.2%}")
print(f"大语言模型准确率: {llm_accuracy:.2%}")
```

### 4.3 成本效益分析

#### **开发成本对比**
```python
# 传统机器学习开发成本
ml_development_cost = {
    '数据收集标注': '2-4周（人工标注1000-5000条）',
    '特征工程': '1-2周（设计和调试特征）',
    '模型训练调优': '1-2周（实验不同算法参数）',
    '系统集成部署': '1周（API开发、测试）',
    '总开发周期': '5-9周',
    '开发人员': '1-2名机器学习工程师'
}

# 大语言模型开发成本
llm_development_cost = {
    '提示工程': '1-2周（设计提示模板）',
    'API集成': '3-5天（调用接口开发）',
    '测试优化': '1周（测试和优化提示）',
    '总开发周期': '2-3周',
    '开发人员': '1名普通开发工程师'
}
```

#### **运行成本对比**
```python
# 假设每日查询量：10,000次
daily_queries = 10000

# 传统机器学习运行成本
ml_running_cost = {
    '服务器': '2核4G云服务器 ≈ ¥200/月',
    '维护': '偶尔模型更新 ≈ 2小时/月',
    '总成本': '¥200-300/月',
    '单次成本': '¥0.0001-0.0003'
}

# 大语言模型运行成本（GPT-3.5-turbo）
llm_running_cost = {
    'API调用': '10000次 × 500token × $0.002/1K = $100/天 ≈ ¥700/天',
    '月成本': '¥21,000/月',
    '单次成本': '¥0.07',
    '缓存优化后': '假设50%缓存命中，¥10,500/月'
}
```

---

## 5. 混合方案设计

### 5.1 三级分类架构

```python
class HybridQueryClassifier:
    """混合查询分类器"""
    
    def __init__(self):
        # 第一级：规则分类器（快速、免费）
        self.rule_classifier = RuleBasedClassifier()
        
        # 第二级：机器学习分类器（平衡）
        self.ml_classifier = MLClassifier()
        
        # 第三级：大语言模型分类器（强大、昂贵）
        self.llm_classifier = LLMClassifier()
        
        # 配置阈值
        self.ml_confidence_threshold = 0.8
        self.llm_confidence_threshold = 0.7
    
    def classify(self, query: str) -> tuple:
        """三级分类策略"""
        
        # 第一级：规则分类（处理80%常见情况）
        rule_result, rule_confidence = self.rule_classifier.classify(query)
        
        if rule_confidence > 0.9:
            # 规则分类置信度高，直接返回
            logger.debug(f"规则分类: {query} -> {rule_result}")
            return rule_result, rule_confidence
        
        # 第二级：机器学习分类（处理15%中等复杂情况）
        ml_result, ml_confidence = self.ml_classifier.classify(query)
        
        if ml_confidence > self.ml_confidence_threshold:
            logger.debug(f"机器学习分类: {query} -> {ml_result}")
            return ml_result, ml_confidence
        
        # 第三级：大语言模型分类（处理5%复杂情况）
        llm_result, llm_confidence = self.llm_classifier.classify(query)
        
        logger.debug(f"大语言模型分类: {query} -> {llm_result}")
        return llm_result, llm_confidence
```

### 5.2 智能路由优化

```python
class SmartQueryRouter:
    """智能查询路由器"""
    
    def __init__(self):
        self.classifier = HybridQueryClassifier()
        self.route_handlers = {
            'simple': self._handle_simple,
            'comparison': self._handle_comparison,
            'complex': self._handle_complex
        }
        
        # 性能监控
        self.metrics = {
            'total_queries': 0,
            'rule_hits': 0,
            'ml_hits': 0,
            'llm_hits': 0,
            'route_distribution': {'simple': 0, 'comparison': 0, 'complex': 0}
        }
    
    def route(self, query: str) -> dict:
        """路由查询"""
        self.metrics['total_queries'] += 1
        
        # 分类查询
        route_type, confidence = self.classifier.classify(query)
        
        # 更新路由分布
        self.metrics['route_distribution'][route_type] += 1
        
        # 根据分类结果选择处理器
        handler = self.route_handlers.get(route_type, self._handle_simple)
        result = handler(query)
        
        return {
            'query': query,
            'route': route_type,
            'confidence': confidence,
            'result': result,
            'handler': handler.__name__
        }
    
    def _handle_simple(self, query: str) -> str:
        """处理简单查询"""
        self.metrics['rule_hits'] += 1
        # 调用简单RAG查询
        return simple_rag_query(query)
    
    def _handle_comparison(self, query: str) -> str:
        """处理对比查询"""
        self.metrics['ml_hits'] += 1
        # 调用对比快速路径
        return comparison_fast_path(query)
    
    def _handle_complex(self, query: str) -> str:
        """处理复杂查询"""
        self.metrics['llm_hits'] += 1
        # 调用Agent多步推理
        return agent_query(query)
    
    def get_performance_report(self) -> dict:
        """获取性能报告"""
        total = self.metrics['total_queries']
        if total == 0:
            return {}
        
        return {
            '总查询数': total,
            '路由分布': self.metrics['route_distribution'],
            '规则命中率': f"{self.metrics['rule_hits']/total:.2%}",
            '机器学习命中率': f"{self.metrics['ml_hits']/total:.2%}",
            '大语言模型命中率': f"{self.metrics['llm_hits']/total:.2%}",
            '平均置信度': self._calculate_average_confidence()
        }
```

---

## 6. 实施建议与路线图

### 6.1 阶段式实施计划

#### **第一阶段：规则优化（1-2周）**
```python
# 1. 扩展规则库
enhanced_rules = {
    'comparison': {
        'keywords': ['对比', '比较', '区别', '差异', '分别', '各是', '相比', '对照'],
        'patterns': [
            r'(\d+).+?和.+?(\d+)',
            r'哪个更.*',
            r'.*vs.*',
            r'.*还是.*好'
        ]
    },
    'complex': {
        'keywords': ['推荐', '合适', '怎么选', '选哪个', '建议', '适合', '划算', '性价比'],
        'patterns': [
            r'如果我.*',
            r'我应该.*',
            r'怎样.*最合适'
        ]
    }
}

# 2. 添加置信度计算
def enhanced_rule_classify(query: str) -> tuple:
    """增强规则分类"""
    scores = {'simple': 0, 'comparison': 0, 'complex': 0}
    
    # 关键词匹配
    for category, rules in enhanced_rules.items():
        for keyword in rules['keywords']:
            if keyword in query:
                scores[category] += 0.3
    
    # 模式匹配
    for category, rules in enhanced_rules.items():
        for pattern in rules['patterns']:
            if re.search(pattern, query):
                scores[category] += 0.5
    
    # 选择最高分类别
    best_category = max(scores, key=scores.get)
    confidence = min(scores[best_category], 1.0)
    
    return best_category, confidence
```

#### **第二阶段：机器学习集成（2-4周）**
```python
# 1. 数据收集管道
class DataCollectionPipeline:
    def __init__(self):
        self.query_log = []
        self.annotation_queue = []
    
    def log_query(self, query: str, user_action: str):
        """记录用户查询和行为"""
        self.query_log.append({
            'query': query,
            'timestamp': datetime.now(),
            'user_action': user_action,
            'session_id': get_session_id()
        })
    
    def select_for_annotation(self, n_samples: int = 100):
        """选择需要标注的样本"""
        # 选择边界案例和不确定性高的样本
        selected = []
        for log in self.query_log[-1000:]:  # 最近1000条
            if self._is_borderline_case(log['query']):
                selected.append(log)
        
        return selected[:n_samples]

# 2. 模型训练流水线
class ModelTrainingPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier()
    
    def train(self, training_data: list):
        """训练模型"""
        queries = [d['query'] for d in training_data]
        labels = [d['label'] for d in training_data]
        
        # 特征提取
        X = self.vectorizer.fit_transform(queries)
        
        # 训练模型
        self.classifier.fit(X, labels)
        
        # 评估模型
        scores = cross_val_score(self.classifier, X, labels, cv=5)
        print(f"模型准确率: {scores.mean():.2%} ± {scores.std():.2%}")
        
        return scores.mean()
    
    def save_model(self, path: str):
        """保存模型"""
        joblib.dump(self.classifier, f'{path}/classifier.pkl')
        joblib.dump(self.vectorizer, f'{path}/vectorizer.pkl')
```

#### **第三阶段：大语言模型集成（1-2周）**
```python
# 1. LLM分类服务
class LLMClassificationService:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.cache = {}
    
    def classify(self, query: str) -> dict:
        """LLM分类"""
        # 检查缓存
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 构建提示
        prompt = self._build_prompt(query)
        
        # 调用API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            result = self._parse_response(response.choices[0].message.content)
            
            # 缓存结果
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"LLM分类失败: {e}")
            return {'category': 'simple', 'confidence': 0.5, 'error': str(e)}
    
    def _build_prompt(self, query: str) -> str:
        """构建提示模板"""
        return f"""分析以下查询，返回JSON格式结果：

查询：{query}

要求：
1. 分类为：simple（事实查询）、comparison（对比查询）、complex（复杂推理）
2. 提供置信度（0-1）
3. 给出判断理由

JSON格式：
{{"category": "...", "confidence": 0.9, "reasoning": "..."}}

JSON输出："""
```

### 6.2 监控与迭代

```python
class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {
            'latency': [],          # 响应时间
            'accuracy': [],         # 准确率
            'confidence': [],       # 置信度
            'route_distribution': defaultdict(int),  # 路由分布
            'error_rate': []        # 错误率
        }
    
    def log_prediction(self, query: str, predicted: str, actual: str, 
                      latency: float, confidence: float):
        """记录预测结果"""
        self.metrics['latency'].append(latency)
        self.metrics['confidence'].append(confidence)
        self.metrics['route_distribution'][predicted] += 1
        
        # 计算准确率
        is_correct = predicted == actual
        self.metrics['accuracy'].append(is_correct)
    
    def generate_report(self) -> dict:
        """生成性能报告"""
        return {
            '平均响应时间': f"{np.mean(self.metrics['latency']):.3f}s",
            '平均准确率': f"{np.mean(self.metrics['accuracy']):.2%}",
            '平均置信度': f"{np.mean(self.metrics['confidence']):.2%}",
            '路由分布': dict(self.metrics['route_distribution']),
            'P95响应时间': f"{np.percentile(self.metrics['latency'], 95):.3f}s"
        }
    
    def detect_drift(self, window_size: int = 1000) -> bool:
        """检测模型漂移"""
        if len(self.metrics['accuracy']) < window_size * 2:
            return False
        
        # 比较最近两个窗口的准确率
        recent = self.metrics['accuracy'][-window_size:]
        previous = self.metrics['accuracy'][-window_size*2:-window_size]
        
        recent_acc = np.mean(recent)
        previous_acc = np.mean(previous)
        
        # 如果准确率下降超过5%，认为发生漂移
        return recent_acc < previous_acc - 0.05
```

---

## 7. 总结与推荐

### 7.1 方案选择指南

| 场景特征 | 推荐方案 | 理由 |
|---------|---------|------|
| **查询模式固定，规则明确** | 传统机器学习 | 成本低，效果好 |
| **查询复杂多变，语义重要** | 大语言模型 | 理解能力强 |
| **需要快速上线** | 大语言模型 | 开发周期短 |
| **预算有限** | 传统机器学习 | 运行成本低 |
| **追求最高准确率** | 混合方案 | 综合优势 |

### 7.2 最终推荐

#### **短期（1-2个月）**
```python
# 推荐方案：增强规则 + 机器学习混合
def recommend_short_term():
    return {
        '方案': '增强规则 + 机器学习混合',
        '理由': '平衡成本与效果，快速见效',
        '实施': [
            '1. 优化现有规则库',
            '2. 收集标注数据',
            '3. 训练简单分类器',
            '4. 逐步替换规则'
        ]
    }
```

#### **中期（3-6个月）**
```python
# 推荐方案：三级混合架构
def recommend_medium_term():
    return {
        '方案': '三级混合架构',
        '理由': '兼顾性能、成本和效果',
        '架构': {
            '第一级': '规则分类器（80%查询）',
            '第二级': '机器学习分类器（15%查询）',
            '第三级': '大语言模型分类器（5%查询）'
        }
    }
```

#### **长期（6个月以上）**
```python
# 推荐方案：大语言模型为主 + 规则辅助
def recommend_long_term():
    return {
        '方案': '大语言模型为主',
        '理由': '随着成本下降和模型提升，大语言模型将成为主流',
        '前提条件': [
            '1. API成本显著下降',
            '2. 模型速度提升',
            '3. 本地化部署成熟'
        ]
    }
```

### 7.3 关键成功因素

1. **数据质量**：无论是机器学习还是大语言模型，高质量的数据都是关键
2. **持续优化**：建立反馈循环，不断改进分类效果
3. **成本控制**：合理使用缓存和批量处理，控制API调用成本
4. **监控告警**：实时监控系统性能，及时发现和解决问题

---

## 附录A：完整代码示例

```python
# 完整的混合分类器实现
class ProductionQueryClassifier:
    """生产环境查询分类器"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 初始化各组件
        self.rule_classifier = self._init_rule_classifier()
        self.ml_classifier = self._init_ml_classifier()
        self.llm_service = self._init_llm_service()
        
        # 性能监控
        self.monitor = PerformanceMonitor()
        
        # 缓存配置
        self.cache = LRUCache(maxsize=config.get('cache_size', 1000))
    
    def classify(self, query: str) -> dict:
        """分类查询"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"classify:{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.monitor.log_cache_hit()
            return cached_result
        
        # 三级分类
        result = self._three_level_classify(query)
        
        # 记录性能
        latency = time.time() - start_time
        self.monitor.log_prediction(query, result['category'], latency)
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    def _three_level_classify(self, query: str) -> dict:
        """三级分类逻辑"""
        # 第一级：规则分类
        rule_result, rule_confidence = self.rule_classifier.classify(query)
        if rule_confidence > self.config.get('rule_threshold', 0.9):
            return {
                'category': rule_result,
                'confidence': rule_confidence,
                'method': 'rule',
                'latency': 0.001
            }
        
        # 第二级：机器学习分类
        ml_result, ml_confidence = self.ml_classifier.classify(query)
        if ml_confidence > self.config.get('ml_threshold', 0.8):
            return {
                'category': ml_result,
                'confidence': ml_confidence,
                'method': 'ml',
                'latency': 0.01
            }
        
        # 第三级：大语言模型分类
        llm_result = self.llm_service.classify(query)
        return {
            'category': llm_result['category'],
            'confidence': llm_result['confidence'],
            'method': 'llm',
            'latency': 1.0,
            'reasoning': llm_result.get('reasoning', '')
        }
```

## 附录B：性能基准测试

```python
# 性能测试脚本
def run_benchmark():
    """运行性能基准测试"""
    test_cases = load_test_cases()
    
    classifiers = {
        '规则方法': RuleBasedClassifier(),
        '机器学习': MLClassifier(),
        '大语言模型': LLMClassifier(),
        '混合方法': HybridClassifier()
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        print(f"测试 {name}...")
        
        # 预热
        for _ in range(10):
            classifier.classify("测试查询")
        
        # 正式测试
        latencies = []
        correct = 0
        
        for query, expected in test_cases:
            start = time.time()
            predicted, confidence = classifier.classify(query)
            latency = time.time() - start
            
            latencies.append(latency)
            if predicted == expected:
                correct += 1
        
        results[name] = {
            '准确率': f"{correct/len(test_cases):.2%}",
            '平均延迟': f"{np.mean(latencies):.3f}s",
            'P95延迟': f"{np.percentile(latencies, 95):.3f}s",
            'P99延迟': f"{np.percentile(latencies, 99):.3f}s"
        }
    
    return results
```

---

**文档版本**：v1.0  
**最后更新**：2026年8月24日  
**维护人员**：技术团队