# 基线描述数据格式说明

## 文件说明

- `manual_descriptions.json` - 人工撰写的智能体描述（纯文本，非结构化）
- `intentgpt_descriptions.json` - IntentGPT few-shot生成的描述（纯文本，非结构化）

## 粘贴格式

每个智能体一行，格式如下：

```json
{
  "选股智能体": "选股智能体是一个专业的股票筛选工具。它可以根据用户输入的各种条件（如市盈率、市净率、成交量、涨跌幅等技术指标，以及行业、概念、地域等基本面信息）从全市场5000多只股票中筛选出符合条件的股票列表。它还支持保存筛选条件、导出结果、设置预警等功能。需要注意的是，它只能进行筛选，不能对个股进行诊断分析，也不能预测未来走势，更不能给出买卖建议。"
}
```

## 注意事项

1. **纯文本格式**：直接粘贴自然语言描述，**不要加C/B/R标记**
2. **长度不限**：可以超过200字，代码会自动压缩
3. **JSON转义**：如果文本中有引号，需要转义（如 `\"`）或使用三引号
4. **UTF-8编码**：确保保存为UTF-8格式

## 代码使用方式

```python
from baselines import load_manual_baseline, load_intentgpt_baseline

# 加载人工描述（使用硬截断压缩到200字）
manual = load_manual_baseline(
    "data/baselines/manual_descriptions.json",
    compress_method="hard_truncate"
)

# 加载IntentGPT描述（使用硬截断压缩到200字）
intentgpt = load_intentgpt_baseline(
    "data/baselines/intentgpt_descriptions.json",
    compress_method="hard_truncate"
)

# 使用不同的压缩方法
from baselines import RandomCompression

compressor = RandomCompression(max_len=200)
text = "很长很长的描述..."

# 方案A：硬截断（直接切前200字）
compressed = compressor.hard_truncate(text)

# 方案B：随机截断（从随机位置取200字）
compressed = compressor.random_truncate(text)

# 方案C：随机句子采样（随机选完整句子）
compressed = compressor.sentence_sample(text)

# 方案D：均匀采样（等间隔抽字符）
compressed = compressor.uniform_sample(text)
```

## 12个智能体列表

按config.py中的顺序：
1. 选股类 / 选股智能体
2. 诊股类 / 诊股智能体
3. 预测类 / 预测智能体
4. 知识库类 / 知识库智能体
5. 新闻类 / 新闻智能体
6. 通用类 / 通用智能体
7. 推荐类 / 推荐智能体
8. 策略类 / 策略智能体
9. 指标查询类 / 指标查询智能体
10. 身份类 / 身份智能体
11. 分时图类 / 分时图智能体
12. K线图类 / K线图智能体

## 对比实验设计

| 实验 | 你的方法 | 基线方法 | 对比目的 |
|------|---------|----------|----------|
| Exp1 | 结构化C/B/R | 非结构化文本 | 验证结构化价值 |
| Exp2 | 白盒初始化 | 随机初始化 | 验证白盒价值 |
| Exp3 | 有演化 | 无演化(仅初始化) | 验证演化必要性 |
| Exp4 | 智能压缩 | 硬截断压缩 | 验证"结构意识"重要性 |

## 技术细节

**为什么基线用非结构化文本？**

IntentGPT等方法生成的是**自由文本**，没有C/B/R结构。我们的对比实验要证明：
1. **结构化 > 非结构化**（相同字数下）
2. **智能压缩 > 随机压缩**（结构意识的价值）

**压缩后的处理方式**

为了兼容现有的`ProfileSet`结构，非结构化文本会：
- 前80字 → `core_capability`
- 中间60字 → `boundary`  
- 后60字 → `rejection_scope`

这种**硬切分**正是基线的特点：**无结构意识**，只是机械分割。

而你的方法（C/B/R结构化）是**语义感知的分割**，这是核心区别。
