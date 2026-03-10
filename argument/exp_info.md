## 实验参数与配置清单 (Experimental Setup)

### 1. 垂直领域 (DOMAIN)

* **领域**：金融股票

### 2. 数据集 (DATASET)

* **黄金测试集 (GOLDEN_TEST)**
* **来源**：金融AI助手真实用户问题
* **标签分布**：总共 11 个意图标签
* **规模**：354 个问题（包含 62 个边界困难样本）


* **历史客服日志 (HISTORICAL_LOGS)**
* **来源**：金融AI助手真实用户问题
* **时间跨度**：从三个交易日的历史记录中提取
* **规模**：1500 个问题



### 3. 智能体白盒信息 (AGENT_INFO)

* **数据状态**：当前暂无数据，预计格式如下。
* **Schema 示例**：

```json
{
  "name": "新闻智能体",
  "description": "检索宏观政策、市场主线、热门话题等具有强时效性的资讯。",
  "parameters": {}
}

```

### 4. 基线方法配置 (BASELINE_DETAIL)

* **INTENT_GPT 方法**
* **模型配置**：qwen2.5-32b
* **Prompt 类型**：基于训练样本自动生成的 Prompt，每个类别的 few-shot 示例数量设置为 5 个。
* **自动生成 Prompt 模板**：


```text
You are a helpful assistant and an expert in natural language processing and specialize in the task of intent detection.

Your task is to analyze the given examples and provide a detailed prompt that will help an AI language model to effectively solve the task of open set intent discovery.

Based on the following training examples, create a comprehensive prompt that includes:
1. Clear guidelines for intent classification
2. Tips for handling ambiguous cases
3. Instructions for creating new intents when necessary
4. Best practices for reusing existing intents

EXAMPLES:

{train_examples}

You must respond using the following format:    
PROMPT: <your detailed prompt for intent detection>

```


* **原始人工描述方法**
* **来源**：由业务人员根据真实用户问题，经过多次手动迭代总结获得。
* **表现**：目前的分类准确率高于 INTENT_GPT 根据 few-shot 自动构建的 Prompt。



### 5. 基座模型 (BACKBONE_MODEL)

* **模型配置**：qwen2.5-32b（用于执行分类推理）

### 6. 计算资源 (COMPUTE_RESOURCES)

* **GPU 硬件**：4 × NVIDIA L20 (48GB 显存)
* **底层环境**：Driver Version 570.86.15, CUDA 12.8

### 7. 演化变异器模型 (OPTIMIZER_LLM)

* **模型配置**：在前期人工构建时，采用 **Gemini Pro** 作为核心的文本特征提取与边界描述优化器。
