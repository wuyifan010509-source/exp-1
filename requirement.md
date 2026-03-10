### 📦 核心实验工程架构 (Project Structure)

建议您的 Python 项目目录结构如下：

```text
intent_routing_experiment/
├── data/
│   ├── golden_dataset.json      # 黄金测试集
│   ├── unlabeled_logs.json      # 未标注的历史客服日志
│   └── agent_whitebox_info.json # 智能体的白盒配置信息(API, RAG元数据)
├── src/
│   ├── 1_zero_shot_init.py      # 白盒特征蒸馏模块
│   ├── 2_margin_sampling.py     # 边界数据挖掘模块
│   ├── 3_prompt_optimizer.py    # 结构化槽位变异模块 (遗传/坐标上升)
│   ├── 4_eval_pipeline.py       # 自动化评估与指标计算模块
│   └── router_engine.py         # 意图分类器本体 (用于被调用)
└── main_pipeline.py             # 实验主流程编排脚本

```

---

### 🛠️ 模块级代码需求与 TODO 清单

#### 模块一：数据结构与定义层 (Data Models)

**代码需求：** 需要严谨的数据结构来承载 $P=\{C,B,R\}$ 以及测试用例，保证各个节点之间数据传递的规范性，建议使用 `Pydantic`。
**TODO 清单：**

* [ ] **定义 `AgentProfile` 类**：包含 `agent_id`, `internal_prompt`, `tool_schema`, `rag_metadata`。
* [ ] **定义 `StructuredPrompt` 类**：包含 `C`, `B`, `R` 三个字符串属性，以及一个自动计算总长度并校验是否 $\le 200$ 字的验证器（Validator）。
* [ ] **定义 `TestCase` 类**：包含 `query`, `expected_agent`, `is_boundary` (布尔值) 等字段。

#### 模块二：零样本冷启动初始化 (1_zero_shot_init.py)

**代码需求：** 将黑盒转白盒，不依赖测试集，直接通过规则和 LLM 摘要生成第一代描述 $P_0$。
**TODO 清单：**

* [ ] **编写 C 槽位提取器**：调用 LLM（配合 Prompt模板），输入几千字的 `internal_prompt`，要求 LLM 基于“信息瓶颈理论”压缩提取核心能力，输出 $C_0$（约 60 字）。
* [ ] **编写 R 槽位推导器（重点）**：写一段纯 Python 集合运算代码。获取全局工具集 `Set(Global)`，获取当前智能体工具集 `Set(Local)`。计算差集 `Set(Global) - Set(Local)`。将差集转化为自然语言 $R_0$（如：“不处理：查询天气、修改密码”）。
* [ ] **编写 B 槽位提取器**：解析 `rag_metadata` 的 JSON 节点（如 `category`, `time_range`），直接拼接成 $B_0$。
* [ ] **组装并导出**：将所有的 $P_0$ 持久化保存为实验的基线（Baseline）。

#### 模块三：边界数据挖掘 (2_margin_sampling.py)

**代码需求：** 实现 Margin Sampling 算法，从海量无标注日志中捞出“困难负样本”。
**TODO 清单：**

* [ ] **对接意图分类器**：编写一个函数，调用您的 Router Engine（如果是大模型，需开启 `logprobs` 参数获取各个意图的预测概率；如果是向量检索，获取 Top-K 相似度得分）。
* [ ] **编写裕度计算逻辑**：批量输入 `unlabeled_logs`，计算每个 Query 的 $Margin = P(Top_1) - P(Top_2)$。
* [ ] **硬样本截断与导出**：设置一个阈值（如 Margin < 0.05），将这些最纠结的 Queries 标记为 Bad Cases 并保存，作为下一模块变异的“饲料”。

#### 模块四：结构化槽位定向变异 (3_prompt_optimizer.py)

**代码需求：** 核心优化引擎，基于上一步的 Bad Cases，局部修改 $P=\{C,B,R\}$，并严格控制字数。
**TODO 清单：**

* [ ] **编写错误归因定位函数**：根据 Bad Case 的实际预测和期望预测，判断该修改哪个槽位（通常是修改对应智能体的 $R$ 槽位，或者目标智能体的 $B$ 槽位）。
* [ ] **编写 LLM 变异算子 (Mutator)**：构建一个 Prompt，喂给大模型：
* *输入*：原始槽位文本、Bad Case、冲突原因。
* *指令*：“请作为一个基因变异算子，提取该 Bad Case 的互斥特征，并补充到当前槽位中。你必须提供 3 个不同的变异版本。”


* [ ] **编写字数惩罚拦截器**：写一段逻辑，将变异后的槽位与不变的槽位拼接 ($C \oplus B \oplus R$)。计算总长度，如果 $> 200$，直接给该变异体打上极低的适应度分数（Fitness Penalty），将其淘汰。

#### 模块五：自动化评估流水线 (4_eval_pipeline.py)

**代码需求：** 实验的裁判员。自动化运行测试，并决定上一阶段的“变异”是否被采纳。
**TODO 清单：**

* [ ] **批量跑分引擎集成**：如果使用 `Promptfoo`，编写一个自动生成 YAML 配置文件的脚本；如果手写，使用一个 `for` 循环遍历“黄金测试集”拿到所有的预测结果。
* [ ] **指标计算器**：引入 `sklearn.metrics`，一键计算 `accuracy_score`, `precision_score`, `recall_score`。
* [ ] **混淆矩阵生成器**：使用 `sklearn.metrics.confusion_matrix` 计算矩阵，并编写一个过滤函数，专门提取“非对角线上数值最大的 Top-3 智能体组合”，打印为简报。
* [ ] **版本回退机制 (Accept/Reject)**：
* `IF` 整体准确率提升 `AND` 没有任何类别的召回率显著下降：覆盖旧版 $P$，保存为 $P_{i+1}$。
* `ELSE`：丢弃当前变异体，回退到 $P_i$ 继续演化。



---

### 💡 技术栈建议 (Tech Stack)

对于这种大模型提示词优化与测试流水线工程，以下技术栈组合是最轻量且强大的：

* **核心语言**：Python 3.10+
* **LLM 交互与抽象**：`LangChain` 或 `LiteLLM`（方便随时切换 OpenAI / 闭源大模型作变异算子）。
* **自动化测试**：`Promptfoo`（强推，极其适合做 Prompt A/B 测试） 或纯 Python `pytest`。
* **数据校验**：`Pydantic`（确保 $P=\{C,B,R\}$ 的结构完整和字数合法性）。
* **指标评估**：`scikit-learn`（现成的 Precision, Recall, Confusion Matrix 算法）。

**下一步建议：** 万事开头难。在这 5 个模块中，**您希望我先帮您把哪一个模块的具体 Python 代码骨架写出来？** 比如，我可以先为您编写 **模块二（推导工具差集生成 $R$ 槽位）** 或者 **模块三（Margin Sampling 算法）** 的核心代码。