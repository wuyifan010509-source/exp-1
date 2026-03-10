# 意图分类优化实验 — 实现方案 (v2)

> 配合阅读：[walkthrough.md](file:///Users/lxy/.gemini/antigravity/brain/b73247f0-bf77-468a-bdbc-543f370792ad/walkthrough.md)（做成什么样算成功） | [design_rationale.md](file:///Users/lxy/.gemini/antigravity/brain/b73247f0-bf77-468a-bdbc-543f370792ad/design_rationale.md)（为什么这么做）

## 概述

在 `wyf-exp1/` 目录下搭建 **结构化描述 P={C,B,R} + 遗传算法演化** 的实验框架，部署在 **4×L20 服务器** 上运行。

## User Review Required

> [!IMPORTANT]
> **智能体白盒数据**：当前暂无数据。代码中先预留接口和 Mock 数据，待数据就绪后填入。

> [!IMPORTANT]
> **推理代价优化**：完整评估需 ~212K 次 qwen2.5-32b 推理。建议使用 **vLLM** 部署模型并启用批量推理，将评估子集缩小到 ~100 条以控制成本。详见 [walkthrough.md 3.4 节](file:///Users/lxy/.gemini/antigravity/brain/b73247f0-bf77-468a-bdbc-543f370792ad/walkthrough.md)。

> [!WARNING]
> **基线方法** DSPy/TextGrad/APE/LLMLingua 为独立第三方库，建议作为后续阶段单独接入。

---

## 实验参数绑定

以下参数来自 [exp_info.md](file:///Users/lxy/lxygit/paper/26master/wyf-exp1/exp_info.md)：

| 参数 | 值 |
|------|-----|
| 意图标签数 | 11 |
| 黄金测试集 | 354 条（含 62 条边界样本） |
| 无标注历史日志 | 1500 条（3个交易日） |
| 分类基座模型 | qwen2.5-32b（本地 4×L20） |
| 演化变异器 | Gemini Pro (API) |
| 字数限制 | ≤200 字/Agent |
| 基线 1 | INTENT_GPT (qwen2.5-32b + few-shot 5) |
| 基线 2 | 原始人工描述 |

---

## 项目目录结构

```
wyf-exp1/
├── requirement.md                    # 已有
├── exp_info.md                       # 已有
├── config.py                         # [NEW] 全局配置
├── requirements.txt                  # [NEW] Python 依赖
│
├── structured_profile/               # [NEW] 模块①：结构化描述数据建模
│   ├── __init__.py
│   └── profile.py
│
├── evaluation/                       # [NEW] 模块②：评估流水线
│   ├── __init__.py
│   ├── classifier.py                 # 意图分类器（调用 qwen2.5-32b）
│   ├── metrics.py                    # 指标计算
│   └── pipeline.py                   # 评估流水线编排
│
├── whitebox_init/                    # [NEW] 模块③：白盒初始化
│   ├── __init__.py
│   ├── tool_parser.py                # 工具签名 → C, R
│   ├── prompt_compressor.py          # System Prompt → 结构化压缩
│   └── metadata_extractor.py         # RAG 元数据 → B
│
├── evolution/                        # [NEW] 模块④：遗传算法演化
│   ├── __init__.py
│   ├── genetic_algorithm.py          # GA 核心引擎
│   ├── llm_mutator.py               # LLM 定向变异算子
│   └── margin_sampling.py            # 边界挖掘
│
├── baselines/                        # [NEW] 模块⑤：基线方法
│   ├── __init__.py
│   ├── manual_baseline.py            # 加载人工描述
│   ├── intent_gpt_baseline.py        # INTENT_GPT 复现
│   └── random_compress.py            # 随机压缩
│
├── experiments/                      # [NEW] 模块⑥：实验编排
│   ├── run_whitebox_init.py          # 阶段1：白盒初始化
│   ├── run_evolution.py              # 阶段2：遗传演化
│   ├── run_comparison.py             # 阶段3：对比实验
│   ├── run_ablation.py               # 阶段4：消融实验
│   └── visualize.py                  # 阶段5：可视化报告
│
├── data/                             # [NEW] 数据目录
│   ├── agents/
│   │   └── sample_agents.json        # 智能体定义 (Mock → 替换为真实数据)
│   ├── golden_test_set.jsonl         # 354 条黄金测试集
│   └── unlabeled_queries.jsonl       # 1500 条历史日志
│
├── results/                          # [NEW] 实验结果输出
│   └── .gitkeep
│
└── tests/                            # [NEW] 单元测试
    ├── test_profile.py
    ├── test_metrics.py
    └── test_ga.py
```

---

## Proposed Changes

### 模块① `structured_profile/profile.py`

`StructuredProfile` 数据类：

```python
@dataclass
class StructuredProfile:
    agent_name: str
    core_capability: str   # C: 核心能力
    boundary: str          # B: 处理边界
    rejection_scope: str   # R: 拒绝范围

    def to_prompt(self) -> str: ...          # "[核心能力] ... [处理边界] ... [拒绝范围] ..."
    def length(self) -> int: ...             # 中文字符计数
    def is_valid(self, max_len=200) -> bool  # 字数约束校验
    def mutate_slot(self, slot, text) -> 'StructuredProfile'  # 返回变异后的新个体
```

`ProfileSet`：管理 11 个智能体的 Profile 集合，支持序列化。

---

### 模块② `evaluation/`

#### `classifier.py` — 意图分类器

```python
class IntentClassifier(ABC):
    def classify(self, query: str, profiles: ProfileSet) -> ClassifyResult:
        """返回 predicted_agent + confidence_scores {agent: prob}"""

class QwenClassifier(IntentClassifier):
    """调用本地 qwen2.5-32b (通过 vLLM OpenAI-compatible API)"""
    def __init__(self, api_url="http://localhost:8000/v1"):
        ...
    def classify_batch(self, queries, profiles) -> List[ClassifyResult]:
        """批量推理，提升吞吐"""
```

关键设计：分类器通过 **vLLM 的 OpenAI-compatible API** 调用本地 qwen2.5-32b，支持批量推理。

#### `metrics.py` — 指标计算

- `accuracy`, `precision_recall_f1`, `confusion_matrix` (基于 sklearn)
- `compute_margin(confidence_scores)` → Top1-Top2 概率差
- `boundary_accuracy(results, boundary_mask)` → 62 条边界样本的准确率

#### `pipeline.py` — 评估流水线

```python
class EvaluationPipeline:
    def evaluate(self, profiles: ProfileSet, test_data) -> EvalResult
    def compute_fitness(self, profiles, test_data, lambda_penalty=100) -> float
        # F(P) = Accuracy - λ · max(0, avg_len - 200)
```

---

### 模块③ `whitebox_init/`

| 子模块 | 输入 | 输出 | 调用模型 |
|--------|------|------|---------|
| `tool_parser.py` | agent.tools[], global_tools | C_text, R_text | 规则映射（无需 LLM） |
| `prompt_compressor.py` | agent.system_prompt | StructuredProfile | Gemini Pro |
| `metadata_extractor.py` | agent.rag_metadata | B_text | 规则映射 |

`generate_initial_population(agents, pop_size=20)` → 生成初始种群

---

### 模块④ `evolution/`

#### `genetic_algorithm.py`

```python
class GeneticAlgorithm:
    def __init__(self, pop_size=20, n_generations=30, 
                 crossover_rate=0.7, mutation_rate=0.3,
                 elite_count=2, tournament_k=3, lambda_penalty=100):
        ...
    def evolve(self, initial_population, eval_pipeline, 
               unlabeled_data=None, margin_interval=5) -> EvolutionResult
```

核心循环：选择 → 交叉 → 变异 → 评估 → 精英保留 → （每5代）Margin 注入

#### `llm_mutator.py`

```python
class LLMMutator:
    def __init__(self, model="gemini-pro", api_key=...):
        ...
    def mutate(self, profile, bad_cases, target_slot) -> StructuredProfile
        # 三种模式：rewrite / abbreviate / add_negation
```

#### `margin_sampling.py`

```python
def compute_margins(classifier, profiles, unlabeled_queries) -> MarginResults
def select_boundary_queries(margins, threshold=0.1, top_k=20) -> List[Query]
```

---

### 模块⑤ `baselines/`

| 基线 | 实现 |
|------|------|
| 原始人工描述 | 加载 JSON，直接作为 Prompt |
| INTENT_GPT | 复现：用 few-shot (5/class) 生成 Prompt → qwen2.5-32b 分类 |
| 随机压缩 | 对人工描述随机截断到 200 字 |

---

### 模块⑥ `experiments/`

五个独立脚本，每个可单独运行：

```bash
# 在服务器上执行
cd /path/to/wyf-exp1

# Step 1: 白盒初始化
python -m experiments.run_whitebox_init

# Step 2: 遗传演化
python -m experiments.run_evolution --generations 30 --pop_size 20

# Step 3-4: 对比 & 消融实验
python -m experiments.run_comparison
python -m experiments.run_ablation

# Step 5: 生成可视化报告
python -m experiments.visualize
```

---

### 配置 `config.py`

```python
# === 模型配置 ===
BACKBONE_MODEL = "qwen2.5-32b"
BACKBONE_API_URL = "http://localhost:8000/v1"   # vLLM 本地服务
OPTIMIZER_MODEL = "gemini-pro"
OPTIMIZER_API_KEY = "..."                        # Gemini Pro API Key

# === GA 超参数 ===
POPULATION_SIZE = 20
N_GENERATIONS = 30
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.3
ELITE_COUNT = 2
TOURNAMENT_K = 3
LAMBDA_PENALTY = 100
MAX_PROFILE_LENGTH = 200

# === 数据路径 ===
GOLDEN_TEST_PATH = "data/golden_test_set.jsonl"
UNLABELED_PATH = "data/unlabeled_queries.jsonl"
AGENTS_PATH = "data/agents/sample_agents.json"
RESULTS_DIR = "results/"

# === 评估配置 ===
EVAL_SUBSET_SIZE = 100       # 适应度评估使用的子集大小（加速）
MARGIN_THRESHOLD = 0.1
MARGIN_SAMPLING_INTERVAL = 5  # 每 5 代做一次 Margin Sampling
MARGIN_TOP_K = 20
```

---

## 服务器部署说明

### 环境准备

```bash
# 1. 创建项目目录
mkdir -p /home/iilab9/scholar-papers/experiments/intention/exp-1
# 2. 复制代码
cp -r wyf-exp1/* /home/iilab9/scholar-papers/experiments/intention/exp-1/
# 3. 安装依赖
pip install -r requirements.txt
# 4. 启动 vLLM 推理服务 (占用 4×L20)
python -m vllm.entrypoints.openai.api_server \
    --model qwen2.5-32b \
    --tensor-parallel-size 4 \
    --port 8000
```

### `requirements.txt`

```
openai>=1.0          # 调用 vLLM 的 OpenAI-compatible API
google-generativeai  # Gemini Pro API
scikit-learn         # 指标计算
matplotlib           # 可视化
seaborn              # 混淆矩阵热图
numpy
tqdm
```

---

## Verification Plan

### Automated Tests

```bash
python -m pytest tests/ -v
```

- `test_profile.py`：Profile 创建、字数计算、序列化、槽位变异、字数校验
- `test_metrics.py`：Accuracy/P/R/F1、混淆矩阵、Margin 计算、适应度惩罚项
- `test_ga.py`：锦标赛选择、结构保留交叉、精英保留逻辑

### End-to-End (Mock 模式)

```bash
python -m experiments.run_evolution --mock --generations 3 --pop_size 5
```

使用 Mock 分类器（随机返回概率），验证整个 GA 流程能正确运行。

### Real Evaluation

在服务器上启动 vLLM 后，使用真实数据运行完整实验，对比基线结果。
