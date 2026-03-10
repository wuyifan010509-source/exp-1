# wyf-exp1 架构设计文档

## 1. 项目概述

本项目实现**结构化描述优化框架**，通过遗传算法(GA)自动搜索最优的意图分类描述。

**核心创新点**：
- 结构化描述 P={C,B,R}（核心能力、处理边界、拒绝范围）
- 白盒初始化 + 黑盒演化 两阶段框架
- LLM驱动的定向变异
- Margin Sampling边界挖掘

---

## 2. 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         实验主流程                               │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: 白盒初始化        Phase 2: 遗传演化                    │
│  ┌──────────────┐          ┌──────────────────────────┐        │
│  │ Tools Parser │─────────▶│    Genetic Algorithm     │        │
│  └──────────────┘          │  ┌────────────────────┐   │        │
│           │                │  │ 1. Selection       │   │        │
│           ▼                │  │ 2. Crossover (C/B/R)│   │        │
│  ┌──────────────┐          │  │ 3. LLM Mutation    │   │        │
│  │  DeepSeek    │─────────▶│  │ 4. Evaluation      │   │        │
│  │   (API)      │          │  │ 5. Elite Preserve  │   │        │
│  └──────────────┘          │  └────────────────────┘   │        │
│                            └──────────────────────────┘        │
│                                       │                        │
│                                       ▼                        │
│                            ┌──────────────────────────┐        │
│                            │    Best Profile Set P*   │        │
│                            └──────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 3: 评估与对比                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Baseline │  │  Full    │  │ Compare  │  │ Visualize│       │
│  │   Eval   │  │  Eval    │  │  Exp     │  │  Report  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 模块详解

### 3.1 structured_profile/ - 结构化描述建模

**设计思想**：将自由文本描述强制约束为三个正交槽位，形成语义空间的"最小完备集"。

**核心类**：

```python
@dataclass
class StructuredProfile:
    agent_name: str
    core_capability: str   # C: 核心能力 - "我能做什么"
    boundary: str          # B: 处理边界 - "在什么条件下做"
    rejection_scope: str   # R: 拒绝范围 - "我明确不做什么"
```

**关键方法**：
- `to_prompt()`: 转换为分类Prompt格式
- `mutate_slot()`: 槽位级变异，保持其他槽位不变
- `is_valid()`: 字数约束检查

**设计理由**：
1. **正交性**：C/B/R三个槽位互相独立，交叉操作不会破坏语义完整性
2. **完备性**：覆盖意图定义的三个方面（正向能力、条件约束、负向排斥）
3. **可解释性**：结构透明，便于调试和分析

---

### 3.2 evaluation/ - 评估流水线

**架构模式**：策略模式 + 流水线模式

#### classifier.py - 意图分类器

**抽象基类**：`IntentClassifier`
- 支持Mock分类器（测试）和真实模型分类器
- 统一接口：`classify()` 和 `classify_batch()`

**实现类**：`QwenClassifier`
- 通过OpenAI-compatible API调用vLLM部署的qwen2.5-32b
- 支持JSON格式输出置信度分布
- 用于计算Margin (Top1-Top2概率差)

**关键设计**：
```python
def classify(query, profiles) -> ClassifyResult:
    predicted_agent: str
    confidence_scores: Dict[str, float]  # 各智能体概率
    raw_response: str
    
def get_margin() -> float:
    """Top1概率 - Top2概率，用于边界样本识别"""
```

#### metrics.py - 指标计算

**核心指标**：
1. **Accuracy**: 总体分类准确率
2. **Boundary Accuracy**: 边界样本准确率（62条）
3. **Average Margin**: 平均置信度间隔
4. **Fitness**: 适应度 = Accuracy - λ·max(0, len-200)

**适应度函数设计**：
```
F(P) = Accuracy(P) - λ·max(0, avg_length - 200)
```
- 硬约束：通过惩罚项强制≤200字
- 软优化：提升准确率的同时控制长度

---

### 3.3 whitebox_init/ - 白盒初始化

**设计思想**：利用智能体的白盒信息（tools描述）生成高质量的初始种群，解决GA冷启动问题。

**数据流**：
```
Tools JSON
    │
    ├─▶ Tool Parser ──▶ C (核心能力): "提供search_stocks等功能"
    │
    ├─▶ Complement Set ──▶ R (拒绝范围): "不处理analyze_stock等"
    │
    └─▶ Description ──▶ B (处理边界): "主要处理选股相关请求"
    │
    ▼
[DeepSeek API] 压缩优化
    │
    ▼
Initial Population (20个变体)
```

**两种模式**：
1. **LLM模式**：调用DeepSeek生成高质量描述
2. **规则模式**：基于关键词匹配快速生成（无API时使用）

**变体生成策略**：
- 温度递增：`temperature = 0.7 + variant_id * 0.05`
- 每个变体在措辞和侧重点上有所差异

---

### 3.4 evolution/ - 遗传算法

#### genetic_algorithm.py - GA核心引擎

**算法参数**：
```python
pop_size = 20          # 种群大小
n_generations = 30     # 迭代代数
crossover_rate = 0.7   # 交叉概率
mutation_rate = 0.3    # 变异概率
elite_count = 2        # 精英保留数
tournament_k = 3       # 锦标赛规模
```

**核心流程**（每代）：
```
1. 评估适应度（调用GPU模型分类）
   └── 20个体 × 100条Query = 2,000次推理

2. 锦标赛选择（k=3）
   └── 从3个随机个体中选最优

3. 结构保留交叉（70%概率）
   └── 交换父代的C/B/R槽位（非字符级）
   
   父代1: C₁⊕B₁⊕R₁    父代2: C₂⊕B₂⊕R₂
   ────────────────────────────────────
   子代1: C₁⊕B₂⊕R₁    子代2: C₂⊕B₁⊕R₂

4. LLM定向变异（30%概率）
   └── 基于Bad Cases重写特定槽位

5. 精英保留（Top-2直接进入下一代）
```

**创新点**：
- **结构保留交叉**：槽位级交换避免语法破碎
- **精英保留**：防止优秀基因丢失
- **检查点保存**：每5代保存中间结果

#### llm_mutator.py - LLM定向变异

**变异策略**：
1. **随机选择**：20%概率变异每个智能体
2. **槽位选择**：随机选C/B/R之一
3. **LLM重写**：使用DeepSeek优化特定槽位

**Prompt设计**：
```
你是一个智能体描述优化专家。

当前：[核心能力] {current_C}

以下案例被错误分类：
- 问题：{bad_case_1.query}，期望：{bad_case_1.expected}

请优化[核心能力]，要求：
1. 字数 ≤ 80字
2. 纠正上述错误分类
3. 优先使用否定式表述

输出优化后的内容：
```

**Fallback机制**：
- API失败时自动回退到规则变异
- 规则变异：缩写/添加否定/重排序

---

## 4. 数据流详解

### 4.1 训练阶段数据流

```
┌─────────────┐
│ Tools JSON  │──┐
└─────────────┘  │
                 ▼
┌─────────────────────────────┐
│    WhiteBox Initializer     │
│  ┌─────────┐ ┌───────────┐ │
│  │Rule Gen │ │LLM Refine │ │
│  └────┬────┘ └─────┬─────┘ │
└───────┼────────────┼───────┘
        │            │
        ▼            ▼
┌─────────────────────────────────┐
│     Initial Population (20)     │
│  ProfileSet × 20                │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│      Genetic Algorithm Loop     │
│                                 │
│  ┌───────────────────────────┐  │
│  │ Generation 1-30           │  │
│  │                           │  │
│  │ ① Evaluate Fitness        │  │
│  │    └── GPU Inference      │  │
│  │    └── Accuracy + Length  │  │
│  │                           │  │
│  │ ② Selection               │  │
│  │    └── Tournament (k=3)   │  │
│  │                           │  │
│  │ ③ Crossover (70%)         │  │
│  │    └── Slot-level         │  │
│  │                           │  │
│  │ ④ Mutation (30%)          │  │
│  │    └── DeepSeek API       │  │
│  │                           │  │
│  │ ⑤ Elite Preserve (Top2)  │  │
│  └───────────────────────────┘  │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│      Best Profile Set P*        │
└─────────────────────────────────┘
```

### 4.2 推理阶段数据流

```
用户Query
    │
    ▼
┌─────────────────────────────────┐
│         Intent Router           │
│  ┌───────────────────────────┐  │
│  │ P* = {C,B,R} for 12 agents│  │
│  └───────────────────────────┘  │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│       Qwen2.5-32b (GPU)         │
│  Prompt: "根据以下描述分类..."  │
│  Input: Query + 12个P*          │
│  Output: Predicted Agent        │
└─────────────────────────────────┘
```

---

## 5. 关键设计决策

### 5.1 为什么用遗传算法？

| 方案 | 问题 | GA优势 |
|------|------|--------|
| 梯度优化 | 文本空间离散不可导 | 无需求导 |
| 贝叶斯优化 | 高维搜索空间(12×200字) | 并行搜索 |
| 随机搜索 | 效率低 | 利用历史信息 |
| **遗传算法** | - | 适合结构化空间、可保留组件 |

### 5.2 为什么用C/B/R结构？

**集合论论证**：
- 定义意图 = 在语义空间划定集合
- C = 集合内部（正例）
- B = 集合边界（条件）  
- R = 集合补集（负例）
- **C+B+R = 最小完备集**

**实践优势**：
- 交叉操作不破坏语法
- 易于定位和修复Bad Cases
- 符合LLM对结构化数据的偏好

### 5.3 为什么用两阶段框架？

```
白盒推导 (Zero-Shot)    黑盒演化 (Few-Shot)
    │                         │
    ▼                         ▼
  P₀ ─────────────────────▶ P*
  (初始猜测)              (精确优化)
  
优势：
1. 利用已有信息，无需从零搜索
2. 大幅减少收敛代数（30代 vs 100+代）
3. 冷启动问题转化为微调问题
```

---

## 6. 性能优化策略

### 6.1 计算代价优化

| 优化策略 | 效果 | 实现 |
|---------|------|------|
| 子集评估 | 3.5x加速 | 354→100条 |
| 批量推理 | 5-10x加速 | vLLM batching |
| 缓存机制 | 60%+命中 | Query-Profile对缓存 |
| 提前停止 | 可变 | 收敛检测 |

### 6.2 代码层面优化

```python
# 1. 延迟加载
classifier = None  # 首次使用时才初始化

# 2. 并行评估（未来扩展）
from multiprocessing import Pool
def evaluate_parallel(population):
    with Pool(4) as p:
        fitnesses = p.map(evaluate_single, population)

# 3. 增量更新
- 只评估变异后的个体
- 其他个体复用上代fitness
```

---

## 7. 扩展性设计

### 7.1 添加新的基线方法

在 `baselines/` 目录下：
```python
class NewBaseline:
    def generate_profiles(self) -> ProfileSet:
        # 实现新的描述生成逻辑
        pass
```

### 7.2 更换分类模型

修改 `config.py`：
```python
BACKBONE_MODEL = "new-model-name"
BACKBONE_API_URL = "http://new-endpoint:8000/v1"
```

### 7.3 添加新的变异算子

在 `llm_mutator.py` 中扩展：
```python
def mutate_slot_new_strategy(self, ...):
    # 实现新的变异策略
    pass
```

---

## 8. 文件组织

```
wyf-exp1/
├── config.py                   # 全局配置
├── requirements.txt            # 依赖
├── ARCHITECTURE.md            # 本文档
│
├── structured_profile/        # 模块1: 数据建模
│   ├── __init__.py           # StructuredProfile, ProfileSet
│   └── profile.py            # (已合并)
│
├── evaluation/                # 模块2: 评估
│   ├── __init__.py           # EvaluationPipeline
│   ├── classifier.py         # IntentClassifier, QwenClassifier
│   └── metrics.py            # 指标计算
│
├── whitebox_init/            # 模块3: 白盒初始化
│   └── __init__.py           # WhiteBoxInitializer
│
├── evolution/                # 模块4: 遗传算法
│   ├── __init__.py           # GeneticAlgorithm, LLMMutator
│   ├── genetic_algorithm.py  # GA核心
│   └── llm_mutator.py        # LLM变异
│
├── baselines/                # 模块5: 基线（预留）
│   └── __init__.py
│
├── experiments/              # 模块6: 实验脚本
│   ├── run_mve.py           # 最小可行实验
│   ├── run_evolution.py     # 完整演化（预留）
│   ├── run_comparison.py    # 对比实验（预留）
│   └── visualize.py         # 可视化（预留）
│
├── data/                     # 数据目录
│   ├── agents/
│   │   └── tools_descriptions.json  # 需填写
│   ├── GOLDEN_TEST.csv      # 黄金测试集
│   └── HISTORICAL_LOGS.csv  # 历史日志
│
└── results/                  # 结果输出
    └── .gitkeep
```

---

## 9. 使用流程

### 9.1 首次运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 填写配置
vim config.py
# - 填入GPU服务器地址
# - 填入DeepSeek API Key

# 3. 检查/填写tools描述
vim data/agents/tools_descriptions.json

# 4. 运行最小实验验证流程
python -m experiments.run_mve --mock

# 5. 运行真实实验（需GPU）
python -m experiments.run_mve
```

### 9.2 配置说明

**必需配置**：
- `BACKBONE_API_URL`: GPU服务器vLLM地址
- `DEEPSEEK_API_KEY`: DeepSeek API密钥（可选，用于LLM变异）

**可选调参**：
- `POPULATION_SIZE`: 种群大小（默认20）
- `N_GENERATIONS`: 迭代代数（默认30）
- `EVAL_SUBSET_SIZE`: 评估子集大小（默认100）

---

## 10. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| GPU连接失败 | URL错误或服务未启动 | 检查config.py和vLLM状态 |
| API调用失败 | Key无效或网络问题 | 检查DeepSeek Key，可改用Mock模式 |
| 内存不足 | 模型太大 | 减小batch_size或使用更小模型 |
| 演化不收敛 | 初始种群质量差 | 检查tools描述是否完整 |
| 准确率过低 | 描述不匹配 | 增加代数或使用更好的初始化 |

---

## 11. 参考文献

1. **遗传算法**: Holland, J.H. (1975). Adaptation in Natural and Artificial Systems
2. **结构化Prompt**: Wei et al. (2023). Chain-of-Thought Prompting
3. **意图分类**: Zhang et al. (2022). Intent Classification with LLMs
4. **Margin Sampling**: Settles, B. (2009). Active Learning Literature Survey

---

**文档版本**: v1.0  
**最后更新**: 2026-03-07  
**作者**: wyf-exp1 Team
