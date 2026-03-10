# 遗传算法实现文档

## 概述

本文档记录意图分类描述优化实验中遗传算法的完整实现思路、关键决策和技术细节。

---

## 1. 问题建模

### 1.1 搜索空间定义

**个体表示**：每个个体是一个 `ProfileSet`，包含12个智能体的结构化描述

```
ProfileSet = {Profile₁, Profile₂, ..., Profile₁₂}

Profileᵢ = {Cᵢ, Bᵢ, Rᵢ}
  - Cᵢ (Core): 核心能力描述，≤80字
  - Bᵢ (Boundary): 处理边界描述，≤60字  
  - Rᵢ (Rejection): 拒绝范围描述，≤60字
```

**总搜索空间维度**：12个智能体 × (80+60+60)字 = 2400字

### 1.2 为什么用结构化表示？

| 方案 | 问题 | 结构化表示优势 |
|------|------|----------------|
| 自由文本 | 交叉操作会破坏语义完整性 | C/B/R正交，交叉不破坏语法 |
| 单一段落 | 难以定位错误来源 | 槽位级可追溯 |
| 无约束生成 | 长度难以控制 | 槽位级字数限制 |

**核心洞察**：C+B+R构成语义空间的**最小完备集**
- C = 集合内部（正例）
- B = 集合边界（条件）
- R = 集合补集（负例）

---

## 2. 遗传算法设计

### 2.1 算法参数

```python
POPULATION_SIZE = 20      # 种群大小
N_GENERATIONS = 30        # 迭代代数
CROSSOVER_RATE = 0.7      # 交叉概率
MUTATION_RATE = 0.3       # 变异概率
ELITE_COUNT = 2           # 精英保留数
TOURNAMENT_K = 3          # 锦标赛规模
```

**参数选择依据**：
- **种群大小20**：平衡探索与计算成本（20个体 × 100条评估 = 2000次推理/代）
- **精英保留2**：保留Top-10%，防止优秀基因丢失
- **锦标赛k=3**：适中选择压力，避免过早收敛

### 2.2 核心流程

```
每代循环:
  1. 适应度评估
     └── 调用GPU模型对100条Query进行分类
     └── 计算准确率 + 字数惩罚
  
  2. 锦标赛选择 (k=3)
     └── 随机选3个个体，取最优作为父代
  
  3. 结构保留交叉 (70%概率)
     └── 槽位级交换（非字符级）
     └── 父代1: C₁⊕B₁⊕R₁  父代2: C₂⊕B₂⊕R₂
     └── 子代1: C₁⊕B₂⊕R₁  子代2: C₂⊕B₁⊕R₂
  
  4. LLM定向变异 (30%概率)
     └── 基于Bad Cases重写特定槽位
  
  5. 精英保留 (Top-2直接进入下一代)
```

---

## 3. 关键算子实现

### 3.1 锦标赛选择

```python
def _tournament_selection(self, population):
    """锦标赛选择"""
    contestants = random.sample(population, k=3)
    return max(contestants, key=lambda x: x.fitness)
```

**设计理由**：
- 比轮盘赌选择更稳定（避免fitness差异过大导致的选择偏差）
- 比排序选择保留更多多样性
- k=3提供适中的选择压力

### 3.2 结构保留交叉

```python
def _crossover(self, parent1, parent2, generation):
    """结构保留交叉 - 槽位级别"""
    for agent_name in parent1.profile_set.profiles.keys():
        profile1 = parent1.profile_set.get_profile(agent_name)
        profile2 = parent2.profile_set.get_profile(agent_name)
        
        # 对每个槽位独立决定是否交换
        for slot in ['C', 'B', 'R']:
            if random.random() < 0.5:
                # 交换该槽位
                child1_slots[slot] = profile2.get_slot(slot)
                child2_slots[slot] = profile1.get_slot(slot)
```

**关键创新**：
- **槽位级交换**：保持每个槽位的语义完整性
- **独立决策**：每个槽位独立决定是否交换，增加组合多样性
- **语法安全**：不会出现字符级交叉导致的语法破碎

**示例**：
```
父代1 - 选股智能体: 
  C="提供选股功能，支持多条件筛选"
  B="处理技术指标、基本面条件的组合查询"
  R="不处理诊股分析、走势预测"

父代2 - 选股智能体:
  C="根据条件筛选股票"  
  B="支持技术指标和基本面条件"
  R="不涉及个股诊断和未来预测"

子代1（交换B槽位）:
  C="提供选股功能，支持多条件筛选"（来自父代1）
  B="支持技术指标和基本面条件"（来自父代2）
  R="不处理诊股分析、走势预测"（来自父代1）
```

### 3.3 LLM定向变异

**变异策略**：
1. **个体选择**：20%概率变异每个智能体（平均每代变异2-3个智能体）
2. **槽位选择**：随机选择C/B/R之一进行变异
3. **LLM重写**：使用DeepSeek API优化特定槽位

**Prompt设计**：
```
你是一个智能体描述优化专家。

当前智能体：{agent_name}
需要优化的槽位：[{slot_name}]
当前内容：{current_text}

请对[{slot_name}]进行优化，要求：
1. 优化后的字数 ≤ {max_len}字
2. 保持语义精确，避免模糊表述
3. 优先使用否定式表述（"不处理..."）来划定边界
4. 突出该智能体的独特能力范围

参考以下分类错误的案例（可帮助理解边界）：
- 问题：{bad_case_1.query}，期望：{bad_case_1.expected}
...

请直接输出优化后的[{slot_name}]内容（不要解释）：
```

**Fallback机制**：
- API失败时自动回退到随机变异
- 随机变异策略：缩写 / 添加否定 / 重排序

---

## 4. 适应度函数

### 4.1 设计

```python
F(P) = Accuracy(P) - λ·max(0, avg_length - 200)

其中:
- Accuracy: 在100条Query上的分类准确率
- avg_length: 平均每个Profile的字数
- λ = 100: 惩罚系数
- 200: 字数上限
```

### 4.2 设计理由

**为什么用惩罚项而非硬约束？**
- 硬约束会过早排除有潜力的个体
- 软约束允许在优化初期探索较长描述
- 演化过程自然收敛到≤200字

**惩罚系数选择**：
- λ=100意味着每超过1字，accuracy需要提升1%才能补偿
- 确保最终解严格满足字数约束

---

## 5. 计算效率优化

### 5.1 评估子集

**完整评估**：354条 × 20个体 = 7,080次推理/代
**子集评估**：100条 × 20个体 = 2,000次推理/代
**加速比**：3.5×

**子集选择策略**：
- 每代随机采样100条（增加多样性压力）
- 最终一代用完整354条评估

### 5.2 增量评估（预留）

```python
# 未来优化：只评估变异后的个体
for ind in population:
    if ind.profile_set.changed:  # 标记是否被变异
        ind.fitness = fitness_func(ind.profile_set)
    # 否则复用上代fitness
```

### 5.3 并行评估（预留）

```python
# 未来优化：多进程并行评估
from multiprocessing import Pool

def evaluate_parallel(population):
    with Pool(4) as p:
        fitnesses = p.map(evaluate_single, population)
    return fitnesses
```

---

## 6. 收敛性保障

### 6.1 精英保留

- 每代Top-2直接进入下一代
- 防止优秀基因因随机性丢失
- 确保fitness单调不减

### 6.2 检查点机制

```python
if (gen + 1) % 5 == 0:
    self._save_checkpoint(population, gen + 1, save_dir)
```

- 每5代保存一次中间结果
- 支持断点续跑
- 便于分析演化轨迹

### 6.3 早停条件（预留）

```python
# 连续5代提升<1%时停止
if len(self.best_fitness_history) >= 5:
    recent_improvement = (
        self.best_fitness_history[-1] - 
        self.best_fitness_history[-5]
    )
    if recent_improvement < 0.01:
        print("[GA] Early stopping triggered")
        break
```

---

## 7. 与白盒初始化的协同

### 7.1 两阶段框架

```
白盒初始化 (Zero-Shot)    遗传演化 (Few-Shot)
     │                          │
     ▼                          ▼
   P₀ ─────────────────────▶  P*
   (初始猜测)              (精确优化)
```

### 7.2 优势互补

| 阶段 | 优势 | 局限 | 协同效果 |
|------|------|------|----------|
| 白盒初始化 | 利用Tools信息，生成合理起点 | 无法感知分类边界 | 提供高质量P₀ |
| 遗传演化 | 通过Bad Cases精确优化 | 冷启动困难 | 30代即可收敛 |

**预期效果**：
- 无白盒初始化：需要100+代才能收敛
- 有白盒初始化：30代即可达到最优

---

## 8. 与基线方法的对比

### 8.1 相比随机搜索

**随机搜索**：
- 独立采样，无历史信息利用
- 收敛慢，浪费计算

**遗传算法**：
- 通过交叉保留优秀组件
- 通过变异探索邻域
- 通过选择聚焦高fitness区域

### 8.2 相比贝叶斯优化

**贝叶斯优化**：
- 需要连续可导空间
- 高维空间（2400维）效果差

**遗传算法**：
- 适合离散空间
- 高维空间表现稳定

### 8.3 相比梯度优化

**梯度优化**：
- 文本空间不可导
- 无法直接应用

**遗传算法**：
- 无需求导
- 直接操作文本

---

## 9. 实现注意事项

### 9.1 深拷贝问题

**陷阱**：Python默认浅拷贝，会导致个体间共享Profile对象

**解决方案**：
```python
def copy(self) -> 'Individual':
    """深拷贝"""
    return Individual(
        profile_set=self.profile_set.copy(),  # 递归深拷贝
        fitness=self.fitness,
        generation=self.generation
    )
```

### 9.2 适应度缓存

**问题**：同一Profile可能被多次评估（精英保留）

**解决方案**：
```python
# 首次评估时计算
if ind.fitness == 0.0:  # 避免重复评估
    ind.fitness = fitness_func(ind.profile_set)
```

### 9.3 API调用失败处理

**问题**：DeepSeek API可能超时或失败

**解决方案**：
- Try-catch包裹API调用
- 失败时自动回退到随机变异
- 打印警告但继续运行

---

## 10. 调试技巧

### 10.1 监控指标

```python
# 每代输出关键信息
print(f"[Gen {gen+1}] Best Fitness: {best_fitness:.4f}")
print(f"[Gen {gen+1}] Avg Fitness: {avg_fitness:.4f}")
print(f"[Gen {gen+1}] Best Length: {population[0].profile_set.average_length():.1f}")
```

### 10.2 可视化演化曲线

```python
import matplotlib.pyplot as plt

plt.plot(best_fitness_history, label='Best')
plt.plot(avg_fitness_history, label='Average')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('evolution_curve.png')
```

### 10.3 分析最佳个体

```python
# 输出最佳Profile的详细内容
best = result.best_individual
for agent_name, profile in best.profile_set.profiles.items():
    print(f"\n{agent_name}:")
    print(f"  C: {profile.core_capability}")
    print(f"  B: {profile.boundary}")
    print(f"  R: {profile.rejection_scope}")
```

---

## 11. 未来扩展

### 11.1 Margin Sampling集成

**思路**：
1. 每5代执行一次Margin Sampling
2. 选取Margin最小的20条Query加入评估集
3. 聚焦边界样本，加速收敛

### 11.2 自适应变异率

**思路**：
```python
# 根据种群多样性调整变异率
if diversity < threshold:
    self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
else:
    self.mutation_rate = max(0.1, self.mutation_rate * 0.9)
```

### 11.3 多目标优化

**思路**：
- 同时优化准确率和平均Margin
- 使用NSGA-II算法
- 获得Pareto前沿

---

## 12. 实验验证

### 12.1 最小实验(MVE)配置

```python
MVE_POP_SIZE = 5
MVE_GENERATIONS = 5
MVE_EVAL_SUBSET = 20
```

**预期结果**：
- 运行时间：5-10分钟（Mock模式）
- 适应度应有提升（随机初始→优化后）
- 无明显错误/异常

### 12.2 完整实验配置

```python
POPULATION_SIZE = 20
N_GENERATIONS = 30
EVAL_SUBSET_SIZE = 100
```

**预期结果**：
- 运行时间：15-20 GPU小时
- 准确率提升：基线 → +5-10%
- 最终适应度：>0.85

---

## 参考文献

1. Holland, J.H. (1975). Adaptation in Natural and Artificial Systems
2. Goldberg, D.E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning
3. Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms
4. Whitley, D. (1994). A Genetic Algorithm Tutorial

---

**文档版本**: v1.0  
**最后更新**: 2026-03-07  
**作者**: wyf-exp1 Team
