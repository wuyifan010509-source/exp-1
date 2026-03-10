# 论文第3.3节：基于贪心策略与裕度采样的描述迭代优化机制

## 写作素材与代码细节

---

### （1）零样本冷启动初始化（白盒特征蒸馏）

**核心思想**：利用智能体的**白盒工具描述**（Tools Descriptions）生成结构化初始描述，避免冷启动时的随机搜索。

**代码实现**（`whitebox_init/__init__.py:144-189`）：

```python
def _generate_rule_based(self, agent_name, tools, description, all_tools, intent):
    """基于规则的生成（简化版）"""
    # C: 直接使用工具描述
    c = description[:MAX_C_LENGTH]  # 核心能力 = 工具描述
    
    # B: 处理边界 = 同类的一个例子，加引号
    agent_examples = self.examples_by_intent.get(intent, [])
    if agent_examples:
        b_example = random.choice(agent_examples)
        b = f"包括'{b_example[:15]}'"  # 带引号
    else:
        b = f"包括'{description[:10]}'"
    
    # R: 拒绝范围 = 不同类例子 + 正确类别标注
    other_examples = [...]  # 从其他智能体采样
    if other_examples:
        r_example, correct_agent = random.choice(other_examples)
        r = f"不包括'{r_example[:10]}'（实际{correct_agent[:6]}）"
    
    return c, b, r
```

**数学表达**：
- 初始描述 $P_0 = \\{C_0, B_0, R_0\\}$
- $C_0 = \\text{ToolDescription}$（零样本蒸馏）
- $B_0 = \\{q_i \\sim \\mathcal{D}_{\\text{intent}}\\}$（同类采样）
- $R_0 = \\{q_j \\sim \\mathcal{D}_{\\neg\\text{intent}}\\}$（异类采样）

---

### （2）边界数据挖掘（Margin Sampling / 裕度采样）

**核心思想**：通过**置信度裕度**（Margin = Top1概率 - Top2概率）识别"模糊边界"样本，优先优化难分样本。

**代码实现**（`evaluation/classifier.py:26-31`）：

```python
@dataclass
class ClassifyResult:
    predicted_agent: str
    confidence_scores: Dict[str, float]  # 各智能体概率
    
    def get_margin(self) -> float:
        """计算Top1-Top2概率差（置信度裕度）"""
        sorted_scores = sorted(self.confidence_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            return sorted_scores[0] - sorted_scores[1]
        return 1.0  # 只有一个类别时margin为1
```

**裕度采样策略**（`whitebox_init/__init__.py:237-250`）：

```python
# 分类正确但置信度低（Margin小）的样本
if expected_agent == result.predicted_agent:
    margin = result.get_margin()  # 计算裕度
    if expected_agent not in agent_to_low_conf:
        agent_to_low_conf[expected_agent] = []
    agent_to_low_conf[expected_agent].append((query, margin))

# 按裕度排序（越小越优先）
low_conf_samples = sorted(agent_to_low_conf[agent_name], key=lambda x: x[1])
lowest_conf_query = low_conf_samples[0][0]  # 取裕度最低的样本
```

**数学表达**：
- $\\text{Margin}(q) = p(\\hat{y}|q) - p(y^{(2)}|q)$
- 其中 $\\hat{y} = \\arg\\max_y p(y|q)$，$y^{(2)} = \\arg\\max_{y \\neq \\hat{y}} p(y|q)$
- **采样策略**：优先选择 $\\text{Margin}(q) < \\tau$ 的样本（低置信度样本）

---

### （3）带长度约束的贪心坐标搜索（定向优化）

**核心思想**：将描述优化视为**坐标下降**问题，每轮选择修改次数最少的槽位进行定向优化。

**算法架构**（`greedy_optimizer.py:33-71`）：

```python
class GreedyOptimizer:
    def __init__(self, 
                 max_iterations=100,
                 candidates_per_slot=3,
                 patience=10,           # 早停耐心值
                 slots_per_iteration=3,  # 每轮优化槽位数
                 window_size=3,         # 滑动窗口
                 improvement_threshold=0.001):
        
        self.slot_selection_history = {}  # 修改历史记录
```

**坐标选择策略**（`greedy_optimizer.py:286-337`）：

```python
def _select_worst_slots(self, profile_set, all_cases, top_k=3):
    """
    选择修改历史最少的槽位（坐标选择）
    策略：同一个智能体每轮只选一个槽位（C/B/R只改一个）
    """
    for agent_name in profile_set.profiles.keys():
        for slot_code in ['C', 'B', 'R']:
            slot_key = (agent_name, slot_code)
            # 基于修改历史选择（次数越少优先级越高）
            selection_count = self.slot_selection_history.get(slot_key, 0)
            score = -selection_count  # 负数：次数少得分高
    
    # 每个智能体只选修改次数最少的一个槽位
    selected = sorted(agent_slot_scores.items(), key=lambda x: x[1], reverse=True)
    return selected[:top_k]
```

**LLM定向变异**（`greedy_optimizer.py:607-682`）：

```python
# 为每个槽位构建专属Prompt
if slot_name == '核心能力':
    prompt = f"""请生成[核心能力]描述：
    1. 字数：严格≤70字
    2. 基于分类正确的例子：{correct_examples}
    3. 必须包含具体示例，例子越多越好"""
    
elif slot_name == '处理边界':
    prompt = f"""请生成[处理边界]描述：
    1. 格式：包括'查询内容'
    2. 基于期望是本智能体的例子：{positive_examples}"""
    
else:  # 拒绝范围
    prompt = f"""请生成[拒绝范围]描述：
    1. 格式：不包括'查询'（实际xxx类）
    2. 基于错分案例：{negative_examples}
    3. 必须标注正确类别"""
```

**数学表达**：
- **坐标**：$(i, j)$ 表示第$i$个智能体的第$j$个槽位
- **目标函数**：$\\max_{P} \\mathcal{F}(P) = \\text{Accuracy}(P) - \\lambda \\cdot \\max(0, \\text{len}(P) - L_{\\max})$
- **约束**：$|C_i| \\leq 70, |B_i| \\leq 70, |R_i| \\leq 70$（每槽位字数限制）

---

### （4）自动化评估与版本决策

**核心思想**：采用**滑动窗口基线**评估改进，自动决定是否接受新版本。

**适应度函数**（`evaluation/metrics.py:136-143`）：

```python
def compute_fitness(accuracy, avg_length, max_length=200, lambda_penalty=100):
    """
    计算适应度函数
    F(P) = Accuracy - λ · max(0, avg_len - max_len)
    """
    length_penalty = lambda_penalty * max(0, avg_length - max_length)
    return accuracy - length_penalty
```

**滑动窗口决策**（`greedy_optimizer.py:151-195`）：

```python
# 【滑动窗口基线】计算最近N轮的平均适应度作为基线
window = window_fitness_history[-self.window_size:]
baseline = sum(window) / len(window)

# 检查是否有改进（必须超过基线+阈值）
improvement_over_baseline = improved_fitness - baseline
accepted = improvement_over_baseline > self.improvement_threshold

if accepted:
    current = improved_profile  # 接受新版本
    fitness_history.append(current_fitness)
    no_improvement_count = 0
else:
    no_improvement_count += 1  # 拒绝，计数+1
    if no_improvement_count >= self.patience:
        break  # 早停
```

**检查点保存机制**（`greedy_optimizer.py:685-721`）：

```python
def _save_checkpoint(self, profile_set, fitness, iteration, save_dir):
    """保存每轮优化结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON格式
    data = {
        "iteration": iteration,
        "fitness": fitness,
        "profiles": profile_set.to_dict()
    }
    with open(f"{save_dir}/iter_{iteration}_{timestamp}.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    # 人类可读格式
    with open(f"{save_dir}/iter_{iteration}_{timestamp}.txt", 'w') as f:
        for agent_name, profile in profile_set.profiles.items():
            f.write(f"【{agent_name}】\\n")
            f.write(f"C-核心能力: {profile.core_capability}\\n")
            f.write(f"B-处理边界: {profile.boundary}\\n")
            f.write(f"R-拒绝范围: {profile.rejection_scope}\\n\\n")
```

**数学表达**：
- **接受条件**：$f(P_{new}) > \\frac{1}{w}\\sum_{i=1}^{w} f(P_{t-i}) + \\epsilon$
  - $w$：滑动窗口大小（3轮）
  - $\\epsilon$：改进阈值（0.001）
- **早停条件**：连续 $patience=10$ 轮无改进则停止

---

## 📊 关键实验参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大迭代次数 | 100 | 防止无限循环 |
| 每轮优化槽位数 | 3-5 | 控制计算成本 |
| 改进阈值 | 0.001 | 最小可接受改进 |
| 早停耐心值 | 10 | 收敛检测 |
| 滑动窗口大小 | 3 | 基线计算范围 |
| 字数限制 | C≤70, B≤70, R≤70 | 结构化约束 |

---

## 🎯 核心创新点总结

1. **白盒特征蒸馏**：利用Tools描述零样本生成初始结构化描述
2. **裕度采样**：通过置信度Margin识别模糊边界样本，优先优化难分案例
3. **坐标下降优化**：每轮选择修改历史最少的槽位，避免重复优化
4. **滑动窗口决策**：基于历史平均基线评估改进，自动接受/拒绝新版本
5. **结构化约束**：C/B/R三槽位正交设计，保证语义完备性和可解释性

---

## 📁 相关代码文件

- `wyf-exp1/whitebox_init/__init__.py` - 白盒初始化
- `wyf-exp1/greedy/greedy_optimizer.py` - 贪心优化器
- `wyf-exp1/evaluation/classifier.py` - 分类器与Margin计算
- `wyf-exp1/evaluation/metrics.py` - 评估指标与适应度函数
- `wyf-exp1/experiments/run_greedy.py` - 实验主流程

---

**生成时间**: 2026-03-09  
**来源**: wyf-exp1代码库分析
