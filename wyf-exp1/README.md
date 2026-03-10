# wyf-exp1: 意图分类优化实验

## 快速开始

### 1. 环境准备

```bash
cd wyf-exp1
pip install -r requirements.txt
```

### 2. 配置填写

编辑 `config.py`，填写以下配置：

```python
# GPU模型配置（必需）
BACKBONE_API_URL = "http://your-gpu-server:8000/v1"
BACKBONE_MODEL = "qwen2.5-32b"

# DeepSeek API（可选，用于更好的变异效果）
DEEPSEEK_API_KEY = "your-key-here"
```

### 3. 数据检查

确保数据文件存在：
- `data/GOLDEN_TEST.csv` - 黄金测试集（354条）
- `data/HISTORICAL_LOGS.csv` - 历史日志（1545条）
- `data/agents/tools_descriptions.json` - 智能体tools描述

### 4. 运行最小实验（MVE）

**Mock模式**（测试流程，无需GPU）：
```bash
python -m experiments.run_mve --mock
```

**真实模式**（需要GPU）：
```bash
python -m experiments.run_mve
```

MVE配置：5代 × 5个体 × 20条评估，约15-30分钟

### 5. 查看架构文档

详细的架构设计说明见：[ARCHITECTURE.md](ARCHITECTURE.md)

## 项目结构

```
wyf-exp1/
├── config.py                   # 配置文件（需修改）
├── requirements.txt            # Python依赖
├── ARCHITECTURE.md            # 架构设计文档
│
├── structured_profile/        # 结构化描述建模
├── evaluation/                # 评估流水线（分类器、指标）
├── whitebox_init/            # 白盒初始化
├── evolution/                # 遗传算法
├── baselines/                # 基线方法（预留）
├── experiments/              # 实验脚本
│   └── run_mve.py           # 最小实验
│
├── data/                     # 数据目录
│   ├── agents/
│   │   └── tools_descriptions.json
│   ├── GOLDEN_TEST.csv
│   └── HISTORICAL_LOGS.csv
│
└── results/                  # 结果输出
```

## 配置说明

### 必需配置

1. **GPU服务器地址**：修改 `config.py` 中的 `BACKBONE_API_URL`
2. **Tools描述**：检查 `data/agents/tools_descriptions.json` 是否完整

### 可选配置

- **DeepSeek API Key**：用于LLM变异，提高效果
- **实验参数**：种群大小、代数等可在 `config.py` 中调整

## 下一步

1. ✓ 运行MVE验证流程
2. → 运行完整演化实验（20代 × 20个体）
3. → 执行对比实验（Exp1-Exp4）
4. → 执行消融实验（Ab1-Ab4）
5. → 生成可视化报告

## 常见问题

**Q: Mock模式和真实模式有什么区别？**
A: Mock模式使用随机分类器，不调用GPU，仅用于验证代码流程。真实模式会连接GPU服务器进行实际分类。

**Q: 实验需要多长时间？**
A: MVE约15-30分钟，完整实验约15-20 GPU小时。

**Q: 没有DeepSeek API Key能用吗？**
A: 可以，会回退到基于规则的变异，效果略差但仍可运行。

**Q: 如何验证实验结果？**
A: 结果保存在 `results/` 目录，包含最佳描述集和演化历史曲线。

## 文档

- **架构设计**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **实验说明**: [walkthrough.md](../walkthrough.md)
- **实现计划**: [implementation_plan.md](../implementation_plan.md)

## 作者

wyf-exp1 Team
