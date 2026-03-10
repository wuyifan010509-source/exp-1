# 意图分类优化实验代码搭建

## 总体目标
根据 `wyf-exp1/requirement.md` 的研究方案，搭建"结构化描述+遗传算法演化"的意图分类优化实验代码框架。

## 任务分解

### 阶段零：规划设计
- [x] 阅读需求文档和 Gemini 对话记录
- [x] 分析已有 `wyf-exp2` 代码结构和模式
- [x] 阅读 `exp_info.md` 实验参数配置
- [x] 编写 `walkthrough.md`（做成什么算成功）
- [x] 编写 `design_rationale.md`（为什么这么做）
- [x] 编写 `implementation_plan.md` v2（如何做 — 含真实实验参数）
- [ ] 用户评审确认

### 阶段一：基础设施
- [ ] 创建项目目录结构和 `config.py`
- [ ] 创建 `requirements.txt`
- [ ] 结构化描述模块 `structured_profile/profile.py`

### 阶段二：核心模块实现
- [ ] 评估流水线 `evaluation/`（classifier + metrics + pipeline）
- [ ] 白盒初始化 `whitebox_init/`（tool_parser + prompt_compressor + metadata_extractor）
- [ ] 遗传算法引擎 `evolution/`（GA + LLM mutator + margin_sampling）

### 阶段三：基线 & 实验
- [ ] 基线方法 `baselines/`（人工描述、INTENT_GPT、随机压缩）
- [ ] 实验脚本 `experiments/`（对比、消融、可视化）
- [ ] Mock 数据准备

### 阶段四：验证
- [ ] 单元测试
- [ ] Mock 模式端到端验证
