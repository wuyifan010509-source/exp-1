# AGENTS.md - 项目指南

## ⚠️ 关键提醒


### UV 项目管理

**添加依赖（不要使用 pip）：**
```bash
uv add <package>                    # 生产依赖
uv add --dev <package>              # 开发依赖
```

**运行脚本（必须使用 uv run）：**
```bash
uv run python script.py
uv run python -m module_name
uv run llamafactory-cli train config.yaml
```

**常见操作：**
```bash
uv sync                             # 同步依赖
uv run python -m src.synthetic_data # 数据生成
uv run python scripts/convert_to_user_simulator.py  # 数据转换
uv run python scripts/train_and_monitor.py attention_only  # 训练
```

### 项目目录结构

```
├── data/                        # 数据文件
│   ├── checkpoints/            # 数据生成检查点
│   ├── processed/              # 处理后的数据 (final_labeled_data.jsonl)
│   ├── raw/                    # 原始数据 (oos_*.csv)
│   └── validation/             # 验证集 (validation_set_*.jsonl)
├── exp/                         # 实验代码
│   ├── data_labeling/          # 数据标注模块
│   ├── evaluation/             # 评估模块
│   ├── semantic_routing/       # 语义路由模块
│   ├── slm_distillation/       # SLM蒸馏模块
│   └── reports/                # 实验报告
├── logs/                        # 日志文件
├── scripts/                     # 独立脚本工具
└── output/                      # 模型输出
```

**完整目录说明见:** `PROJECT_STRUCTURE.md`

### 运行命令
```bash
# 允许操作，自动化执行
claude --permission-mode bypassPermissions
```

已完成三份详细的规划文档，互相独立又交叉引用：

文档	内容
walkthrough.md	做成什么算成功？量化指标（Accuracy/边界Accuracy/Margin）、端到端运行流程预览、关键技术细节（分类调用方式、GA参数、推理代价估算 ~212K次→优化方案、LLM变异Prompt模板）、与 INTENT_GPT 基线的对比
design_rationale.md	为什么这么做？集合论/SVM类比/信息瓶颈/注意力稀释 → 论证 P={C,B,R} 的完备性；离散不可导 → 论证 GA 的必要性；冷启动问题 → 论证白盒初始化的价值；Margin Sampling → 闭环优化
implementation_plan.md (v2)	如何做？已绑定 exp_info.md 的真实参数（354条/1500条/11标签/qwen2.5-32b/4×L20/Gemini Pro），详细的模块接口设计、服务器部署步骤、requirements.txt

请评审这三份文档，特别关注：

推理代价：walkthrough 3.4 节估算了 ~120-240h GPU 时间，提出了 4 种优化策略（缩小评估子集、批量推理、提前停止、缓存），是否可接受？
