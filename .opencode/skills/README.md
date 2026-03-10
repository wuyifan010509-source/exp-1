# Claude Skills 使用指南

本目录包含 **76个** 专业的 Claude 技能（Skills），用于协助各种软件工程和 AI 研究任务。

---

## 📋 目录

- [通用开发技能 (6个)](#通用开发技能)
- [AI 研究技能 (70个)](#ai-研究技能)
  - [模型架构](#模型架构)
  - [分布式训练](#分布式训练)
  - [推理服务](#推理服务)
  - [模型优化](#模型优化)
  - [RAG & 向量数据库](#rag--向量数据库)
  - [多模态](#多模态)
  - [数据处理](#数据处理)
  - [模型评估](#模型评估)
  - [MLOps](#mlops)
  - [Prompt 工程](#prompt-工程)
  - [Tokenization](#tokenization)
  - [基础设施](#基础设施)
  - [新兴技术](#新兴技术)
  - [可解释性](#可解释性)
  - [论文写作与研究](#论文写作与研究)

---

## 通用开发技能

> 这些技能适用于所有类型的软件开发工作，提供最佳实践和工作流程指导。

| 技能名 | 说明 | 何时使用 |
|--------|------|----------|
| **humanizer** | 去除 AI 生成文本的典型特征，使文本更自然 | 编辑或润色 AI 生成的文字内容 |
| **drawio** | 创建和编辑 draw.io 流程图，支持实时浏览器预览 | 需要绘制流程图、架构图、UML 图 |
| **pdf** | 全面的 PDF 处理工具包 | 提取 PDF 文本/表格、创建 PDF、填写表单 |
| **skills-manager** | 技能管理器，列出所有可用技能 | 查看有哪些技能可用 |

### Superpowers（开发最佳实践）

| 技能名 | 说明 | 何时使用 |
|--------|------|----------|
| **brainstorming** | 任何创造性工作前的头脑风暴 | 创建功能、构建组件、添加功能前 |
| **writing-plans** | 编写多步骤任务的实施计划 | 有明确需求但需规划执行步骤时 |
| **executing-plans** | 执行已编写的实施计划 | 已有书面计划需要分步执行 |
| **test-driven-development** | 测试驱动开发 | 实现功能或修复 bug 前写测试 |
| **subagent-driven-development** | 使用子代理执行独立任务 | 当前会话有多个独立任务时 |
| **dispatching-parallel-agents** | 并行派遣多个代理 | 有 2+ 个无依赖的独立任务时 |
| **systematic-debugging** | 系统性调试方法论 | 遇到 bug、测试失败或异常行为 |
| **verification-before-completion** | 完成前验证工作 | 声称工作完成前必须运行验证 |
| **receiving-code-review** | 处理代码审查反馈 | 收到 code review 需要实现建议时 |
| **requesting-code-review** | 请求代码审查 | 完成任务、实现功能或合并前 |
| **using-git-worktrees** | 使用 Git worktrees 隔离工作 | 需要隔离当前工作空间时 |
| **finishing-a-development-branch** | 完成开发分支 | 实现完成、测试通过后的收尾工作 |

---

## AI 研究技能

### 模型架构

> 用于实现和理解各种 LLM 架构的代码库。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **nanogpt** | Andrej Karpathy 的 300 行教育版 GPT 实现 | 学习 Transformer 架构，从头理解 GPT |
| **litgpt** | Lightning AI 的 20+ 预训练架构实现 | 需要清晰的模型实现，生产级微调 |
| **mamba** | 状态空间模型，O(n) 复杂度 vs Transformer O(n²) | 需要长序列建模（百万 token），无 KV cache |
| **rwkv** | RNN+Transformer 混合，线性时间推理 | 需要无限上下文、O(n) 推理、训练并行化 |
| **torchtitan** | PyTorch 原生分布式 LLM 预训练 | 8-512+ GPU 规模预训练 Llama 3.1 等 |

### 分布式训练

> 大规模模型训练框架和优化技术。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **deepspeed** | Microsoft DeepSpeed，ZeRO 优化 | 内存受限的大模型训练 |
| **accelerate** | HuggingFace 统一分布式训练 API | 最简单的方式添加分布式支持 |
| **pytorch-fsdp2** | PyTorch FSDP2 (fully_shard) | 模型超出单卡内存，需要 DTensor 分片 |
| **megatron-core** | NVIDIA Megatron-Core | 训练 2B-462B 参数模型，最大 GPU 效率 |
| **pytorch-lightning** | 高级 PyTorch 训练框架 | 从笔记本扩展到超级计算机 |
| **ray-train** | Ray 分布式训练编排 | 1000+ 节点的大规模分布式训练 |

### 推理服务

> 生产环境的 LLM 部署和推理优化。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **vllm** | 高吞吐量 LLM 服务（PagedAttention） | 生产 API 部署，优化延迟/吞吐量 |
| **sglang** | 快速结构化生成和推理 | JSON/正则输出，约束解码，工具调用 |
| **llama-cpp** | 在 CPU/Apple Silicon 上运行 LLM | 边缘部署，无 NVIDIA GPU 场景 |
| **tensorrt-llm** | NVIDIA TensorRT LLM 优化 | A100/H100 上的最大吞吐量和最低延迟 |

### 模型优化

> 模型压缩、量化、注意力优化技术。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **flash-attention** | Flash Attention 优化 | 长序列 (>512 tokens)，GPU 内存问题 |
| **bitsandbytes** | 8-bit / 4-bit 量化 | GPU 内存受限，QLoRA 训练 |
| **gptq** | 4-bit 后训练量化 | 70B/405B 模型在消费级 GPU 上部署 |
| **awq** | 激活感知权重量化 | 比 GPTQ 更快，精度损失更小 |
| **hqq** | 半二次量化，无需校准数据 | 快速量化工作流，vLLM/HF 部署 |
| **gguf** | llama.cpp 量化格式 | 消费者硬件、Apple Silicon、CPU 推理 |

### RAG & 向量数据库

> 检索增强生成和向量相似性搜索。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **sentence-transformers** | 最先进的句子/文本/图像嵌入 | 生成 RAG 嵌入，语义搜索 |
| **faiss** | Facebook 向量相似性搜索 | 十亿级向量，GPU 加速 |
| **qdrant** | 高性能向量搜索引擎 | 生产 RAG 系统，Rust 驱动 |
| **pinecone** | 托管向量数据库 | 生产级无服务器，<100ms 延迟 |
| **chroma** | 开源嵌入数据库 | 本地开发，开源项目 |

### 多模态

> 视觉-语言模型和跨模态 AI。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **clip** | OpenAI 视觉-语言模型 | 零样本图像分类，图文匹配 |
| **llava** | 大型语言和视觉助手 | 图像对话，视觉问答 |
| **stable-diffusion** | 文本到图像生成 | 图像生成，图像翻译，修复 |
| **blip-2** | 视觉-语言预训练 | 图像描述，视觉问答，图文检索 |
| **segment-anything** | 零样本图像分割 | 自动分割图像中的任意对象 |
| **whisper** | OpenAI 语音识别模型 | 99 种语言的语音转文本 |
| **audiocraft** | 音频生成（MusicGen, AudioGen） | 文本到音乐，文本到音效 |

### 数据处理

> 大规模数据集清洗和预处理。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **nemo-curator** | NVIDIA GPU 加速数据清洗 | 模糊去重，质量过滤，PII 脱敏 |
| **ray-data** | 可扩展的数据处理 | 批量推理，数据预处理，多模态加载 |

### 模型评估

> 学术基准测试和模型质量评估。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **evaluating-llms-harness** | EleutherAI lm-evaluation-harness | 60+ 学术基准（MMLU, HumanEval, GSM8K） |
| **nemo-evaluator** | NVIDIA NeMo 模型评估 | NeMo 框架的端到端评估 |
| **bigcode-evaluation-harness** | 代码模型评估 | HumanEval, MBPP 等代码基准 |

### MLOps

> 机器学习实验跟踪和模型管理。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **mlflow** | 实验跟踪和模型注册 | 跟踪指标，版本化模型，生产部署 |
| **tensorboard** | 训练可视化和调试 | 训练指标，模型图，性能分析 |
| **weights-and-biases** | W&B 实验跟踪平台 | 实时可视化，超参数调优 |

### Prompt 工程

> 结构化输出和 LLM 输出控制。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **dspy** | 声明式 LLM 编程 | 模块化 RAG，自动提示优化 |
| **instructor** | 带 Pydantic 验证的结构化输出 | 提取结构化数据，类型安全 |
| **outlines** | 保证有效 JSON/XML/代码 | 本地模型，结构化生成 |
| **guidance** | 正则和语法约束生成 | 微软的约束解码框架 |

### Tokenization

> 文本分词和词汇表训练。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **sentencepiece** | 语言无关的分词器 | 多语言，CJK 语言，可复现性 |
| **huggingface-tokenizers** | Rust 实现的高速分词 | 1GB 文本 <20 秒，自定义词汇表 |

### 基础设施

> 云 GPU 和训练环境管理。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **modal-serverless-gpu** | Modal 无服务器 GPU 云 | 按需 GPU，自动扩展 |
| **skypilot-multi-cloud-orchestration** | SkyPilot 多云编排 | 跨云训练，抢占实例恢复 |
| **lambda-labs-gpu-cloud** | Lambda Labs GPU 云 | 预留实例，SSH 访问，多节点集群 |

### 新兴技术

> 最新的 LLM 研究和优化技术。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **speculative-decoding** | 推测性解码加速 | 3x 推理加速，不损失精度 |
| **knowledge-distillation** | 知识蒸馏 | 大模型压缩到小模型 |
| **model-merging** | 模型融合技术 | 组合多个微调模型 |
| **model-pruning** | 模型剪枝 | 减少模型参数和计算量 |
| **moe-training** | 专家混合模型训练 | 稀疏激活的大规模模型 |
| **long-context** | 长上下文建模技术 | 处理超长序列 |

### 可解释性

> 神经网络的机制解释和分析。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **transformer-lens** | Transformer 可解释性工具 | 激活分析，注意力可视化 |
| **nnsight** | 神经网络内部检查 | 逐层分析，特征提取 |
| **pyvene** | 因果干预工具 | 模型编辑，因果分析 |
| **saelens** | 稀疏自编码器训练 | 学习可解释特征 |

### 论文写作与研究

> 学术论文撰写和研究方法论。

| 技能名 | 说明 | 适用场景 |
|--------|------|----------|
| **ml-paper-writing** | ML/AI 论文写作 | NeurIPS, ICML, ICLR, ACL 等投稿 |
| **brainstorming-research-ideas** | 研究想法头脑风暴 | 寻找研究方向和创新点 |
| **creative-thinking-for-research** | 研究的创造性思维 | 突破传统思路 |

---

## 🔍 如何选择合适的技能

### 按任务类型选择

| 任务类型 | 推荐技能 |
|----------|----------|
| **文本生成/对话** | litgpt, vllm, sglang |
| **长上下文处理** | mamba, flash-attention, long-context |
| **模型部署** | vllm, tensorrt-llm, llama-cpp |
| **RAG 应用** | sentence-transformers + qdrant/faiss/chroma |
| **多模态应用** | clip, llava, blip-2, whisper |
| **模型训练** | deepspeed, accelerate, megatron-core |
| **数据准备** | nemo-curator, ray-data |
| **模型评估** | evaluating-llms-harness |
| **模型压缩** | gptq, awq, bitsandbytes |
| **代码生成** | evaluating-llms-harness (HumanEval) |

### 按资源限制选择

| 资源限制 | 推荐技能 |
|----------|----------|
| **GPU 内存有限** | bitsandbytes, gptq, awq, flash-attention |
| **无 GPU / CPU 部署** | llama-cpp, gguf |
| **大规模集群** | megatron-core, deepspeed, ray-train |
| **边缘设备** | llama-cpp, model-pruning, knowledge-distillation |

### 按开发阶段选择

| 开发阶段 | 推荐技能 |
|----------|----------|
| **研究/原型** | nanogpt, litgpt, accelerate |
| **生产部署** | vllm, sglang, tensorrt-llm |
| **大规模训练** | megatron-core, deepspeed, torchtitan |
| **监控/调试** | tensorboard, mlflow, weights-and-biases |

---

## 🚀 快速开始

使用技能非常简单，在对话中直接请求即可：

```
用户：帮我用 vllm 部署一个 Llama 模型
用户：使用 flash-attention 优化我的 transformer
用户：用 ml-paper-writing 帮我写一篇 NeurIPS 论文
```

Claude 会自动加载对应的技能并提供专业指导。

---

## 📝 贡献和更新

此技能库持续更新，新增技能会自动出现在此目录下。如需了解最新技能，可使用：

```bash
~/.opencode/skills/skills-manager/scripts/list_skills.py
```

---

*最后更新：2026年3月1日*
*总计技能数：76个*
