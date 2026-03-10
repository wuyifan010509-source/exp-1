---
name: brainstorm-logger
description: 头脑风暴对话记录器。用于在头脑风暴、研究构思、论证讨论时自动记录完整对话。激活时机：(1) 用户请求记录头脑风暴对话时；(2) 进行研究思路讨论、实验设计、论证分析时；(3) 与 brainstorming、research-ideation、21-research-ideation 等 skills 配合使用时。关键词：头脑风暴、brainstorm、研究思路、论证、讨论记录、对话保存、argument。
---

# Brainstorm Logger

头脑风暴对话记录器，自动将讨论过程保存到 `./argument` 文件夹。

## 工作流程

1. **识别主题** - 根据讨论内容确定主题名称（如"研究思路"、"实验设计"、"论文框架"）
2. **创建/追加文件** - 在 `./argument` 目录创建 `{主题}.qmd` 文件
3. **记录对话** - 按时间戳记录用户问题和Claude回答
4. **持续追加** - 同一主题的后续讨论追加到同一文件

## 记录格式

每个 .qmd 文件结构：

```markdown
# {主题名称}

## {YYYY-MM-DD HH:MM}

**用户**: {用户问题或观点}

**Claude**: {Claude回答}

---

## {YYYY-MM-DD HH:MM} (后续讨论)

**用户**: ...

**Claude**: ...

---
```

## 使用方法

### 开始记录

当开始头脑风暴时，调用脚本创建或追加记录：

```bash
uv run python scripts/save_conversation.py --topic "主题名称" --user "用户输入" --assistant "Claude回复"
```

### 参数说明

- `--topic`: 主题名称（用于生成文件名）
- `--user`: 用户的问题或观点
- `--assistant`: Claude的回答

### 示例

```bash
# 记录关于实验设计的讨论
uv run python scripts/save_conversation.py \
  --topic "实验设计" \
  --user "我想设计一个对比实验来验证假设" \
  --assistant "好的，让我们来设计这个实验..."
```

## 自动激活条件

此 skill 应在以下情况自动激活：

1. 用户明确要求记录讨论
2. 检测到研究性讨论（关键词：研究、实验、假设、论证、思路）
3. 其他研究相关 skills 被激活时（如 brainstorming, research-ideation）

## 文件位置

所有记录保存在项目根目录的 `./argument/` 文件夹中：

```
project/
└── argument/
    ├── 研究思路.qmd
    ├── 实验设计.qmd
    └── 论文框架.qmd
```

## 注意事项

- 每条记录包含时间戳，便于追溯讨论历程
- 同一主题的多次讨论追加到同一文件，保持连贯性
- 文件使用 .qmd 格式（Quarto Markdown），支持后续渲染为多种格式
