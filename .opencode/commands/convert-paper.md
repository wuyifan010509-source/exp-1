---
description: 将 PDF 论文转换为 Markdown 文件（PDF → PNG → Markdown）
---

# Convert Paper Command

当用户输入 `@convert-paper <pdf_path>` 时，执行以下流程：

## 执行流程

### 步骤 1: PDF 转图片
调用 Python 脚本将 PDF 转换为图片：
```bash
uv run scripts/convert_pdf_to_img.py <pdf_path>
```

这会在 PDF 同级目录生成 `{pdf_name}_png/` 文件夹，包含所有页面图片（如 `xxx_001.png`, `xxx_002.png`...）。

### 步骤 2: 创建 Markdown 输出文件夹
在 PDF 同级目录创建 `{pdf_name}_md/` 文件夹，用于存放转换后的 Markdown 文件。

例如：
- PDF: `papers/Agent4Rec.pdf`
- PNG 文件夹: `papers/Agent4Rec_png/`
- MD 文件夹: `papers/Agent4Rec_md/`

### 步骤 3: 扫描 PNG 文件
列出 `{pdf_name}_png/` 文件夹中所有 `.png` 文件，按文件名排序。

### 步骤 4: 主 AI 直接处理 Markdown 转换
按文件名顺序逐个处理每张图片：

1. 读取 PNG 文件内容
2. 应用「单文件转换规范」转换为 Markdown
3. 保存到 `{pdf_name}_md/` 文件夹（同名，仅改扩展名）
4. **如果任一文件处理失败，立即停止整个流程并报错**

处理完一个再处理下一个，保持顺序。

### 步骤 5: 返回汇总结果
所有文件处理完成后，返回：
```
PDF 转换完成：{pdf_name}.pdf

📄 生成图片：{N} 张
   📁 位置：{pdf_path}_png/

✓ Markdown 转换：{N}/{N} 成功
   📁 位置：{pdf_path}_md/

📂 输出文件：
   - {name}_001.md
   - {name}_002.md
   - ...
```

---

## 单文件转换规范

处理每张图片时遵循以下规范：

### 1. 文字提取顺序
- 识别双列排版，按**先左列后右列**的顺序提取文字
- 保持原文段落结构，不合并不同段落
- 保留原文的章节标题层级（如 4.2、4.3）
- **重要：纯文字部分直接提取原文，不要添加任何理解、总结或额外解释**

### 2. 表格处理（Table）
- 将表格转换为 Markdown 表格格式
- 保留表格标题（如：Table 5: Statistics of...）
- 在表格后添加 blockquote 详细描述，包括：
  - 表格对比的核心内容
  - 关键数据差异（如数值对比、倍数关系）
  - 得出的结论或观察

### 3. 图表处理（Figure）
- 保留图表标题（如：Figure 2: Comparison between...）
- 在标题后添加 blockquote 详细描述，包括：
  - **图表类型**：箱线图/柱状图/折线图/流程图等
  - **坐标轴说明**：X轴、Y轴含义及数值范围
  - **数据系列**：各组数据的表现（如中位数、分布范围）
  - **关键结论**：图表传达的核心发现
  - **子图划分**：如有(a)(b)子图，分别描述

### 4. 格式规范
- **纯文字段落**：直接提取原文，保持原样，**不要添加任何理解、总结或解释**
- 使用 **粗体** 标注重点术语和图表标题
- 使用 *斜体* 标注变量或强调内容
- 数学公式保留 LaTeX 格式（如 $q_0$、$a_i$）
- 使用 `>` blockquote 包裹所有图表的详细描述（**只有遇到图表时才需要添加详细描述**）

### 5. 输出要求
- 保存为 `.md` 文件
- **输出文件夹**：将 Markdown 文件保存到 `_md` 文件夹
  - 例如：`paper1_png/xxx_001.png` → `paper1_md/xxx_001.md`
- **文件名格式：与原始 PNG 图片文件名保持一致，仅将扩展名从 `.png` 改为 `.md`**
- 内容完整，不遗漏任何文字段落

---

## 输出格式示例

### 表格示例

```markdown
**Table 5:** Statistics of the collected dataset...

| 列1 | 列2 | 列3 |
|:---|:---:|:---:|
| 数据 | 数据 | 数据 |

> **表格详细描述：**
>
> 该表格对比了...的核心统计指标：
> - 关键发现1（含具体数值对比）
> - 关键发现2
> - 结论：...
```

### 图表示例

```markdown
**Figure 2:** Comparison between...

> **图表内容详细描述：**
>
> 该图包含两个并排的箱线图...：
>
> **(a) 子图标题：**
> - Y轴：含义及范围
> - 数据对比：左侧...右侧...
> - 结论：...
>
> **(b) 子图标题：**
> - ...
```

---

## 处理要点总结

| 元素 | 处理方式 |
|:---|:---|
| **输入** | 单个 PDF 文件路径 |
| **PDF 转图片** | 调用 `convert_pdf_to_img.py` 脚本 |
| **图片输出** | `{pdf_name}_png/` 文件夹 |
| **Markdown 输出** | `{pdf_name}_md/` 文件夹 |
| **图片质量** | 2x 缩放，确保文字清晰（固定，不可调） |
| **处理方式** | 主 AI 直接处理，不调用 agent |
| **错误处理** | 任一文件失败立即停止并报错 |
| **双列排版** | 先左后右，逐列提取 |
| **表格** | Markdown 表格 + 详细描述 blockquote |
| **图表** | 保留标题 + 详细描述 blockquote |
| **章节标题** | 保持层级（### 4.2 XXX）|
| **公式变量** | LaTeX 格式（$...$）|
| **强调内容** | 粗体或斜体 |
| **描述块** | 统一使用 `>` blockquote 格式 |
| **输出文件名** | 与源图片同名，仅改扩展名 |
| **中间图片** | 保留 `_png` 文件夹 |
| **Markdown 合并** | 无需合并，保持分页输出 |

---

## 使用示例

**用户输入：**
```
@convert-paper papers/Agent4Rec/Agent4Rec.pdf
```

**AI 执行步骤：**
1. 调用 `uv run scripts/convert_pdf_to_img.py papers/Agent4Rec/Agent4Rec.pdf`
2. 生成图片到 `papers/Agent4Rec/Agent4Rec_png/` 文件夹
3. 创建 `papers/Agent4Rec/Agent4Rec_md/` 文件夹
4. 扫描到 12 张 PNG 文件
5. 主 AI 直接逐个处理图片转 Markdown（第1张 → 第2张 → ... → 第12张）
6. 全部完成后返回：\n   ```
   PDF 转换完成：Agent4Rec.pdf

   📄 生成图片：12 张
      📁 位置：papers/Agent4Rec/Agent4Rec_png/

   ✓ Markdown 转换：12/12 成功
      📁 位置：papers/Agent4Rec/Agent4Rec_md/

   📂 输出文件：
      - Agent4Rec_001.md
      - Agent4Rec_002.md
      - ...
      - Agent4Rec_012.md
   ```
