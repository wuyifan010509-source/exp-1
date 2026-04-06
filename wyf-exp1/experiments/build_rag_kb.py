"""
生成RAG知识库脚本
从训练数据或人工标注数据构建RAG知识库

RAG知识库格式：
[
    {
        "query": "用户问题（用于匹配）",
        "answer": "分析结论（作为上下文）",
        "intent": "意图类别（可选）"
    },
    ...
]
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict

# 导入RAG模块
from rag_local_embedding import SimpleRAG, LocalEmbeddingModel


def build_kb_from_csv(csv_path: str, query_col: str = "问题", answer_col: str = "分析", 
                      intent_col: str = "预期意图") -> List[Dict]:
    """
    从CSV文件构建知识库
    
    Args:
        csv_path: CSV文件路径
        query_col: 问题列名
        answer_col: 答案/分析列名
        intent_col: 意图列名（可选）
        
    Returns:
        metadata列表
    """
    print(f"[Build KB] 从CSV加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    metadata_list = []
    for _, row in df.iterrows():
        item = {
            "query": str(row[query_col]).strip(),
            "answer": str(row[answer_col]).strip() if pd.notna(row[answer_col]) else "",
        }
        if intent_col in df.columns:
            item["intent"] = str(row[intent_col]).strip()
        metadata_list.append(item)
    
    print(f"[Build KB] 共加载 {len(metadata_list)} 条记录")
    return metadata_list


def build_kb_from_json(json_path: str) -> List[Dict]:
    """
    从JSON文件构建知识库
    
    Args:
        json_path: JSON文件路径，格式为 [{"query": "...", "answer": "..."}, ...]
        
    Returns:
        metadata列表
    """
    print(f"[Build KB] 从JSON加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    print(f"[Build KB] 共加载 {len(metadata_list)} 条记录")
    return metadata_list


def build_kb_from_manual_data() -> List[Dict]:
    """
    从手动标注的数据构建知识库
    这里可以添加你认为对分类有帮助的案例
    """
    print("[Build KB] 使用手动标注数据")
    
    # 手动标注的典型案例
    # query: 用于匹配的问题
    # answer: 分析结论，会作为上下文拼接到prompt中
    metadata_list = [
        {
            "query": "603703能进吗",
            "answer": "用户输入股票名称、股票代码，是在请求系统给出该股票综合评价",
        },
        {
            "query": "计算机板块",
            "answer": "用户输入板块名称，是在请求系统列出该板块成分股，属于选股类",
        },
        {
            "query": "贵州茅台前景",
            "answer": "用户问题带有'往后'、'未来'、'明天'、'下一步'、'前景'等词汇，且提及了具体的股票。是在对该股票未来的预测请求。如果没有具体股票则肯定不是预测类！如'明天买什么股票'不是预测类，是推荐类",
        },
        {
            "query": "沪深300股票有哪些？",
            "answer": "用户只输入选股条件，是在请求筛选该条件的股票",
        },
        {
            "query": "在哪里看北向资金的实时动向？",
            "answer": " 用户提及‘擒龙平台’、‘黄金坑’，是在询问软件操作知识",
        },
        {
            "query": "什么是MACD指标的底背离？",
            "answer": " 用户提及‘大盘量能’、‘价值决策’，是在询问这个概念的含义",
        },
        {
            "query": "红利轮动策略",
            "answer": "用户提及‘xx策略’，要转为策略类。",
        },
        {
            "query": "东方财富的板块",
            "answer": "用户提及如‘贵州茅台(股票）的板块(指标名称）’,是在查询这指标",
        }
    ]
    
    print(f"[Build KB] 共 {len(metadata_list)} 条手动标注记录")
    return metadata_list


def save_kb(metadata_list: List[Dict], output_path: str, embedding_dim: int = 1024):
    """
    保存知识库为RAG格式（包含向量）
    
    Args:
        metadata_list: metadata列表
        output_path: 输出文件路径
        embedding_dim: Embedding维度
    """
    print(f"\n[Save KB] 正在生成向量并保存知识库...")
    
    # 初始化Embedding模型
    model = LocalEmbeddingModel()
    
    # 初始化RAG
    rag = SimpleRAG(
        embedding_model=model,
        dim=embedding_dim,
        key_field="query",
        value_field="answer"
    )
    
    # 添加文档（会自动编码）
    rag.add_documents(metadata_list)
    
    # 保存
    rag.save(output_path)
    print(f"[Save KB] 知识库已保存: {output_path}")
    
    # 同时保存纯JSON版本（方便查看）
    json_path = output_path.replace('.json', '_metadata.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    print(f"[Save KB] Metadata已保存: {json_path}")


def main():
    """
    主函数：构建RAG知识库
    """
    print("=" * 60)
    print("RAG知识库构建工具")
    print("=" * 60)
    
    # 方式1：从手动标注数据构建（推荐用于快速测试）
    print("\n[Option 1] 使用手动标注数据构建知识库")
    metadata_list = build_kb_from_manual_data()
    
    # 方式2：从CSV构建（如果你有标注好的CSV文件）
    # print("\n[Option 2] 从CSV文件构建知识库")
    # csv_path = "/path/to/your/labeled_data.csv"
    # metadata_list = build_kb_from_csv(
    #     csv_path, 
    #     query_col="问题", 
    #     answer_col="分析结论",
    #     intent_col="意图"
    # )
    
    # 方式3：从JSON构建（如果你已有JSON格式数据）
    # print("\n[Option 3] 从JSON文件构建知识库")
    # json_path = "/path/to/your/data.json"
    # metadata_list = build_kb_from_json(json_path)
    
    # 保存知识库
    output_path = "rag_kb.json"
    save_kb(metadata_list, output_path, embedding_dim=1024)  # bge-m3是1024维
    
    print("\n" + "=" * 60)
    print("知识库构建完成！")
    print("=" * 60)
    print(f"\n使用方式:")
    print(f"  python test_prompt.py --rag-kb {output_path}")
    print(f"\n可选参数:")
    print(f"  --rag-threshold 0.8    # 相似度阈值（默认0.8）")
    print(f"  --rag-top-k 1          # 返回结果数（默认1）")


if __name__ == "__main__":
    main()
