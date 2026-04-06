"""
最简单的RAG匹配实现 - 纯Python，无需向量数据库
只需numpy，适合文档数量 < 1万条的场景
"""
import numpy as np
from typing import List, Tuple, Optional
import json


class SimpleRAG:
    """
    极简RAG检索器
    
    原理：
    1. 将所有文档编码成向量
    2. 查询时也编码成向量
    3. 计算查询向量与所有文档向量的余弦相似度
    4. 返回最相似的top-k个文档
    """
    
    def __init__(self, embedding_model=None, dim: int = 384):
        """
        Args:
            embedding_model: 你的embedding模型，需要实现 encode(texts: List[str]) -> np.ndarray
            dim: 向量维度（如果你的模型是768维就填768）
        """
        self.model = embedding_model
        self.dim = dim
        self.documents: List[str] = []      # 存储原文
        self.vectors: np.ndarray = None     # 存储向量 (n_docs, dim)
        self._is_normalized = False         # 标记向量是否已归一化
    
    def add_documents(self, texts: List[str], vectors: Optional[np.ndarray] = None):
        """
        添加文档到知识库
        
        Args:
            texts: 文档文本列表
            vectors: 如果传入，则直接使用这些向量；否则用embedding_model编码
        """
        if not texts:
            return
        
        # 1. 获取向量
        if vectors is not None:
            # 直接传入向量
            new_vectors = np.array(vectors, dtype=np.float32)
        elif self.model is not None:
            # 用模型编码
            print(f"正在编码 {len(texts)} 篇文档...")
            new_vectors = self.model.encode(texts)
            new_vectors = np.array(new_vectors, dtype=np.float32)
        else:
            raise ValueError("请提供 embedding_model 或 vectors")
        
        # 2. 归一化向量（为了计算余弦相似度）
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_vectors = new_vectors / (norms + 1e-8)
        
        # 3. 添加到现有数据
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        
        self.documents.extend(texts)
        self._is_normalized = True
        
        print(f"✓ 已添加 {len(texts)} 篇文档，当前共 {len(self.documents)} 篇")
    
    def search(self, query: str, query_vector: Optional[np.ndarray] = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        搜索最相似的文档
        
        Args:
            query: 查询文本
            query_vector: 如果传入，则直接使用；否则用embedding_model编码
            top_k: 返回最相似的前k个文档
        
        Returns:
            [(document_text, similarity_score), ...]  按相似度降序排列
        """
        if len(self.documents) == 0:
            return []
        
        # 1. 获取查询向量
        if query_vector is not None:
            q_vector = np.array(query_vector, dtype=np.float32)
        elif self.model is not None:
            q_vector = self.model.encode([query])[0]
            q_vector = np.array(q_vector, dtype=np.float32)
        else:
            raise ValueError("请提供 embedding_model 或 query_vector")
        
        # 2. 归一化查询向量
        q_vector = q_vector / (np.linalg.norm(q_vector) + 1e-8)
        
        # 3. 计算余弦相似度（归一化后的向量点积 = 余弦相似度）
        similarities = np.dot(self.vectors, q_vector)  # shape: (n_docs,)
        
        # 4. 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 5. 组装结果
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save(self, filepath: str):
        """保存知识库到文件"""
        data = {
            'documents': self.documents,
            'vectors': self.vectors.tolist() if self.vectors is not None else [],
            'dim': self.dim
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"✓ 知识库已保存到: {filepath}")
    
    def load(self, filepath: str):
        """从文件加载知识库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data['documents']
        if data['vectors']:
            self.vectors = np.array(data['vectors'], dtype=np.float32)
        self.dim = data.get('dim', self.dim)
        self._is_normalized = True
        print(f"✓ 已加载 {len(self.documents)} 篇文档")


# ==================== 使用示例 ====================

def example_with_mock_model():
    """示例1：使用模拟的embedding模型（演示用）"""
    print("=" * 60)
    print("示例1：基础用法（模拟embedding模型）")
    print("=" * 60)
    
    # 1. 定义模拟的embedding模型
    class MockEmbeddingModel:
        """
        模拟embedding模型
        实际使用时，替换为你自己的模型，例如：
        - OpenAI API
        - HuggingFace模型  
        - 本地API服务
        """
        def __init__(self, dim=384):
            self.dim = dim
            # 模拟一些关键词到固定向量的映射，使演示更真实
            self.keywords = {
                'rag': [1, 0, 0, 0] * (dim // 4),
                '向量': [0, 1, 0, 0] * (dim // 4),
                '数据库': [0, 1, 1, 0] * (dim // 4),
                '检索': [1, 1, 0, 0] * (dim // 4),
                'python': [0, 0, 1, 0] * (dim // 4),
                '机器学习': [0, 0, 0, 1] * (dim // 4),
                'ai': [1, 0, 0, 1] * (dim // 4),
            }
        
        def encode(self, texts: List[str]) -> np.ndarray:
            """
            模拟编码：根据文本中的关键词生成向量
            实际使用时，替换为真实embedding模型调用
            """
            vectors = []
            for text in texts:
                # 简单的模拟：根据关键词混合向量
                vec = np.random.randn(self.dim).astype(np.float32) * 0.1
                text_lower = text.lower()
                for keyword, base_vec in self.keywords.items():
                    if keyword in text_lower:
                        vec += np.array(base_vec[:self.dim]) * 0.5
                vectors.append(vec)
            return np.array(vectors)
    
    # 2. 初始化
    model = MockEmbeddingModel(dim=384)
    rag = SimpleRAG(embedding_model=model, dim=384)
    
    # 3. 添加文档
    documents = [
        "RAG（检索增强生成）是一种结合检索系统和生成模型的技术",
        "向量数据库用于高效存储和检索高维向量数据",
        "余弦相似度是衡量两个向量方向相似性的指标",
        "Python是数据科学领域最流行的编程语言之一",
        "机器学习是人工智能的一个重要分支",
        "大语言模型如GPT、Claude可以用于各种NLP任务",
        "Embedding模型将文本转换为数值向量表示"
    ]
    rag.add_documents(documents)
    
    # 4. 搜索测试
    test_queries = [
        "什么是RAG技术？",
        "如何存储向量数据？",
        "Python和机器学习",
        "大模型怎么用？"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        results = rag.search(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.4f}] {doc}")
    
    # 5. 保存和加载
    rag.save("knowledge_base.json")
    
    # 新建实例并加载
    rag2 = SimpleRAG(dim=384)
    rag2.load("knowledge_base.json")
    print(f"\n加载后文档数: {len(rag2.documents)}")


def example_with_precomputed_vectors():
    """示例2：使用预计算的向量（跳过编码步骤）"""
    print("\n" + "=" * 60)
    print("示例2：使用预计算向量（无需模型）")
    print("=" * 60)
    
    # 假设你已经有现成的向量（比如从文件加载的）
    texts = [
        "这是一篇关于股票的文章",
        "这是一篇关于债券的文章",
        "这是一篇关于基金的文章"
    ]
    
    # 模拟预计算的向量（实际使用时是你的真实向量）
    vectors = np.array([
        [1.0, 0.5, 0.2, 0.1],  # 股票
        [0.9, 0.6, 0.3, 0.1],  # 债券（与股票相似）
        [0.2, 0.1, 0.9, 0.8],  # 基金（不同）
    ])
    
    # 创建RAG实例（不传入模型）
    rag = SimpleRAG(dim=4)
    
    # 直接传入文本和向量
    rag.add_documents(texts, vectors=vectors)
    
    # 搜索时也要传入向量
    query_vec = np.array([1.0, 0.5, 0.3, 0.1])  # 股票相关的查询向量
    results = rag.search("股票查询", query_vector=query_vec, top_k=2)
    
    print(f"\n🔍 查询向量: {query_vec}")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {doc}")


def example_api_integration():
    """示例3：接入真实的API模型（以OpenAI为例）"""
    print("\n" + "=" * 60)
    print("示例3：接入OpenAI Embedding API（代码框架）")
    print("=" * 60)
    print("""
# 安装: pip install openai

import openai
import numpy as np

class OpenAIEmbedding:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        openai.api_key = api_key
        self.model = model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        # 调用OpenAI API
        response = openai.Embedding.create(
            model=self.model,
            input=texts
        )
        # 提取向量
        vectors = [item['embedding'] for item in response['data']]
        return np.array(vectors, dtype=np.float32)

# 使用
model = OpenAIEmbedding(api_key="your-api-key")
rag = SimpleRAG(embedding_model=model, dim=1536)  # ada-002是1536维

# 后续使用与示例1相同
""")


if __name__ == "__main__":
    # 运行示例
    example_with_mock_model()
    example_with_precomputed_vectors()
    example_api_integration()
    
    print("\n" + "=" * 60)
    print("总结：")
    print("1. 最简单的RAG只需要 numpy + 余弦相似度")
    print("2. 数据量 < 1万条时完全够用")
    print("3. 核心API就3个：add_documents, search, save/load")
    print("4. 接入你的模型只需实现 encode() 方法")
    print("=" * 60)
