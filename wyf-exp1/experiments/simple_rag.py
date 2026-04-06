"""
最简单的RAG匹配实现 - 纯Python，无需向量数据库
适合文档数量 < 1万条的场景
"""
import numpy as np
from typing import List, Tuple

class SimpleRAG:
    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: 你的embedding模型，需要有 encode(texts) -> vectors 方法
        """
        self.model = embedding_model
        self.documents = []  # 存储原文
        self.vectors = []    # 存储向量
    
    def add_documents(self, texts: List[str]):
        """添加文档到知识库"""
        # 1. 编码文档
        vectors = self.model.encode(texts)  # shape: (n_docs, dim)
        
        # 2. 存储
        self.documents.extend(texts)
        self.vectors.extend(vectors)
        print(f"已添加 {len(texts)} 篇文档，当前共 {len(self.documents)} 篇")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        搜索最相似的文档
        
        Returns:
            [(document_text, similarity_score), ...]
        """
        if not self.documents:
            return []
        
        # 1. 编码查询
        query_vector = self.model.encode([query])[0]  # shape: (dim,)
        
        # 2. 计算余弦相似度
        doc_vectors = np.array(self.vectors)
        
        # 归一化
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        doc_norms = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-8)
        
        # 计算相似度
        similarities = np.dot(doc_norms, query_norm)  # shape: (n_docs,)
        
        # 3. 获取top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 假设你有这样的embedding模型
    class MockEmbeddingModel:
        """模拟你的embedding模型"""
        def encode(self, texts: List[str]) -> np.ndarray:
            """
            这里替换为你实际的embedding模型调用
            例如：
            - OpenAI: openai.Embedding.create()
            - HuggingFace: model.encode()
            - 本地API: requests.post()
            """
            # 模拟：生成随机向量（实际使用时替换为真实embedding）
            dim = 384  # embedding维度，根据你的模型调整
            return np.random.randn(len(texts), dim).astype(np.float32)
    
    # 1. 初始化
    model = MockEmbeddingModel()
    rag = SimpleRAG(model)
    
    # 2. 添加文档
    documents = [
        "RAG（检索增强生成）是一种结合检索和生成的技术",
        "向量数据库用于高效存储和检索高维向量",
        "余弦相似度是衡量向量相似性的常用指标",
        "Python是数据科学领域最流行的编程语言之一",
        "机器学习是人工智能的一个重要分支"
    ]
    rag.add_documents(documents)
    
    # 3. 搜索
    query = "什么是RAG技术？"
    results = rag.search(query, top_k=3)
    
    print(f"\n查询：{query}")
    print("\n最相似的文档：")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.4f}] {doc}")
