"""
使用FAISS的RAG匹配实现 - 高性能向量检索
安装：pip install faiss-cpu  （或 faiss-gpu 如果你有NVIDIA显卡）
"""
import numpy as np
import faiss
from typing import List, Tuple

class FaissRAG:
    def __init__(self, embedding_model, dim: int = 384):
        """
        Args:
            embedding_model: embedding模型
            dim: 向量维度（根据你的模型调整，如384、768、1024等）
        """
        self.model = embedding_model
        self.dim = dim
        self.documents = []
        
        # 创建FAISS索引（使用内积，相当于余弦相似度在归一化后）
        self.index = faiss.IndexFlatIP(dim)  # IP = Inner Product
        
        # 如果你有GPU，可以使用：
        # res = faiss.StandardGpuResources()
        # self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    
    def add_documents(self, texts: List[str]):
        """添加文档"""
        if not texts:
            return
            
        # 1. 编码
        vectors = self.model.encode(texts)
        vectors = np.array(vectors).astype('float32')
        
        # 2. 归一化（为了使用内积近似余弦相似度）
        faiss.normalize_L2(vectors)
        
        # 3. 添加到索引
        self.index.add(vectors)
        self.documents.extend(texts)
        
        print(f"已添加 {len(texts)} 篇文档，当前共 {len(self.documents)} 篇")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """搜索"""
        if not self.documents:
            return []
        
        # 1. 编码查询
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        # 2. 归一化
        faiss.normalize_L2(query_vector)
        
        # 3. 搜索
        scores, indices = self.index.search(query_vector, top_k)
        
        # 4. 组装结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 表示没有足够结果
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """保存索引和文档"""
        faiss.write_index(self.index, f"{path}.index")
        import json
        with open(f"{path}.docs.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False)
        print(f"已保存到 {path}")
    
    def load(self, path: str):
        """加载索引和文档"""
        self.index = faiss.read_index(f"{path}.index")
        import json
        with open(f"{path}.docs.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"已加载 {len(self.documents)} 篇文档")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 模拟你的embedding模型
    class MockEmbeddingModel:
        def __init__(self, dim=384):
            self.dim = dim
        
        def encode(self, texts: List[str]) -> np.ndarray:
            """替换为你的实际embedding模型调用"""
            return np.random.randn(len(texts), self.dim).astype(np.float32)
    
    # 1. 初始化
    model = MockEmbeddingModel(dim=384)
    rag = FaissRAG(model, dim=384)
    
    # 2. 添加文档
    documents = [
        "RAG（检索增强生成）结合了检索系统和生成模型",
        "FAISS是Facebook开发的向量相似度搜索库",
        "余弦相似度适合衡量文本语义相似性",
        "向量检索是RAG系统的核心组件"
    ]
    rag.add_documents(documents)
    
    # 3. 搜索
    query = "RAG系统如何工作？"
    results = rag.search(query, top_k=2)
    
    print(f"\n查询：{query}")
    print("\n最相似的文档：")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [相似度: {score:.4f}] {doc}")
    
    # 4. 保存和加载
    # rag.save("my_knowledge_base")
    # rag.load("my_knowledge_base")
