"""
本地Embedding API的RAG实现
适配你的EmbeddingClient风格，但改为同步调用
"""
import os
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
from openai import OpenAI  # 同步客户端

os.environ["EMBEDDING_BASE_URL"] = "http://127.0.0.1:3002/v1"
os.environ["EMBEDDING_MODEL"] = "/Data1/wyf_workspace/embedding/bge-m3"
class LocalEmbeddingModel:
    """
    本地Embedding模型适配器
    
    使用方式：
    model = LocalEmbeddingModel(
        base_url="http://your-embedding-server:8080/v1",
        api_key="your-api-key",
        model_name="your-embedding-model"
    )
    """
    
    def __init__(
        self, 
        base_url: str = None,
        api_key: str = None,
        model_name: str = None
    ):
        """
        Args:
            base_url: Embedding服务地址，如 "http://172.17.160.46:8080/v1"
            api_key: API密钥
            model_name: 模型名称
        """
        # 优先使用传入的参数，否则从环境变量读取
        self.base_url = base_url or os.environ.get("EMBEDDING_BASE_URL")
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY", "sk-placeholder")
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL")
        
        if not self.base_url:
            raise ValueError("请提供 base_url 或设置 EMBEDDING_BASE_URL 环境变量")
        if not self.model_name:
            raise ValueError("请提供 model_name 或设置 EMBEDDING_MODEL 环境变量")
        
        # 创建同步客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60
        )
        
        print(f"✓ Embedding模型已初始化: {self.model_name}")
        print(f"  服务地址: {self.base_url}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy数组，shape: (len(texts), dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float"
            )
            
            # 提取向量
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            print(f"调用Embedding API失败: {e}")
            raise


class SimpleRAG:
    """
    极简RAG检索器 - 支持metadata（key-value匹配）
    
    使用场景：
    - key: 用于匹配的问题/关键词（aaa）
    - value: 匹配成功后返回的内容（bbb）
    """
    
    def __init__(self, embedding_model=None, dim: int = 768, key_field: str = "query", value_field: str = "answer"):
        """
        Args:
            embedding_model: Embedding模型
            dim: 向量维度
            key_field: metadata中用于匹配的字段名（默认"query"）
            value_field: metadata中返回的字段名（默认"answer"）
        """
        self.model = embedding_model
        self.dim = dim
        self.key_field = key_field
        self.value_field = value_field
        self.metadata_list: List[Dict] = []  # 存储完整的metadata
        self.vectors: np.ndarray = None
    
    def add_documents(self, metadata_list: List[Dict], vectors: Optional[np.ndarray] = None):
        """
        添加带metadata的文档
        
        Args:
            metadata_list: metadata列表，每个元素是dict，必须包含key_field指定的字段
                例如: [{"query": "问题1", "answer": "答案1", "category": "类别1"}, ...]
            vectors: 可选，预计算的向量（如果不提供，用embedding_model编码key_field字段）
        """
        if not metadata_list:
            return
        
        # 获取用于匹配的文本（key_field字段）
        key_texts = [item[self.key_field] for item in metadata_list]
        
        # 获取向量
        if vectors is not None:
            new_vectors = np.array(vectors, dtype=np.float32)
        elif self.model is not None:
            print(f"正在编码 {len(metadata_list)} 条记录...")
            new_vectors = self.model.encode(key_texts)
        else:
            raise ValueError("请提供 embedding_model 或 vectors")
        
        # 归一化
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_vectors = new_vectors / (norms + 1e-8)
        
        # 添加数据
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        
        self.metadata_list.extend(metadata_list)
        print(f"✓ 已添加 {len(metadata_list)} 条记录，当前共 {len(self.metadata_list)} 条")
    
    def search(self, query: str, top_k: int = 3, return_metadata: bool = False):
        """
        搜索最相似的记录
        
        Args:
            query: 查询文本
            top_k: 返回最相似的前k个
            return_metadata: 是否返回完整metadata，False则只返回value_field字段
            
        Returns:
            如果return_metadata=False（默认）:
                [(value, similarity_score), ...]
            如果return_metadata=True:
                [(metadata_dict, similarity_score), ...]
        """
        if len(self.metadata_list) == 0:
            return []
        
        # 编码查询
        q_vector = self.model.encode([query])[0]
        q_vector = q_vector / (np.linalg.norm(q_vector) + 1e-8)
        
        # 计算余弦相似度
        similarities = np.dot(self.vectors, q_vector)
        
        # 获取top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 组装结果
        results = []
        for idx in top_indices:
            metadata = self.metadata_list[idx]
            score = float(similarities[idx])
            if return_metadata:
                results.append((metadata, score))
            else:
                results.append((metadata.get(self.value_field, ""), score))
        
        return results
    
    def save(self, filepath: str):
        """保存知识库"""
        data = {
            'metadata_list': self.metadata_list,
            'vectors': self.vectors.tolist() if self.vectors is not None else [],
            'dim': self.dim,
            'key_field': self.key_field,
            'value_field': self.value_field
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"✓ 知识库已保存: {filepath}")
    
    def load(self, filepath: str):
        """加载知识库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.metadata_list = data['metadata_list']
        if data['vectors']:
            self.vectors = np.array(data['vectors'], dtype=np.float32)
        self.dim = data.get('dim', self.dim)
        self.key_field = data.get('key_field', self.key_field)
        self.value_field = data.get('value_field', self.value_field)
        print(f"✓ 已加载 {len(self.metadata_list)} 条记录")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    
    print("=" * 60)
    print("本地Embedding API + RAG (含Metadata) 示例")
    print("=" * 60)
    
    # 设置环境变量（用于测试）

    
    # 初始化模型
    model = LocalEmbeddingModel()
    
    # 初始化RAG，指定key_field（用于匹配）和value_field（返回给用户）
    # 这里用 "query" 作为匹配字段，"answer" 作为返回字段
    rag = SimpleRAG(
        embedding_model=model, 
        dim=1024,  # bge-m3是1024维
        key_field="query",    # aaa - 用于匹配的字段
        value_field="answer"  # bbb - 返回的字段
    )
    
    # 添加带metadata的文档
    # 用 "query" 匹配，返回 "answer"
    metadata_list = [
        {
            "query": "沪深300股票有哪些？",           # aaa - 匹配用
            "answer": "用户只输入选股条件，是在请求筛选该条件的股票",  # bbb - 返回用
        },
        {
            "query": "红利轮动策略",
            "answer": "用户提及“xx策略”，要转为策略类。",
        },
        {
            "query": "在哪里看北向资金的实时动向？",
            "answer": "用户提及“擒龙平台”、“黄金坑”，是在询问软件操作知识",
        },
        {
            "query": "景旺电子交易策略",
            "answer": "用户提及“xx策略”，要转为策略类。",
        },
    ]
    
    rag.add_documents(metadata_list)
    
    # 搜索测试 - 用户输入ccc，匹配aaa，返回bbb
    queries = [
        "成分股有哪些",           # ccc - 应该匹配 "沪深300股票有哪些？"
        "景旺电子怎么操作",      # ccc - 应该匹配 "景旺电子交易策略"
        "北向资金哪里看",        # ccc - 应该匹配 "在哪里看北向资金的实时动向？"
    ]
    
    print("\n" + "=" * 60)
    print("搜索结果（只返回answer字段）：")
    print("=" * 60)
    
    for query in queries:
        print(f"\n🔍 用户查询 (ccc): {query}")
        # 默认返回 (answer, score)
        results = rag.search(query, top_k=2)
        for i, (answer, score) in enumerate(results, 1):
            print(f"  {i}. [相似度: {score:.4f}]")
            print(f"     答案 (bbb): {answer}")
    
    # 如果需要完整metadata，设置 return_metadata=True
    print("\n" + "=" * 60)
    print("搜索结果（返回完整metadata）：")
    print("=" * 60)
    
    query = "景旺电子怎么操作"
    print(f"\n🔍 查询: {query}")
    results = rag.search(query, top_k=2, return_metadata=True)
    for i, (metadata, score) in enumerate(results, 1):
        print(f"\n  {i}. [相似度: {score:.4f}]")
        print(f"     匹配的问题 (aaa): {metadata['query']}")
        print(f"     返回的答案 (bbb): {metadata['answer']}")
        print(f"     分类: {metadata['category']}")
    
    # 保存知识库
    rag.save("my_rag_kb.json")
