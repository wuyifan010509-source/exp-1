"""
意图分类器 - 调用GPU模型进行分类
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import requests

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BACKBONE_MODEL, BACKBONE_API_URL, INTENT_TO_AGENT
from structured_profile import ProfileSet
import os


@dataclass
class ClassifyResult:
    """分类结果"""
    predicted_agent: str
    confidence_scores: Dict[str, float]  # agent_name -> probability
    raw_response: str
    
    def get_margin(self) -> float:
        """计算Top1-Top2概率差"""
        sorted_scores = sorted(self.confidence_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            return sorted_scores[0] - sorted_scores[1]
        return 1.0  # 只有一个类别时margin为1
    
    def is_correct(self, expected_intent: str) -> bool:
        """判断是否正确分类"""
        expected_agent = INTENT_TO_AGENT.get(expected_intent)
        return self.predicted_agent == expected_agent


class IntentClassifier(ABC):
    """意图分类器抽象基类"""
    
    @abstractmethod
    def classify(self, query: str, profiles: ProfileSet) -> ClassifyResult:
        """单条分类"""
        pass
    
    @abstractmethod
    def classify_batch(self, queries: List[str], profiles: ProfileSet) -> List[ClassifyResult]:
        """批量分类"""
        pass


class QwenClassifier(IntentClassifier):
    """调用本地GPU部署的qwen模型（带缓存优化）"""
    
    def __init__(self, api_url: str = BACKBONE_API_URL, model: str = BACKBONE_MODEL):
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy"  # vLLM不需要真实key
        }
        # 缓存机制：缓存ProfileSet的prompt转换结果
        self._profile_cache = {}  # {profile_set_id: (prompt, timestamp)}
        self._cache_hits = 0
        self._cache_misses = 0
        print(f"[QwenClassifier] Initialized with URL: {api_url}")
    
    def _get_cached_prompt(self, profiles: ProfileSet) -> str:
        """获取缓存的prompt，如果不存在则构建并缓存"""
        # 使用ProfileSet的id作为缓存键
        cache_key = id(profiles)
        
        if cache_key in self._profile_cache:
            self._cache_hits += 1
            return self._profile_cache[cache_key][0]
        
        # 缓存未命中，构建prompt
        self._cache_misses += 1
        prompt = self._build_agent_descriptions(profiles)
        self._profile_cache[cache_key] = (prompt, time.time())
        
        # 限制缓存大小（最多保留10个不同的ProfileSet）
        if len(self._profile_cache) > 10:
            # 移除最旧的缓存
            oldest_key = min(self._profile_cache.keys(), key=lambda k: self._profile_cache[k][1])
            del self._profile_cache[oldest_key]
        
        return prompt
    
    def _build_agent_descriptions(self, profiles: ProfileSet) -> str:
        """构建智能体描述（用于缓存）"""
        agent_descriptions = []
        for agent_name, profile in profiles.profiles.items():
            agent_descriptions.append(f"{agent_name}: {profile.to_prompt()}")
        return "\n".join(agent_descriptions)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_profiles": len(self._profile_cache)
        }
    
    def _build_prompt(self, query: str, profiles: ProfileSet) -> str:
        """构建分类Prompt（使用缓存）"""
        system_prompt = """你是一个意图分类路由器。根据以下智能体描述，判断用户问题应该由哪个智能体处理。
仅输出智能体名称，不要解释。

可用智能体及其描述："""
        
        # 使用缓存的描述（避免重复构建）
        agent_descriptions = self._get_cached_prompt(profiles)
        
        user_prompt = f"""
{agent_descriptions}

用户问题：{query}

请输出最匹配的智能体名称（只需输出名称）："""
        
        return system_prompt + user_prompt
    
    def _build_json_prompt(self, query: str, profiles: ProfileSet) -> str:
        """构建输出JSON格式的Prompt（使用缓存）"""
        system_prompt = """你是一个意图分类路由器。根据以下智能体描述，判断用户问题应该由哪个智能体处理。
请以JSON格式输出，包含agent和confidence字段。

可用智能体及其描述："""
        
        # 使用缓存的描述（避免重复构建）
        agent_descriptions = self._get_cached_prompt(profiles)
        
        user_prompt = f"""
{agent_descriptions}

用户问题：{query}

请以JSON格式输出：{{"agent": "智能体名称", "confidence": {{"智能体1": 0.8, "智能体2": 0.1, ...}}}}"""
        
        return system_prompt + user_prompt
    
    def classify(self, query: str, profiles: ProfileSet, use_json: bool = False) -> ClassifyResult:
        """单条分类"""
        if use_json:
            prompt = self._build_json_prompt(query, profiles)
        else:
            prompt = self._build_prompt(query, profiles)
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"].strip()
            
            # 解析JSON或纯文本
            if use_json:
                try:
                    parsed = json.loads(raw_response)
                    predicted_agent = parsed.get("agent", "")
                    confidence_scores = parsed.get("confidence", {})
                    # 归一化置信度
                    total = sum(confidence_scores.values())
                    if total > 0:
                        confidence_scores = {k: v/total for k, v in confidence_scores.items()}
                    return ClassifyResult(predicted_agent, confidence_scores, raw_response)
                except json.JSONDecodeError:
                    # JSON解析失败，使用文本匹配
                    pass
            
            # 文本匹配方式
            predicted_agent = self._extract_agent_name(raw_response, profiles)
            confidence_scores = {name: 0.0 for name in profiles.profiles.keys()}
            if predicted_agent:
                confidence_scores[predicted_agent] = 1.0
            
            return ClassifyResult(predicted_agent, confidence_scores, raw_response)
            
        except Exception as e:
            print(f"[Error] Classification failed: {e}")
            # 返回空结果
            return ClassifyResult("", {}, str(e))
    
    def classify_batch(self, queries: List[str], profiles: ProfileSet, batch_size: int = 10) -> List[ClassifyResult]:
        """批量分类（串行实现，vLLM会自动batch）"""
        results = []
        start_time = time.time()
        for i, query in enumerate(queries):
            if (i + 1) % 10 == 0:
                print(f"[Batch] Processed {i+1}/{len(queries)} queries")
            result = self.classify(query, profiles)
            results.append(result)
            # 无延迟，vLLM自动处理并发
        
        # 打印缓存统计
        elapsed = time.time() - start_time
        stats = self.get_cache_stats()
        print(f"[Cache] Hits: {stats['hits']}, Misses: {stats['misses']}, "
              f"Hit Rate: {stats['hit_rate']:.1%}, Time: {elapsed:.1f}s")
        
        return results
    
    def _extract_agent_name(self, text: str, profiles: ProfileSet) -> str:
        """从响应中提取智能体名称"""
        text = text.strip()
        # 尝试直接匹配
        for agent_name in profiles.profiles.keys():
            if agent_name in text:
                return agent_name
        # 尝试匹配意图类别
        from config import INTENT_TO_AGENT
        for intent, agent in INTENT_TO_AGENT.items():
            if intent in text or intent.replace("类", "") in text:
                return agent
        return ""


class MockClassifier(IntentClassifier):
    """Mock分类器，用于测试"""
    
    def __init__(self, seed: int = 42):
        import random
        self.random = random.Random(seed)
        print("[MockClassifier] Initialized (for testing)")
    
    def classify(self, query: str, profiles: ProfileSet) -> ClassifyResult:
        """随机分类"""
        agent_names = list(profiles.profiles.keys())
        # 随机选择，但给第一个较高的概率（模拟某种模式）
        probs = [0.4] + [0.6 / (len(agent_names) - 1)] * (len(agent_names) - 1)
        predicted = self.random.choices(agent_names, weights=probs, k=1)[0]
        
        confidence_scores = {}
        for name in agent_names:
            if name == predicted:
                confidence_scores[name] = 0.5 + self.random.random() * 0.4
            else:
                confidence_scores[name] = self.random.random() * 0.3
        
        # 归一化
        total = sum(confidence_scores.values())
        confidence_scores = {k: v/total for k, v in confidence_scores.items()}
        
        return ClassifyResult(predicted, confidence_scores, f"Mock: {predicted}")
    
    def classify_batch(self, queries: List[str], profiles: ProfileSet) -> List[ClassifyResult]:
        return [self.classify(q, profiles) for q in queries]
