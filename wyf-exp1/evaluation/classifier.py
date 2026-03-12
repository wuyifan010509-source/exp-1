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
    
    def classify(self, query: str, profiles: ProfileSet, use_json: bool = False, use_logprobs: bool = True) -> ClassifyResult:
        """单条分类
        
        Args:
            query: 查询文本
            profiles: 智能体画像
            use_json: 是否使用JSON格式输出（备用方案）
            use_logprobs: 是否使用API的logprobs获取真实概率（推荐）
        """
        prompt = self._build_prompt(query, profiles)
        
        try:
            # 构建请求参数
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 50,  # 只需要输出agent名称，不需要太多token
            }
            
            # 如果使用logprobs，添加相关参数
            if use_logprobs:
                request_data["logprobs"] = True
                request_data["top_logprobs"] = 20  # 获取Top 20个token的概率
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # 解析响应
            choice = result["choices"][0]
            raw_response = choice["message"]["content"].strip()
            
            # 如果使用logprobs，从token概率计算agent概率
            if use_logprobs and "logprobs" in choice:
                confidence_scores = self._extract_probs_from_logprobs(
                    choice["logprobs"], raw_response, profiles
                )
                predicted_agent = max(confidence_scores.items(), key=lambda x: x[1])[0]
                return ClassifyResult(predicted_agent, confidence_scores, raw_response)
            
            # 备用：JSON格式
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
                    pass
            
            # 最后的备用：文本匹配
            predicted_agent = self._extract_agent_name(raw_response, profiles)
            confidence_scores = {name: 0.0 for name in profiles.profiles.keys()}
            if predicted_agent:
                confidence_scores[predicted_agent] = 1.0
            
            return ClassifyResult(predicted_agent, confidence_scores, raw_response)
            
        except Exception as e:
            print(f"[Error] Classification failed: {e}")
            import traceback
            traceback.print_exc()
            # 返回空结果
            return ClassifyResult("", {}, str(e))
    
    def _extract_probs_from_logprobs(self, logprobs_data: dict, raw_response: str, profiles: ProfileSet) -> Dict[str, float]:
        """从API返回的logprobs中提取agent概率（改进版）
        
        策略：
        1. 从raw_response确定实际输出的agent
        2. 从第一个token的top_logprobs获取各agent开头的概率
        3. 基于这些信息估算各agent的整体概率
        """
        import math
        
        agent_names = list(profiles.profiles.keys())
        agent_scores = {name: 0.0 for name in agent_names}
        
        try:
            content_logprobs = logprobs_data.get("content", [])
            
            if not content_logprobs:
                print("[Warning] No logprobs content found")
                # 使用文本匹配方式
                predicted = self._extract_agent_name(raw_response, profiles)
                if predicted:
                    agent_scores[predicted] = 1.0
                return agent_scores
            
            # 第一步：从raw_response确定实际输出的agent
            predicted_agent = self._extract_agent_name(raw_response, profiles)
            
            # 第二步：从第一个token的top_logprobs获取概率分布
            first_token = content_logprobs[0]
            top_logprobs = first_token.get("top_logprobs", [])
            
            # 构建token到概率的映射
            token_probs = {}
            for candidate in top_logprobs:
                token = candidate.get("token", "").strip()
                logprob = candidate.get("logprob", -float('inf'))
                prob = math.exp(logprob)
                token_probs[token] = prob
            
            # 调试打印（通过环境变量控制，默认不打印）
            if os.environ.get('DEBUG_LOGPROBS'):
                print(f"[Logprobs] First token candidates: {[(t, f'{p:.4f}') for t, p in list(token_probs.items())[:5]]}")
            
            # 第三步：将token概率映射到agent概率
            # 策略：对于每个agent，检查其名称的第一个词是否在token_probs中
            for agent_name in agent_names:
                agent_short = agent_name.replace("智能体", "").strip()
                
                # 直接匹配完整名称
                if agent_name in token_probs:
                    agent_scores[agent_name] = token_probs[agent_name]
                # 匹配简称（如"选股"）
                elif agent_short in token_probs:
                    agent_scores[agent_name] = token_probs[agent_short]
                # 检查token是否包含在agent名称中
                else:
                    for token, prob in token_probs.items():
                        if token in agent_name or token in agent_short:
                            agent_scores[agent_name] = max(agent_scores[agent_name], prob)
            
            # 归一化
            total = sum(agent_scores.values())
            if total > 0:
                agent_scores = {k: v/total for k, v in agent_scores.items()}
            else:
                # 如果没有匹配到，给预测agent 1.0
                if predicted_agent:
                    agent_scores[predicted_agent] = 1.0
            
            return agent_scores
            
        except Exception as e:
            print(f"[Error] Failed to extract probs from logprobs: {e}")
            import traceback
            traceback.print_exc()
            # 使用文本匹配方式作为fallback
            predicted = self._extract_agent_name(raw_response, profiles)
            agent_scores = {name: 0.0 for name in agent_names}
            if predicted:
                agent_scores[predicted] = 1.0
            return agent_scores
    
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
