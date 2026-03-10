"""
基线分类器 - 使用完整的System Prompt直接分类
兼容现有的EvaluationPipeline
"""
import json
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structured_profile import StructuredProfile, ProfileSet


class BaselinePromptLoader:
    """加载完整的分类器Prompt作为基线"""
    
    def __init__(self):
        # 12个意图类别的映射
        self.intent_classes = [
            "选股类", "诊股类", "预测类", "知识库类", "新闻类",
            "通用类", "推荐类", "策略类", "指标查询类", "身份类",
            "分时图类", "K线图类"
        ]
    
    def load_manual_prompt(self, file_path: str = "data/baselines/manual_descriptions.json") -> str:
        """加载人工撰写的完整Prompt"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def load_intentgpt_prompt(self, file_path: str = "data/baselines/intentgpt_descriptions.json") -> str:
        """加载IntentGPT的完整Prompt"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def create_dummy_profile_set(self, prompt_text: str, name: str = "baseline") -> ProfileSet:
        """
        将完整Prompt转换为ProfileSet格式（兼容现有代码）
        
        策略：把整个prompt放入"通用智能体"的core_capability
        其他槽位留空或放占位符
        """
        # 压缩到200字以内（如果需要）
        if len(prompt_text) > 200:
            prompt_text = prompt_text[:200]
        
        # 硬切分（模拟非结构化）
        c = prompt_text[:80]
        b = prompt_text[80:140] if len(prompt_text) > 80 else "处理相关请求"
        r = prompt_text[140:200] if len(prompt_text) > 140 else "不处理其他请求"
        
        profile = StructuredProfile(
            agent_name=name,
            core_capability=c,
            boundary=b,
            rejection_scope=r
        )
        
        return ProfileSet([profile])
    
    def load_as_profile_set(self, file_path: str, name: str = "baseline") -> ProfileSet:
        """便捷方法：直接加载为ProfileSet"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.create_dummy_profile_set(content, name)


# 便捷函数
def load_manual_baseline(file_path: str = "data/baselines/manual_descriptions.json") -> ProfileSet:
    """加载人工Prompt基线"""
    loader = BaselinePromptLoader()
    return loader.load_as_profile_set(file_path, name="manual_baseline")


def load_intentgpt_baseline(file_path: str = "data/baselines/intentgpt_descriptions.json") -> ProfileSet:
    """加载IntentGPT Prompt基线"""
    loader = BaselinePromptLoader()
    return loader.load_as_profile_set(file_path, name="intentgpt_baseline")
