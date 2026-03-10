"""
随机压缩基线方法
模拟"人工试图压缩描述但缺乏结构意识"的场景
"""
import json
import random
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structured_profile import StructuredProfile, ProfileSet


class RandomCompression:
    """随机压缩器 - 多种压缩策略"""
    
    def __init__(self, max_len: int = 200):
        self.max_len = max_len
    
    def hard_truncate(self, text: str) -> str:
        """
        方案A：硬截断 - 直接切前200字
        模拟：人工写长描述后被迫硬性截断
        """
        if len(text) <= self.max_len:
            return text
        return text[:self.max_len]
    
    def random_truncate(self, text: str) -> str:
        """
        方案B：随机截断 - 从随机位置取200字
        模拟：随机选取描述的一部分
        """
        if len(text) <= self.max_len:
            return text
        
        start = random.randint(0, len(text) - self.max_len)
        return text[start:start + self.max_len]
    
    def sentence_sample(self, text: str) -> str:
        """
        方案C：随机句子采样 - 随机选取句子直到接近200字
        模拟：挑选几个完整句子，不考虑连贯性
        """
        # 按常见标点分句
        import re
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return self.hard_truncate(text)
        
        # 随机打乱句子顺序
        random.shuffle(sentences)
        
        result = ""
        for s in sentences:
            if len(result) + len(s) + 1 <= self.max_len:
                result += s + "。"
            else:
                break
        
        return result if result else self.hard_truncate(text)
    
    def uniform_sample(self, text: str) -> str:
        """
        方案D：均匀采样 - 等间隔采样字符
        模拟：均匀抽取关键词
        """
        if len(text) <= self.max_len:
            return text
        
        step = len(text) / self.max_len
        result = ""
        for i in range(self.max_len):
            idx = int(i * step)
            result += text[idx]
        return result
    
    def compress(self, text: str, method: str = "hard_truncate") -> str:
        """
        压缩文本
        
        Args:
            text: 原始长文本
            method: 压缩方法 ['hard_truncate', 'random_truncate', 'sentence_sample', 'uniform_sample']
        
        Returns:
            压缩后的文本（≤200字）
        """
        method_func = getattr(self, method, self.hard_truncate)
        return method_func(text)


def text_to_profile(agent_name: str, text: str) -> StructuredProfile:
    """
    将非结构化文本转换为StructuredProfile（保持文本原貌）
    
    为了兼容ProfileSet，但保持非结构化特性：
    - 将所有文本放入core_capability
    - boundary和rejection_scope留空或使用通用占位符
    
    Args:
        agent_name: 智能体名称
        text: 描述文本（已压缩到≤200字）
    
    Returns:
        StructuredProfile对象
    """
    # 如果文本较短，直接放入core_capability
    if len(text) <= 80:
        return StructuredProfile(
            agent_name=agent_name,
            core_capability=text,
            boundary="处理相关请求",
            rejection_scope="不处理其他请求"
        )
    
    # 如果文本较长，简单切分（硬分割，不考虑语义）
    # 这种硬切分正是基线的特点：无结构意识
    c = text[:80]  # 前80字
    remaining = text[80:]
    
    if len(remaining) <= 60:
        b = remaining
        r = "不处理其他请求"
    else:
        b = remaining[:60]
        r = remaining[60:120] if len(remaining) > 120 else remaining[60:]
    
    return StructuredProfile(
        agent_name=agent_name,
        core_capability=c,
        boundary=b,
        rejection_scope=r
    )


def load_text_baseline(
    text_file: str,
    compress_method: str = "",
    max_len: int = 200
) -> ProfileSet:
    """
    加载文本描述基线（非结构化）
    
    Args:
        text_file: JSON文件路径，格式为 {"agent_name": "description text", ...}
        compress_method: 压缩方法，None表示不压缩直接读取
                        可选: 'hard_truncate', 'random_truncate', 'sentence_sample', 'uniform_sample'
        max_len: 最大字数限制
    
    Returns:
        ProfileSet对象
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    compressor = RandomCompression(max_len=max_len)
    
    profiles = []
    for agent_name, description in data.items():
        # 跳过注释字段
        if agent_name.startswith('_'):
            continue
        
        # 压缩（如果需要）
        if compress_method:
            text = compressor.compress(description, compress_method)
        else:
            text = description[:max_len]  # 简单截断到max_len
        
        # 转换为Profile（非结构化方式）
        profile = text_to_profile(agent_name, text)
        profiles.append(profile)
    
    return ProfileSet(profiles)


# 便捷函数，用于不同基线场景
def load_manual_baseline(
    manual_file: str = "data/baselines/manual_descriptions.json",
    compress_method: str = "hard_truncate"
) -> ProfileSet:
    """
    加载人工描述基线
    
    Args:
        manual_file: 人工描述文件路径
        compress_method: 压缩方法，默认硬截断
    
    Returns:
        人工描述基线的ProfileSet
    """
    return load_text_baseline(manual_file, compress_method)


def load_intentgpt_baseline(
    intentgpt_file: str = "data/baselines/intentgpt_descriptions.json",
    compress_method: str = "hard_truncate"
) -> ProfileSet:
    """
    加载IntentGPT基线（非结构化文本）
    
    Args:
        intentgpt_file: IntentGPT描述文件路径
        compress_method: 压缩方法，默认硬截断
    
    Returns:
        IntentGPT基线的ProfileSet
    """
    return load_text_baseline(intentgpt_file, compress_method)


def load_unstructured_baseline(
    source_file: str,
    method: str = "hard_truncate"
) -> ProfileSet:
    """
    加载非结构化基线（兼容旧接口）
    
    等同于load_text_baseline，用于消融实验中"无C/B/R结构"的对照组
    """
    return load_text_baseline(source_file, method)


# 保持向后兼容的别名
load_manual = load_manual_baseline
load_intentgpt = load_intentgpt_baseline
load_compressed = load_unstructured_baseline
