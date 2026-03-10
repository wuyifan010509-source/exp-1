"""
结构化描述 P={C,B,R} 数据建模
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import re


@dataclass
class StructuredProfile:
    """单个智能体的结构化描述"""
    agent_name: str
    core_capability: str      # C: 核心能力
    boundary: str             # B: 处理边界
    rejection_scope: str      # R: 拒绝范围
    
    def to_prompt(self) -> str:
        """转换为Prompt格式"""
        return f"[核心能力]{self.core_capability}[处理边界]{self.boundary}[拒绝范围]{self.rejection_scope}"
    
    def to_display_format(self) -> str:
        """转换为显示格式"""
        return f"""[核心能力] {self.core_capability}
[处理边界] {self.boundary}
[拒绝范围] {self.rejection_scope}"""
    
    def length(self) -> int:
        """计算中文字符数"""
        # 移除标记符号，只计算内容长度
        content = self.core_capability + self.boundary + self.rejection_scope
        # 计算中文字符和英文单词
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        english_words = len(re.findall(r'[a-zA-Z]+', content))
        return chinese_chars + english_words
    
    def is_valid(self, max_len: int = 200) -> bool:
        """检查字数约束"""
        return self.length() <= max_len
    
    def mutate_slot(self, slot: str, new_text: str) -> 'StructuredProfile':
        """变异特定槽位，返回新个体"""
        if slot == 'C':
            return StructuredProfile(
                agent_name=self.agent_name,
                core_capability=new_text,
                boundary=self.boundary,
                rejection_scope=self.rejection_scope
            )
        elif slot == 'B':
            return StructuredProfile(
                agent_name=self.agent_name,
                core_capability=self.core_capability,
                boundary=new_text,
                rejection_scope=self.rejection_scope
            )
        elif slot == 'R':
            return StructuredProfile(
                agent_name=self.agent_name,
                core_capability=self.core_capability,
                boundary=self.boundary,
                rejection_scope=new_text
            )
        else:
            raise ValueError(f"Unknown slot: {slot}")
    
    def get_slot(self, slot: str) -> str:
        """获取特定槽位内容"""
        if slot == 'C':
            return self.core_capability
        elif slot == 'B':
            return self.boundary
        elif slot == 'R':
            return self.rejection_scope
        else:
            raise ValueError(f"Unknown slot: {slot}")
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuredProfile':
        """从字典反序列化"""
        return cls(**data)


class ProfileSet:
    """管理12个智能体的Profile集合"""
    
    def __init__(self, profiles: List[StructuredProfile]):
        self.profiles = {p.agent_name: p for p in profiles}
        self._validate()
    
    def _validate(self):
        """验证所有意图类别都有对应的Profile"""
        from config import INTENT_TO_AGENT
        required_agents = set(INTENT_TO_AGENT.values())
        existing_agents = set(self.profiles.keys())
        if not required_agents.issubset(existing_agents):
            missing = required_agents - existing_agents
            raise ValueError(f"Missing profiles for agents: {missing}")
    
    def get_profile(self, agent_name: str) -> StructuredProfile:
        """获取特定智能体的Profile"""
        return self.profiles[agent_name]
    
    def get_profile_by_intent(self, intent: str) -> StructuredProfile:
        """通过意图类别获取Profile"""
        from config import INTENT_TO_AGENT
        agent_name = INTENT_TO_AGENT.get(intent)
        if not agent_name:
            raise ValueError(f"Unknown intent: {intent}")
        return self.profiles[agent_name]
    
    def to_prompt_list(self) -> List[str]:
        """转换为用于分类的Prompt列表"""
        return [f"{name}: {p.to_prompt()}" for name, p in self.profiles.items()]
    
    def to_dict(self) -> Dict:
        """序列化"""
        return {
            name: profile.to_dict() 
            for name, profile in self.profiles.items()
        }
    
    def total_length(self) -> int:
        """计算所有Profile的总字数"""
        return sum(p.length() for p in self.profiles.values())
    
    def average_length(self) -> float:
        """计算平均字数"""
        return self.total_length() / len(self.profiles)
    
    def is_valid(self, max_len_per_agent: int = 200) -> bool:
        """检查所有Profile是否满足字数约束"""
        return all(p.is_valid(max_len_per_agent) for p in self.profiles.values())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProfileSet':
        """从字典反序列化"""
        profiles = [StructuredProfile.from_dict(p) for p in data.values()]
        return cls(profiles)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ProfileSet':
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_json(self, filepath: str):
        """保存到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def copy(self) -> 'ProfileSet':
        """创建深拷贝"""
        return ProfileSet.from_dict(self.to_dict())
