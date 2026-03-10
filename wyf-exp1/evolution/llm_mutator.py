"""
LLM定向变异算子
使用DeepSeek API进行智能体描述的变异
"""
import random
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
    MAX_C_LENGTH, MAX_B_LENGTH, MAX_R_LENGTH
)
from structured_profile import StructuredProfile, ProfileSet


class LLMMutator:
    """LLM定向变异器"""
    
    def __init__(self, api_key: str = DEEPSEEK_API_KEY, 
                 api_url: str = DEEPSEEK_API_URL,
                 model: str = DEEPSEEK_MODEL,
                 positive_examples_func=None):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.positive_examples_func = positive_examples_func  # 获取正例的函数
        
        # 检查API key
        if api_key == "your-deepseek-api-key-here":
            print("[Warning] DeepSeek API Key not configured. Mutations will be random.")
            self.use_llm = False
        else:
            self.use_llm = True
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=api_url
                )
                print(f"[LLMMutator] Initialized with model: {model}")
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenAI client: {e}")
                self.use_llm = False
    
    def mutate(self, profile_set: ProfileSet, generation: int,
               bad_cases: Optional[List[Dict]] = None) -> ProfileSet:
        """
        对整个ProfileSet进行变异
        
        策略：
        1. 随机选择一个智能体
        2. 随机选择一个槽位(C/B/R)
        3. 使用LLM对该槽位进行重写/缩写/添加否定
        """
        # 创建拷贝
        new_profile_set = profile_set.copy()
        
        # 随机选择要变异的智能体（50%概率变异每个智能体）
        for agent_name in new_profile_set.profiles.keys():
            if random.random() < 0.5:  # 50%概率变异每个智能体
                profile = new_profile_set.get_profile(agent_name)
                
                # 随机选择槽位
                slot = random.choice(['C', 'B', 'R'])
                slot_name_map = {'C': '核心能力', 'B': '处理边界', 'R': '拒绝范围'}
                slot_name = slot_name_map[slot]
                max_len_map = {'C': MAX_C_LENGTH, 'B': MAX_B_LENGTH, 'R': MAX_R_LENGTH}
                max_len = max_len_map[slot]
                
                # 获取当前内容
                current_text = profile.get_slot(slot)
                
                # 执行变异
                if self.use_llm:
                    new_text = self._llm_mutate_slot(
                        agent_name, slot_name, current_text, max_len, bad_cases
                    )
                else:
                    new_text = self._random_mutate_slot(current_text, max_len)
                
                # 应用变异
                new_profile = profile.mutate_slot(slot, new_text)
                new_profile_set.profiles[agent_name] = new_profile
        
        return new_profile_set
    
    def _llm_mutate_slot(self, agent_name: str, slot_name: str, 
                        current_text: str, max_len: int,
                        bad_cases: Optional[List[Dict]] = None) -> str:
        """使用LLM变异特定槽位"""
        
        # 根据槽位类型筛选不同类型的错例
        # C/B: 使用期望是本智能体的案例（本该处理但被错分）
        # R: 使用实际是本智能体的案例（不该处理但被错收）
        should_accept_cases = []  # 本该处理但被错分
        should_reject_cases = []  # 不该处理但被错收
        
        if bad_cases:
            for case in bad_cases:
                expected_agent = case.get('expected_agent', '')
                predicted_agent = case.get('predicted_agent', '')
                
                if expected_agent == agent_name:
                    # 本该是本智能体，但被分到了其他 → 应该强调接纳
                    should_accept_cases.append(case)
                elif predicted_agent == agent_name:
                    # 被分到了本智能体，但本该是其他 → 应该强调拒绝
                    should_reject_cases.append(case)
        
        # 根据槽位类型选择要展示的案例
        slot_code = 'C' if slot_name == '核心能力' else ('B' if slot_name == '处理边界' else 'R')
        
        if slot_code in ['C', 'B']:
            # 核心能力和处理边界：展示本该处理但被错分的案例
            selected_cases = should_accept_cases
            case_direction = "本该由本智能体处理，但被错误分到了其他智能体"
            case_instruction = "请强化本智能体处理这类问题的能力，并举例说明"
            # 如果没有错例，从数据集获取正例
            if not selected_cases and self.positive_examples_func:
                positive_queries = self.positive_examples_func(agent_name, n=3)
                selected_cases = [{'query': q, 'expected_agent': agent_name, 'predicted_agent': '其他'} 
                                 for q in positive_queries]
                case_direction = "以下是本智能体应该处理的问题示例"
                case_instruction = "请参考这些示例，描述本智能体的核心能力"
        else:  # slot_code == 'R'
            # 拒绝范围：展示不该处理但被错收的案例
            selected_cases = should_reject_cases
            case_direction = "被错误分到了本智能体，但本该由其他智能体处理"
            case_instruction = "请明确本智能体不处理这类问题，并举例说明"
            # 如果没有错例，从所有错例里选一些
            if not selected_cases and bad_cases:
                # 从其他错例中随机选3个作为反面示例
                import random
                other_cases = [c for c in bad_cases if c.get('predicted_agent') != agent_name]
                if other_cases:
                    selected_cases = random.sample(other_cases, min(3, len(other_cases)))
                    case_direction = "以下是其他智能体处理的问题（本智能体不处理）"
                    case_instruction = "请明确本智能体不处理这类问题"
        
        # 构建Prompt
        prompt = f"""你是一个智能体描述优化专家。

当前智能体：{agent_name}
需要优化的槽位：[{slot_name}]
当前内容：{current_text}

请对[{slot_name}]进行优化，要求：
1. 优化后的字数 ≤ {max_len}字，超出字数则减少描述性语言，保留例子
2. 突出该智能体的独特能力范围
3. **必须包含具体示例，例子越多越好，描述要少**
4. **拒绝范围只说"不处理"**
5. **边界范围、核心能力只说"只处理"**

诊股智能体拒绝范围举例：不处理概念辨析、新闻查询。例如：“河钢资源参与制造（新闻类）”、“潜伏吸筹意思（知识库类）”。
诊股智能体核心能力举例：只处理单只股票诊断。例如：“航天工程”诊股、“我的自选股表现”。
"""
        
        if selected_cases and len(selected_cases) > 0:
            prompt += f"\n【{case_direction}】\n"
            prompt += f"{case_instruction}：\n"
            for i, case in enumerate(selected_cases[:5]):  # 最多5个案例
                query = case.get('query', 'N/A')
                expected = case.get('expected_agent', 'N/A')
                predicted = case.get('predicted_agent', 'N/A')
                if expected == predicted:
                    prompt += f"正确输入示例:'{query}' :{expected}\n"
                else: 
                    prompt += f"错误输入示例:'{query}' :{expected}\n"

        
        prompt += f"\n请直接输出优化后的[{slot_name}]内容，不要解释，但必须包含具体示例，我们只新增内容，不要过于修改原文、要保留当前内容的例子。如果字数超过了就削减概述，保留例子："
        
        # 打印prompt用于调试
        print(f"\n[LLM Prompt] 发送给DeepSeek的Prompt:")
        print("-" * 60)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 60)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            
            # 打印LLM返回的完整答案
            print(f"\n[LLM Response] DeepSeek返回的优化结果:")
            print("=" * 60)
            print(result)
            print("=" * 60)
            
            # 清理结果（移除可能的标记）
            result = result.replace(f"[{slot_name}]", "").strip()
            
            # 检查字数
            if len(result) > max_len * 1.5:  # 允许一些余量
                print(f"[Warning] 结果过长 ({len(result)}字 > {max_len}字限制)，进行截断")
                # 截断
                result = result[:max_len] + "..." if len(result) > max_len else result
            
            print(f"[LLM Result] 最终采用: {result[:50]}...")
            
            return result
            
        except Exception as e:
            print(f"[Error] LLM mutation failed: {e}")
            return self._random_mutate_slot(current_text, max_len)
    
    def _random_mutate_slot(self, current_text: str, max_len: int) -> str:
        """随机变异（当LLM不可用时使用）"""
        strategies = ['abbreviate', 'add_negation', 'reorder']
        strategy = random.choice(strategies)
        
        if strategy == 'abbreviate':
            # 简单缩写：移除一些字符
            words = current_text.split('，')
            if len(words) > 2:
                return '，'.join(words[:2]) + '等。'
            return current_text
        
        elif strategy == 'add_negation':
            # 添加否定表述
            if '不处理' not in current_text and '不涉及' not in current_text:
                return current_text + '不涉及其他类型问题。'
            return current_text
        
        else:  # reorder
            # 简单重排
            words = current_text.split('，')
            if len(words) > 2:
                random.shuffle(words)
                return '，'.join(words)
            return current_text
