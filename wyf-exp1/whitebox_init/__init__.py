"""
白盒初始化模块
根据tools描述生成初始的结构化描述
"""
import json
import random
import pandas as pd
from typing import List, Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    AGENTS_PATH, INTENT_TO_AGENT, 
    MAX_C_LENGTH, MAX_B_LENGTH, MAX_R_LENGTH,
    POPULATION_SIZE, DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
    HISTORICAL_LOGS_PATH
)
from structured_profile import StructuredProfile, ProfileSet


class WhiteBoxInitializer:
    """白盒初始化器"""
    
    def __init__(self, tools_path: str = AGENTS_PATH, examples_path: str = HISTORICAL_LOGS_PATH):
        self.tools_path = tools_path
        self.examples_path = examples_path
        self.agents_data = self._load_tools()
        self.examples_by_intent = self._load_examples_by_intent()
        
        # 初始化DeepSeek客户端
        if DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_API_URL
                )
                self.use_llm = True
                print("[WhiteBoxInit] Using DeepSeek for initialization")
            except:
                self.use_llm = False
                print("[WhiteBoxInit] Using rule-based initialization")
        else:
            self.use_llm = False
            print("[WhiteBoxInit] Using rule-based initialization (No API key)")
    
    def _load_tools(self) -> Dict:
        """加载tools描述"""
        try:
            with open(self.tools_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[Error] Tools file not found: {self.tools_path}")
            # 返回一个空模板
            return {"agents": []}
    
    def _load_examples_by_intent(self) -> Dict[str, List[str]]:
        """从HISTORICAL_LOGS.csv加载每个意图的代表性例子"""
        try:
            df = pd.read_csv(self.examples_path)
            examples = {}
            
            # 对每个意图，收集所有查询
            for intent in df['预期意图'].unique():
                if intent == 'oos':  # 跳过oos
                    continue
                intent_queries = df[df['预期意图'] == intent]['问题'].tolist()
                examples[intent] = intent_queries
            
            print(f"[WhiteBoxInit] Loaded examples for {len(examples)} intents")
            return examples
        except Exception as e:
            print(f"[Warning] Failed to load examples: {e}")
            return {}
    
    def generate_initial_population(self, pop_size: int = POPULATION_SIZE, 
                                    classifier=None, 
                                    sample_size: int = 200) -> List[ProfileSet]:
        """
        生成初始种群
        
        Args:
            pop_size: 种群大小
            classifier: 可选的分类器，用于获取置信度低的样本
            sample_size: 用于置信度分析的样本数
        """
        if not self.agents_data.get('agents'):
            print("[Error] No agent data available")
            return []
        
        population = []
        
        # 生成pop_size个个体
        for i in range(pop_size):
            print(f"[WhiteBoxInit] Generating individual {i+1}/{pop_size}...")
            profile_set = self._generate_single_profile_set(variant_id=i)
            
            # 如果提供了分类器，使用置信度低的样本来优化B槽位
            if classifier is not None:
                print(f"[WhiteBoxInit] Refining with confidence analysis...")
                profile_set = self._refine_with_low_confidence(
                    profile_set, classifier, sample_size
                )
            
            population.append(profile_set)
        
        return population
    
    def _generate_single_profile_set(self, variant_id: int = 0) -> ProfileSet:
        """生成单个ProfileSet"""
        profiles = []
        
        all_tools = set()
        for agent in self.agents_data['agents']:
            for tool in agent.get('tools', []):
                all_tools.add(tool.get('name', ''))
        
        # 建立agent_name到intent的映射
        intent_to_agent = {v: k for k, v in INTENT_TO_AGENT.items()}
        
        for agent in self.agents_data['agents']:
            agent_name = agent['name']
            tools = agent.get('tools', [])
            description = agent.get('description', '')
            
            # 找到对应的intent
            intent = intent_to_agent.get(agent_name, "")
            
            # 生成C, B, R（使用规则生成，不调用LLM）
            c, b, r = self._generate_rule_based(agent_name, tools, description, all_tools, intent)
            
            profile = StructuredProfile(
                agent_name=agent_name,
                core_capability=c,
                boundary=b,
                rejection_scope=r
            )
            profiles.append(profile)
        
        return ProfileSet(profiles)
    
    def _generate_rule_based(self, agent_name: str, tools: List[Dict], 
                            description: str, all_tools: set, intent: str = "") -> tuple:
        """基于规则的生成（简化版）
        
        C = 工具描述
        B = 同类的一个例子（该智能体的查询），带引号
        R = 不同类的一个例子（其他智能体的查询），带引号并标注正确类别
        """
        # C: 直接使用工具描述
        c = description[:MAX_C_LENGTH]
        
        # 获取该智能体的例子（同类）
        agent_examples = []
        if intent and intent in self.examples_by_intent:
            agent_examples = self.examples_by_intent[intent]
        
        # B: 处理边界 = 一个同类例子，加引号
        if agent_examples:
            b_example = random.choice(agent_examples)
            b = f"包括'{b_example[:15]}'"
        else:
            b = f"包括'{description[:10]}'"
        
        # R: 拒绝范围 = 一个不同类例子（从其他智能体取），加引号并标注正确类别
        # INTENT_TO_AGENT 已经是 {'意图': '智能体'} 的映射，直接使用
        other_examples_with_agent = []
        for other_intent, examples in self.examples_by_intent.items():
            if other_intent != intent and examples:
                correct_agent = INTENT_TO_AGENT.get(other_intent, "其他")
                for ex in examples:
                    other_examples_with_agent.append((ex, correct_agent))
        
        if other_examples_with_agent:
            r_example, correct_agent = random.choice(other_examples_with_agent)
            r = f"不包括'{r_example[:10]}'（实际{correct_agent[:6]}）"
        else:
            r = "不包括其他类型"
        
        # 截断到长度限制（B和R限制40字以内）
        c = c[:MAX_C_LENGTH]
        b = b[:40]
        r = r[:40]
        
        return c, b, r
    
    def _refine_with_low_confidence(self, profile_set: ProfileSet, 
                                    classifier, 
                                    sample_size: int = 200) -> ProfileSet:
        """
        使用置信度低的同类样本优化B槽位
        
        逻辑：
        1. 采样样本
        2. 用初始描述分类所有样本
        3. 对每个智能体，找出"期望是该智能体但置信度低"的样本
        4. 用这些样本更新B槽位
        """
        import pandas as pd
        from config import INTENT_TO_AGENT
        
        try:
            # 加载样本
            df = pd.read_csv(self.examples_path)
            df = df[df['预期意图'] != 'oos']
            
            # 分层采样 - 确保每个类别都有样本
            unique_intents = df['预期意图'].unique()
            n_intents = len(unique_intents)
            samples_per_intent = max(1, sample_size // n_intents)
            
            sampled_dfs = []
            for intent in unique_intents:
                intent_df = df[df['预期意图'] == intent]
                # 每个意图至少取1个，最多取samples_per_intent个
                n = min(samples_per_intent, len(intent_df))
                if n > 0:
                    sampled_dfs.append(intent_df.sample(n=n))
            
            df = pd.concat(sampled_dfs).head(sample_size)
            
            queries = df['问题'].tolist()
            intents = df['预期意图'].tolist()
            
            # 分类所有样本
            print(f"[WhiteBoxInit] Classifying {len(queries)} samples for confidence analysis...")
            results = classifier.classify_batch(queries, profile_set)
            
            # 为每个智能体收集低置信度样本
            agent_to_low_conf = {}
            
            for intent, result, query in zip(intents, results, queries):
                expected_agent = INTENT_TO_AGENT.get(intent)
                if expected_agent and expected_agent == result.predicted_agent:
                    # 分类正确但置信度低
                    margin = result.get_margin()
                    if expected_agent not in agent_to_low_conf:
                        agent_to_low_conf[expected_agent] = []
                    agent_to_low_conf[expected_agent].append((query, margin))
            
            # 对每个智能体，排序并选择置信度最低的样本
            for agent_name in profile_set.profiles.keys():
                if agent_name in agent_to_low_conf and agent_to_low_conf[agent_name]:
                    # 按置信度排序（margin越小置信度越低）
                    low_conf_samples = sorted(agent_to_low_conf[agent_name], key=lambda x: x[1])
                    # 取置信度最低的一个样本
                    lowest_conf_query = low_conf_samples[0][0]
                    
                    # 更新B槽位（加引号）
                    profile = profile_set.get_profile(agent_name)
                    new_b = f"包括'{lowest_conf_query[:15]}'"
                    profile.boundary = new_b[:40]
                    print(f"[WhiteBoxInit] {agent_name} B槽位更新为低置信样本: {new_b[:40]}")
            
        except Exception as e:
            print(f"[Warning] Confidence refinement failed: {e}")
            import traceback
            traceback.print_exc()
        
        return profile_set
    
    def _generate_with_llm(self, agent_name: str, tools: List[Dict],
                          description: str, all_tools: set, variant_id: int, intent: str = "") -> tuple:
        """使用LLM生成
        
        生成C（核心能力）和简短的B（处理边界）、R（拒绝范围）
        """
        
        tools_text = "\n".join([
            f"- {t.get('name', '')}: {t.get('description', '')}"
            for t in tools
        ])
        
        # 获取代表性例子
        examples_text = ""
        if intent and intent in self.examples_by_intent:
            examples = self.examples_by_intent[intent][:3]
            examples_text = "\n".join([f"- {ex}" for ex in examples])
        
        # 根据是否有示例构建不同的prompt
        if examples_text:
            examples_instruction = f"该智能体的典型查询示例（必须参考这些示例）：\n{examples_text}\n\n重要：B和R的描述必须基于上述真实示例，不能凭空编造例子！"
        else:
            examples_instruction = "该智能体暂无典型查询示例。\n\n重要：B和R的描述只能基于智能体说明，不能编造具体例子！"
        
        prompt = f"""根据以下智能体信息，生成核心能力、处理边界和拒绝范围描述。

智能体名称：{agent_name}
智能体说明：{description}
可用工具：
{tools_text}

{examples_instruction}

请生成以下内容：
1. 核心能力（C）：严格≤{MAX_C_LENGTH}字，描述主要负责什么
2. 处理边界（B）：严格≤20字，只描述该智能体处理什么
   - 有示例时：简短描述+1-2个真实示例（如：包括股票诊断，如华西股份、长信科技）
   - 无示例时：只写简短描述，不要编造例子（如：包括股票诊断）
3. 拒绝范围（R）：严格≤20字，只描述该智能体不包括什么
   - 格式：不包括xxx
   - 不要编造具体例子

要求：
- **极其简短**：B和R控制在20字以内，越短越好
- **真实示例**：有示例时必须用提供的真实示例，无示例时绝不编造
- **不要解释**：直接输出结果

请按以下格式输出：
[核心能力]xxx
[处理边界]xxx  
[拒绝范围]xxx"""
        
        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 + variant_id * 0.05,  # 递增温度产生变体
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析输出
            c = self._extract_section(content, "核心能力")
            b = self._extract_section(content, "处理边界")
            r = self._extract_section(content, "拒绝范围")
            
            # 验证长度
            c = c[:MAX_C_LENGTH]
            b = b[:20] if b else f"包括{description[:15]}"[:20]
            r = r[:20] if r else "不包括其他"[:20]
            
            return c, b, r
            
        except Exception as e:
            print(f"[Error] LLM generation failed: {e}, falling back to rule-based")
            return self._generate_rule_based(agent_name, tools, description, all_tools, intent)
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """从文本中提取特定章节"""
        import re
        pattern = rf'\[{section_name}\](.+?)(?=\[|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match and match.group(1):
            return match.group(1).strip()
        return f"处理{section_name}相关请求"


def generate_initial_population(pop_size: int = POPULATION_SIZE,
                                classifier=None,
                                sample_size: int = 200) -> List[ProfileSet]:
    """便捷函数：生成初始种群

    Args:
        pop_size: 种群大小
        classifier: 可选的分类器，用于获取置信度低的样本优化B槽位
        sample_size: 用于置信度分析的样本数
    """
    initializer = WhiteBoxInitializer()
    return initializer.generate_initial_population(pop_size, classifier, sample_size)
