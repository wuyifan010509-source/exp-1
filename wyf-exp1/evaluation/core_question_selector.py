"""
核心问题选择器 - 从历史数据中提取关键问题用于评估

功能：
1. 从HISTORICAL_LOGS加载历史数据
2. 使用底层API获取Top-1和Top-2概率
3. 识别两类核心问题：
   - 模型不确定的问题（Top-1和Top-2概率差距小）
   - 模型分错的问题（预测与预期不符）
4. 选出500个核心问题并保存
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.classifier import QwenClassifier, ClassifyResult, MockClassifier
from structured_profile import ProfileSet
from config import HISTORICAL_LOGS_PATH, INTENT_TO_AGENT, RESULTS_DIR


@dataclass
class CoreQuestion:
    """核心问题数据结构"""
    query: str
    expected_intent: str
    expected_agent: str
    predicted_agent: str
    top1_prob: float
    top2_prob: float
    margin: float  # top1 - top2
    is_correct: bool
    all_probs: Dict[str, float]  # 所有类别的概率分布
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CoreQuestion':
        """从字典创建"""
        return cls(**data)


class CoreQuestionSelector:
    """
    核心问题选择器
    
    用于从历史数据中选择最具代表性的问题作为评估集
    选择标准：
    1. 模型不确定的问题（margin < threshold）
    2. 模型分错的问题（is_correct = False）
    """
    
    def __init__(self, classifier, margin_threshold: float = 0.1):
        """
        初始化
        
        Args:
            classifier: 分类器实例（QwenClassifier或MockClassifier）
            margin_threshold: margin阈值，低于此值视为不确定
        """
        self.classifier = classifier
        self.margin_threshold = margin_threshold
        self.core_questions: List[CoreQuestion] = []
        self.stats: Dict = {}
        print(f"[CoreQuestionSelector] 初始化完成，margin阈值: {margin_threshold}")
    
    def load_historical_data(self, sample_size: int = None) -> pd.DataFrame:
        """
        加载历史数据
        
        Args:
            sample_size: 如果指定，则随机采样这么多条
            
        Returns:
            DataFrame包含'问题'和'预期意图'列
        """
        print(f"\n[1/5] 加载历史数据: {HISTORICAL_LOGS_PATH}")
        df = pd.read_csv(HISTORICAL_LOGS_PATH)
        
        # 过滤掉oos意图
        df = df[df['预期意图'] != 'oos']
        
        print(f"[1/5] 加载完成，共 {len(df)} 条数据（已过滤oos）")
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"[1/5] 随机采样 {sample_size} 条")
        
        return df
    
    def get_probabilities_from_api(self, query: str, profiles: ProfileSet) -> Tuple[Dict[str, float], str]:
        """
        从底层API获取完整的概率分布
        
        这是关键方法：调用底层API获取logprobs，然后计算每个类别的概率
        
        Args:
            query: 问题文本
            profiles: 智能体画像集合
            
        Returns:
            (概率分布字典, 原始响应文本)
        """
        try:
            # 使用classifier的classify方法获取结果
            # 启用logprobs以获取真实的token概率分布
            result = self.classifier.classify(query, profiles, use_logprobs=True)
            
            # 获取概率分布
            all_probs = result.confidence_scores
            
            # 如果概率为空，尝试从其他方式获取
            if not all_probs or all(all_probs.get(k, 0) == 0 for k in all_probs):
                # 如果没有概率，基于agent名称做简单分配
                all_probs = self._estimate_probs_from_response(result.raw_response, profiles, result.predicted_agent)
            
            return all_probs, result.raw_response
            
        except Exception as e:
            print(f"[Error] 获取概率失败: {e}")
            # 返回均匀分布
            all_probs = {name: 1.0 / len(profiles.profiles) for name in profiles.profiles.keys()}
            return all_probs, str(e)
    
    def _estimate_probs_from_response(self, response: str, profiles: ProfileSet, predicted_agent: str) -> Dict[str, float]:
        """
        当API没有返回概率时，基于响应文本估计概率
        这是一个备选方案
        """
        agents = list(profiles.profiles.keys())
        probs = {}
        
        # 给预测到的agent高概率
        for agent in agents:
            if agent == predicted_agent:
                probs[agent] = 0.6
            elif agent in response:
                probs[agent] = 0.3
            else:
                probs[agent] = 0.1 / (len(agents) - 1) if len(agents) > 1 else 0.1
        
        # 归一化
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def calculate_margin(self, probs: Dict[str, float]) -> Tuple[float, float, float]:
        """
        计算Top-1和Top-2的margin
        
        Args:
            probs: 概率分布字典
            
        Returns:
            (top1_prob, top2_prob, margin)
        """
        if not probs:
            return 0.0, 0.0, 0.0
        
        sorted_probs = sorted(probs.values(), reverse=True)
        top1 = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2
        
        return top1, top2, margin
    
    def classify_and_analyze(self, query: str, expected_intent: str, profiles: ProfileSet) -> CoreQuestion:
        """
        对单个问题进行分类并分析
        
        Args:
            query: 问题文本
            expected_intent: 预期意图
            profiles: 智能体画像
            
        Returns:
            CoreQuestion对象
        """
        # 获取概率分布
        all_probs, raw_response = self.get_probabilities_from_api(query, profiles)
        
        # 计算margin
        top1_prob, top2_prob, margin = self.calculate_margin(all_probs)
        
        # 确定预测的agent
        predicted_agent = max(all_probs.items(), key=lambda x: x[1])[0] if all_probs else ""
        
        # 确定预期的agent
        expected_agent = INTENT_TO_AGENT.get(expected_intent, "")
        
        # 判断是否分类正确
        is_correct = (predicted_agent == expected_agent)
        
        return CoreQuestion(
            query=query,
            expected_intent=expected_intent,
            expected_agent=expected_agent,
            predicted_agent=predicted_agent,
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            margin=margin,
            is_correct=is_correct,
            all_probs=all_probs
        )
    
    def select_core_questions(self, profiles: ProfileSet, 
                             total_size: int = 500,
                             uncertainty_ratio: float = 0.5,
                             sample_size: int = None) -> List[CoreQuestion]:
        """
        选择核心问题
        
        Args:
            profiles: 智能体画像（用于评估）
            total_size: 要选出的核心问题总数
            uncertainty_ratio: 不确定问题的比例（剩余为错误问题）
            sample_size: 从历史数据中采样的数量（None表示全部）
            
        Returns:
            CoreQuestion列表
        """
        print("\n" + "="*80)
        print("开始选择核心问题")
        print("="*80)
        
        # 1. 加载数据
        df = self.load_historical_data(sample_size=sample_size)
        
        # 2. 对每个问题进行分类和分析
        print(f"\n[2/5] 对所有问题进行分类分析（共 {len(df)} 条）...")
        all_questions = []
        
        for idx, row in df.iterrows():
            if (idx + 1) % 50 == 0:
                print(f"  已处理 {idx+1}/{len(df)} 条...")
            
            query = row['问题']
            intent = row['预期意图']
            
            try:
                core_q = self.classify_and_analyze(query, intent, profiles)
                all_questions.append(core_q)
            except Exception as e:
                print(f"  [Warning] 处理失败: {e}")
                continue
        
        print(f"[2/5] 分类完成，共分析 {len(all_questions)} 条")
        
        # 3. 按Margin排序选择核心问题
        print(f"\n[3/5] 识别核心问题...")
        
        # 按Margin从小到大排序（越小的越不确定）
        all_questions_sorted = sorted(all_questions, key=lambda x: x.margin)
        
        # 取前10%作为高优先级（最不确定的）
        high_priority_count = max(1, int(total_size * 0.1))
        high_priority_questions = all_questions_sorted[:high_priority_count]
        
        # 剩下的问题
        remaining_questions = all_questions_sorted[high_priority_count:]
        
        # 从剩下的问题中按错误/正确分类选择
        remaining_errors = [q for q in remaining_questions if not q.is_correct]
        remaining_correct = [q for q in remaining_questions if q.is_correct]
        
        # 优先选择错误问题，其次选择正确问题
        needed = total_size - high_priority_count
        n_errors = min(len(remaining_errors), needed // 2 + needed % 2)
        n_correct = min(len(remaining_correct), needed // 2)
        
        selected_errors = remaining_errors[:n_errors]
        selected_correct = remaining_correct[:n_correct]
        
        # 如果还不够，从剩余中补充
        selected = high_priority_questions + selected_errors + selected_correct
        existing_queries = {q.query for q in selected}
        
        if len(selected) < total_size:
            remaining = [q for q in remaining_questions if q.query not in existing_queries]
            np.random.shuffle(remaining)
            selected.extend(remaining[:total_size - len(selected)])
        
        self.core_questions = selected[:total_size]
        
        print(f"[3/5] 选择完成，共 {len(self.core_questions)} 个核心问题")
        print(f"    - 高优先级（Margin最低{high_priority_count}个）: {len(high_priority_questions)}")
        print(f"    - 错误问题: {len([q for q in self.core_questions if not q.is_correct])}")
        print(f"    - 不确定问题(Margin<{self.margin_threshold}): {len([q for q in self.core_questions if q.margin < self.margin_threshold])}")
        
        # 5. 统计信息
        self._generate_stats(all_questions)
        
        return self.core_questions
    
    def _generate_stats(self, all_questions: List[CoreQuestion]):
        """生成统计信息"""
        margins = [q.margin for q in all_questions]
        correct_count = sum(1 for q in all_questions if q.is_correct)
        
        self.stats = {
            "total_analyzed": len(all_questions),
            "total_correct": correct_count,
            "total_errors": len(all_questions) - correct_count,
            "accuracy": correct_count / len(all_questions) if all_questions else 0,
            "margin_mean": np.mean(margins),
            "margin_std": np.std(margins),
            "margin_median": np.median(margins),
            "margin_min": min(margins) if margins else 0,
            "margin_max": max(margins) if margins else 0,
            "low_margin_count": sum(1 for m in margins if m < self.margin_threshold)
        }
        
        print(f"\n[4/5] 统计分析")
        print(f"  总体准确率: {self.stats['accuracy']:.2%}")
        print(f"  Margin均值: {self.stats['margin_mean']:.4f}")
        print(f"  Margin中位数: {self.stats['margin_median']:.4f}")
        print(f"  Margin标准差: {self.stats['margin_std']:.4f}")
        print(f"  低Margin(<{self.margin_threshold})数量: {self.stats['low_margin_count']}")
    
    def save_core_questions(self, filepath: str = None) -> str:
        """
        保存核心问题到文件
        
        Args:
            filepath: 保存路径，默认保存到results/core_questions.pkl
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"core_questions_{timestamp}.pkl")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "core_questions": [q.to_dict() for q in self.core_questions],
            "stats": self.stats,
            "margin_threshold": self.margin_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n[5/5] 核心问题已保存: {filepath}")
        
        # 同时保存一个JSON版本便于查看
        json_path = filepath.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[5/5] JSON版本: {json_path}")
        
        return filepath
    
    def load_core_questions(self, filepath: str) -> List[CoreQuestion]:
        """
        从文件加载核心问题
        
        Args:
            filepath: pkl文件路径
            
        Returns:
            CoreQuestion列表
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.core_questions = [CoreQuestion.from_dict(q) for q in data['core_questions']]
        self.stats = data['stats']
        self.margin_threshold = data.get('margin_threshold', 0.1)
        
        print(f"[CoreQuestionSelector] 已加载 {len(self.core_questions)} 个核心问题")
        return self.core_questions
    
    def get_evaluation_data(self) -> Tuple[List[str], List[str], List[bool]]:
        """
        获取用于评估的数据格式
        
        Returns:
            (queries列表, intents列表, is_boundary列表)
            注意：is_boundary对核心问题默认为False
        """
        queries = [q.query for q in self.core_questions]
        intents = [q.expected_intent for q in self.core_questions]
        is_boundary = [False] * len(self.core_questions)  # 核心问题没有边界标记
        
        return queries, intents, is_boundary
    
    def print_sample_questions(self, n: int = 5, show_type: str = 'uncertain'):
        """
        打印示例问题
        
        Args:
            n: 显示数量
            show_type: 'uncertain'(不确定), 'errors'(错误), 'all'(所有)
        """
        print(f"\n{'='*80}")
        print(f"示例问题 ({show_type})")
        print('='*80)
        
        if show_type == 'uncertain':
            questions = [q for q in self.core_questions if q.margin < self.margin_threshold]
            questions.sort(key=lambda x: x.margin)
        elif show_type == 'errors':
            questions = [q for q in self.core_questions if not q.is_correct]
        else:
            questions = self.core_questions
        
        for i, q in enumerate(questions[:n], 1):
            print(f"\n[{i}] {q.query}")
            print(f"    预期: {q.expected_intent} ({q.expected_agent})")
            print(f"    预测: {q.predicted_agent}")
            print(f"    Top-1概率: {q.top1_prob:.4f}")
            print(f"    Top-2概率: {q.top2_prob:.4f}")
            print(f"    Margin: {q.margin:.4f}")
            print(f"    是否正确: {'是' if q.is_correct else '否'}")
            print(f"    概率分布: {q.all_probs}")


# ==================== 便捷函数 ====================

def create_core_questions(profiles: ProfileSet, 
                         classifier,
                         total_size: int = 500,
                         margin_threshold: float = 0.1,
                         save: bool = True) -> Tuple[List[CoreQuestion], str]:
    """
    便捷函数：创建并保存核心问题
    
    Args:
        profiles: 智能体画像
        classifier: 分类器
        total_size: 核心问题总数
        margin_threshold: margin阈值
        save: 是否保存
        
    Returns:
        (核心问题列表, 保存路径)
    """
    selector = CoreQuestionSelector(classifier, margin_threshold=margin_threshold)
    core_questions = selector.select_core_questions(
        profiles, 
        total_size=total_size,
        uncertainty_ratio=0.5
    )
    
    # 打印示例
    selector.print_sample_questions(n=5, show_type='uncertain')
    selector.print_sample_questions(n=5, show_type='errors')
    
    if save:
        filepath = selector.save_core_questions()
        return core_questions, filepath
    
    return core_questions, ""


def load_core_questions_for_evaluation(filepath: str) -> Tuple[List[str], List[str], List[bool]]:
    """
    加载核心问题用于评估
    
    Args:
        filepath: 核心问题文件路径
        
    Returns:
        (queries, intents, is_boundary) 用于EvaluationPipeline
    """
    selector = CoreQuestionSelector(classifier=None)
    selector.load_core_questions(filepath)
    return selector.get_evaluation_data()


def create_test_profile():
    """创建测试用的ProfileSet"""
    from config import INTENT_TO_AGENT
    from structured_profile import StructuredProfile
    
    profiles = []
    for intent, agent_name in INTENT_TO_AGENT.items():
        profile = StructuredProfile(
            agent_name=agent_name,
            core_capability=f"处理{intent}相关的问题",
            boundary=f"{intent}范围",
            rejection_scope=f"非{intent}问题"
        )
        profiles.append(profile)
    
    return ProfileSet(profiles)


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("核心问题选择器测试")
    print("="*80)
    
    # 使用Mock分类器进行测试
    from structured_profile import ProfileSet, StructuredProfile
    
    # 创建测试用的profile
    test_profile = create_test_profile()
    print(f"[测试] 创建测试Profile，共 {len(test_profile.profiles)} 个智能体")
    
    # 创建Mock分类器
    classifier = MockClassifier(seed=42)
    
    # 创建选择器
    selector = CoreQuestionSelector(classifier, margin_threshold=0.1)
    
    # 选择核心问题（先测试50个）
    print("\n开始测试核心问题选择...")
    core_questions = selector.select_core_questions(
        test_profile,
        total_size=50,
        uncertainty_ratio=0.5,
        sample_size=200  # 先采样200条测试
    )
    
    # 打印统计
    print(f"\n{'='*80}")
    print("统计信息")
    print('='*80)
    for key, value in selector.stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 保存
    filepath = selector.save_core_questions()
    print(f"\n测试完成！文件保存至: {filepath}")
