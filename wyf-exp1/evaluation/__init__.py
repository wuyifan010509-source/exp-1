"""
评估流水线编排
"""
import random
from typing import List, Tuple, Dict, Optional
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GOLDEN_TEST_PATH, HISTORICAL_LOGS_PATH, INTENT_TO_AGENT, EVAL_SUBSET_SIZE,
    LAMBDA_PENALTY, MAX_PROFILE_LENGTH
)
from structured_profile import ProfileSet
from evaluation.classifier import IntentClassifier, ClassifyResult
from evaluation.metrics import (
    compute_accuracy, compute_boundary_accuracy, compute_average_margin,
    compute_fitness, get_bad_cases, compute_metrics_report
)


class EvaluationPipeline:
    """评估流水线"""
    
    def __init__(self, classifier: IntentClassifier):
        self.classifier = classifier
        self.intent_to_agent = INTENT_TO_AGENT
        self.agent_names = list(set(INTENT_TO_AGENT.values()))
    
    def load_test_data(self, filepath: str = HISTORICAL_LOGS_PATH, 
                       sample_size: Optional[int] = None) -> Tuple[List[str], List[str], List[bool]]:
        """
        加载测试数据
        
        Returns:
            (queries, intents, is_boundary)
        """
        df = pd.read_csv(filepath)
        
        # 去除oos类别
        df = df[df['预期意图'] != 'oos']
        
        queries = df['问题'].tolist()
        intents = df['预期意图'].tolist()
        # HISTORICAL_LOGS.csv没有边界列，GOLDEN_TEST.csv有
        if '是否处于意图边界' in df.columns:
            is_boundary = df['是否处于意图边界'].fillna('否').map(lambda x: str(x).strip() == '是').tolist()
        else:
            is_boundary = [False] * len(queries)
        
        # 子集采样
        if sample_size and sample_size < len(queries):
            # 分层采样，确保每个意图都有样本
            indices = []
            for intent in set(intents):
                intent_indices = [i for i, x in enumerate(intents) if x == intent]
                n_samples = max(1, int(sample_size * len(intent_indices) / len(queries)))
                indices.extend(random.sample(intent_indices, min(n_samples, len(intent_indices))))
            
            # 如果不够，随机补充
            while len(indices) < sample_size:
                remaining = list(set(range(len(queries))) - set(indices))
                if remaining:
                    indices.append(random.choice(remaining))
                else:
                    break
            
            indices = indices[:sample_size]
            queries = [queries[i] for i in indices]
            intents = [intents[i] for i in indices]
            is_boundary = [is_boundary[i] for i in indices]
        
        return queries, intents, is_boundary
    
    def evaluate(self, profiles: ProfileSet, 
                test_data: Optional[Tuple[List[str], List[str], List[bool]]] = None,
                sample_size: Optional[int] = None) -> Dict:
        """
        评估ProfileSet的性能
        
        Args:
            profiles: 要评估的描述集合
            test_data: 测试数据，如果为None则加载默认数据
            sample_size: 评估样本数
        
        Returns:
            评估结果字典
        """
        if test_data is None:
            queries, intents, is_boundary = self.load_test_data(
                sample_size=sample_size or EVAL_SUBSET_SIZE
            )
        else:
            queries, intents, is_boundary = test_data
        
        print(f"[Evaluate] Evaluating {len(queries)} samples...")
        
        # 批量分类
        results = self.classifier.classify_batch(queries, profiles)
        
        # 准备预测和标签
        predictions = [r.predicted_agent for r in results]
        labels = [self.intent_to_agent.get(intent) for intent in intents]
        
        # 计算指标
        accuracy = compute_accuracy(predictions, labels)
        boundary_accuracy = compute_boundary_accuracy(predictions, labels, is_boundary)
        avg_margin = compute_average_margin(results)
        avg_length = profiles.average_length()
        
        # 计算适应度
        fitness = compute_fitness(accuracy, avg_length, MAX_PROFILE_LENGTH, LAMBDA_PENALTY)
        
        # 获取错误案例
        confidence_scores = [r.confidence_scores for r in results]
        bad_cases = get_bad_cases(predictions, labels, queries, confidence_scores, top_k=20)
        
        return {
            "accuracy": accuracy,
            "boundary_accuracy": boundary_accuracy,
            "average_margin": avg_margin,
            "average_length": avg_length,
            "fitness": fitness,
            "total_samples": len(queries),
            "boundary_samples": sum(is_boundary),
            "bad_cases": bad_cases
        }
    
    def compute_fitness_simple(self, profiles: ProfileSet, 
                               sample_size: Optional[int] = None) -> float:
        """
        简化的适应度计算（只返回fitness值）
        
        用于GA中快速评估
        """
        queries, intents, is_boundary = self.load_test_data(
            sample_size=sample_size or EVAL_SUBSET_SIZE
        )
        
        results = self.classifier.classify_batch(queries, profiles)
        
        predictions = [r.predicted_agent for r in results]
        labels = [self.intent_to_agent.get(intent) for intent in intents]
        
        accuracy = compute_accuracy(predictions, labels)
        avg_length = profiles.average_length()
        
        return compute_fitness(accuracy, avg_length, MAX_PROFILE_LENGTH, LAMBDA_PENALTY)
    
    def get_bad_cases(self, profiles: ProfileSet, 
                      sample_size: Optional[int] = None,
                      top_k: int = 10) -> List[Dict]:
        """
        获取分类错误的案例（Bad Cases）
        
        Args:
            profiles: 要评估的ProfileSet
            sample_size: 评估样本数
            top_k: 返回多少个错误案例
        
        Returns:
            Bad cases列表，每个包含query、expected_intent、predicted_agent等
        """
        queries, intents, is_boundary = self.load_test_data(
            sample_size=sample_size or EVAL_SUBSET_SIZE
        )
        
        results = self.classifier.classify_batch(queries, profiles)
        
        # 组合结果（注意：get_bad_cases期望3个元素的元组）
        results_with_query = []
        for intent, result, query in zip(intents, results, queries):
            results_with_query.append((intent, result, query))
        
        # 获取错误案例
        bad_cases = get_bad_cases(results_with_query, self.intent_to_agent, top_k=top_k)
        
        return bad_cases
