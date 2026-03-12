"""
核心问题评估集成 - 用于替代黄金测试集的评估器

功能：
1. 自动从HISTORICAL_LOGS生成或使用已存在的核心问题
2. 提供与原始黄金测试集兼容的评估接口
3. 支持增量更新核心问题
"""
import os
import sys
import glob
import pickle
from typing import Tuple, Optional, Callable
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.core_question_selector import (
    CoreQuestionSelector, CoreQuestion, 
    create_core_questions, load_core_questions_for_evaluation
)
from evaluation.core_question_analyzer import CoreQuestionAnalyzer
from evaluation.classifier import QwenClassifier, MockClassifier
from evaluation import EvaluationPipeline
from structured_profile import ProfileSet
from config import RESULTS_DIR


class CoreQuestionEvaluator:
    """
    核心问题评估器
    
    替代原有的黄金测试集评估，使用从历史数据中提取的核心问题
    """
    
    def __init__(self, classifier, core_questions_path: str = None):
        """
        初始化
        
        Args:
            classifier: 分类器实例
            core_questions_path: 核心问题文件路径，None则自动查找或生成
        """
        self.classifier = classifier
        self.core_questions_path = core_questions_path
        self.core_questions = []
        self.selector = CoreQuestionSelector(classifier)
        
        # 尝试加载已存在的核心问题
        if self.core_questions_path and os.path.exists(self.core_questions_path):
            self.load_core_questions(self.core_questions_path)
        else:
            # 自动查找最新的核心问题文件
            latest_file = self._find_latest_core_questions()
            if latest_file:
                self.load_core_questions(latest_file)
    
    def _find_latest_core_questions(self) -> Optional[str]:
        """查找最新的核心问题文件"""
        pattern = os.path.join(RESULTS_DIR, 'core_questions_*.pkl')
        files = glob.glob(pattern)
        if files:
            return max(files, key=os.path.getctime)
        return None
    
    def load_core_questions(self, filepath: str):
        """加载核心问题"""
        print(f"[CoreQuestionEvaluator] 加载核心问题: {filepath}")
        self.core_questions = self.selector.load_core_questions(filepath)
        self.core_questions_path = filepath
        print(f"[CoreQuestionEvaluator] 已加载 {len(self.core_questions)} 个核心问题")
    
    def generate_core_questions(self, 
                               profiles: ProfileSet,
                               total_size: int = 500,
                               margin_threshold: float = 0.1,
                               sample_size: int = None,
                               force_regenerate: bool = False) -> str:
        """
        生成新的核心问题
        
        Args:
            profiles: 智能体画像（用于评估）
            total_size: 要选出的核心问题总数
            margin_threshold: margin阈值
            sample_size: 从历史数据中采样的数量
            force_regenerate: 是否强制重新生成（即使已有文件）
            
        Returns:
            保存的文件路径
        """
        # 检查是否已存在且不需要强制重新生成
        if not force_regenerate and self.core_questions:
            print(f"[CoreQuestionEvaluator] 核心问题已存在（{len(self.core_questions)}个），跳过生成")
            print(f"[CoreQuestionEvaluator] 使用现有文件: {self.core_questions_path}")
            return self.core_questions_path
        
        print("\n" + "="*80)
        print("生成核心问题评估集")
        print("="*80)
        
        # 创建新的选择器（避免缓存问题）
        self.selector = CoreQuestionSelector(self.classifier, margin_threshold=margin_threshold)
        
        # 选择核心问题
        self.core_questions = self.selector.select_core_questions(
            profiles=profiles,
            total_size=total_size,
            uncertainty_ratio=0.5,  # 50%不确定，50%错误
            sample_size=sample_size
        )
        
        # 保存
        filepath = self.selector.save_core_questions()
        self.core_questions_path = filepath
        
        return filepath
    
    def get_evaluation_data(self) -> Tuple[list, list, list]:
        """
        获取评估数据（与黄金测试集格式一致）
        
        Returns:
            (queries, intents, is_boundary)
        """
        if not self.core_questions:
            raise ValueError("核心问题未加载，请先调用generate_core_questions或load_core_questions")
        
        return self.selector.get_evaluation_data()
    
    def create_fitness_function(self, pipeline: EvaluationPipeline) -> Callable:
        """
        创建适应度函数（用于GreedyOptimizer）
        
        Args:
            pipeline: EvaluationPipeline实例
            
        Returns:
            fitness函数
        """
        # 获取核心问题数据
        queries, intents, is_boundary = self.get_evaluation_data()
        
        print(f"[CoreQuestionEvaluator] 使用{len(queries)}个核心问题进行评估")
        
        def fitness_func(profile_set: ProfileSet, test_data=None) -> float:
            """使用核心问题评估准确率"""
            try:
                # 始终使用核心问题集
                result = pipeline.evaluate(
                    profile_set, 
                    test_data=(queries, intents, is_boundary)
                )
                
                fitness = result['fitness']
                print(f"    Fitness: {fitness:.4f} (Acc: {result['accuracy']:.2%})")
                return fitness
            except Exception as e:
                print(f"    [Error] {e}")
                import traceback
                traceback.print_exc()
                return 0.0
        
        return fitness_func
    
    def analyze_current_core_questions(self):
        """分析当前的核心问题质量"""
        if not self.core_questions:
            print("[CoreQuestionEvaluator] 没有加载核心问题")
            return None
        
        analyzer = CoreQuestionAnalyzer()
        analyzer.core_questions = self.core_questions
        analyzer.stats = self.selector.stats
        
        # 显示分析结果
        analyzer.show_summary()
        analyzer.analyze_margin_distribution()
        analyzer.verify_selection_quality()
        
        return analyzer
    
    def get_stats(self) -> dict:
        """获取核心问题统计信息"""
        if not self.core_questions:
            return {}
        
        return {
            "total": len(self.core_questions),
            "uncertain": sum(1 for q in self.core_questions if q.margin < 0.1),
            "errors": sum(1 for q in self.core_questions if not q.is_correct),
            "stats": self.selector.stats
        }
    
    def print_sample_questions(self, n: int = 5):
        """打印示例问题"""
        if not self.core_questions:
            print("[CoreQuestionEvaluator] 没有加载核心问题")
            return
        
        analyzer = CoreQuestionAnalyzer()
        analyzer.core_questions = self.core_questions
        
        analyzer.show_sample_with_details(n=n, filter_type='uncertain')
        analyzer.show_sample_with_details(n=n, filter_type='errors')


# ==================== 便捷函数 ====================

def setup_core_question_evaluation(classifier, 
                                   profiles: ProfileSet,
                                   force_regenerate: bool = False,
                                   total_size: int = 500) -> CoreQuestionEvaluator:
    """
    便捷函数：设置核心问题评估
    
    Args:
        classifier: 分类器实例
        profiles: 智能体画像
        force_regenerate: 是否强制重新生成
        total_size: 核心问题总数
        
    Returns:
        CoreQuestionEvaluator实例
    """
    evaluator = CoreQuestionEvaluator(classifier)
    
    # 生成或加载核心问题
    evaluator.generate_core_questions(
        profiles=profiles,
        total_size=total_size,
        margin_threshold=0.1,
        force_regenerate=force_regenerate
    )
    
    # 打印统计
    stats = evaluator.get_stats()
    print(f"\n[CoreQuestionEvaluator] 评估集统计:")
    print(f"  总数: {stats.get('total', 0)}")
    print(f"  不确定: {stats.get('uncertain', 0)}")
    print(f"  错误: {stats.get('errors', 0)}")
    
    return evaluator


def create_fitness_function_with_core_questions(
    classifier,
    pipeline: EvaluationPipeline,
    profiles: ProfileSet,
    core_questions_path: str = None,
    force_regenerate: bool = False
) -> Callable:
    """
    便捷函数：直接使用核心问题创建适应度函数
    
    这是最主要的接口，可以直接在run_greedy.py中使用
    
    Args:
        classifier: 分类器
        pipeline: EvaluationPipeline
        profiles: 初始画像（用于生成核心问题）
        core_questions_path: 核心问题文件路径（可选）
        force_regenerate: 是否强制重新生成
        
    Returns:
        fitness函数
    """
    evaluator = CoreQuestionEvaluator(classifier, core_questions_path)
    
    # 如果需要生成或强制重新生成
    if force_regenerate or not evaluator.core_questions:
        evaluator.generate_core_questions(
            profiles=profiles,
            total_size=500,
            margin_threshold=0.1,
            force_regenerate=force_regenerate
        )
    
    return evaluator.create_fitness_function(pipeline)


if __name__ == "__main__":
    """测试核心问题评估器"""
    print("="*80)
    print("核心问题评估器测试")
    print("="*80)
    
    # 使用Mock分类器测试
    from structured_profile import StructuredProfile, ProfileSet
    from config import INTENT_TO_AGENT
    
    # 创建测试Profile
    def create_test_profile():
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
    
    test_profile = create_test_profile()
    print(f"[测试] 创建测试Profile，共 {len(test_profile.profiles)} 个智能体")
    
    # 创建分类器
    classifier = MockClassifier(seed=42)
    
    # 创建评估器
    evaluator = CoreQuestionEvaluator(classifier)
    
    # 生成核心问题（测试用50个）
    print("\n[测试] 生成核心问题...")
    filepath = evaluator.generate_core_questions(
        profiles=test_profile,
        total_size=50,
        margin_threshold=0.1,
        sample_size=200,
        force_regenerate=True
    )
    
    # 获取评估数据
    queries, intents, is_boundary = evaluator.get_evaluation_data()
    print(f"\n[测试] 获取评估数据: {len(queries)} 条")
    print(f"  示例问题: {queries[0]}")
    print(f"  示例意图: {intents[0]}")
    
    # 分析核心问题
    print("\n[测试] 分析核心问题质量...")
    evaluator.analyze_current_core_questions()
    
    # 测试适应度函数
    print("\n[测试] 测试适应度函数...")
    pipeline = EvaluationPipeline(classifier)
    fitness_func = evaluator.create_fitness_function(pipeline)
    
    # 计算一次适应度
    fitness = fitness_func(test_profile)
    print(f"\n[测试] 适应度: {fitness:.4f}")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
