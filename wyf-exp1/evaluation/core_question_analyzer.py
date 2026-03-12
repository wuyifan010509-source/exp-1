"""
核心问题分析工具 - 可视化和验证核心问题质量
"""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.core_question_selector import CoreQuestion, CoreQuestionSelector


class CoreQuestionAnalyzer:
    """核心问题分析器 - 用于可视化和验证核心问题质量"""
    
    def __init__(self, filepath: str = None):
        """
        初始化
        
        Args:
            filepath: 核心问题文件路径（.pkl或.json）
        """
        self.filepath = filepath
        self.core_questions: List[CoreQuestion] = []
        self.stats: Dict = {}
        
        if filepath:
            self.load(filepath)
    
    def load(self, filepath: str):
        """加载核心问题文件"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.core_questions = [CoreQuestion.from_dict(q) for q in data['core_questions']]
            self.stats = data.get('stats', {})
        else:  # .pkl
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.core_questions = [CoreQuestion.from_dict(q) for q in data['core_questions']]
            self.stats = data.get('stats', {})
        
        print(f"[Analyzer] 已加载 {len(self.core_questions)} 个核心问题")
    
    def show_summary(self):
        """显示摘要统计"""
        print("\n" + "="*80)
        print("核心问题摘要统计")
        print("="*80)
        
        # 基础统计
        n_total = len(self.core_questions)
        n_uncertain = sum(1 for q in self.core_questions if q.margin < 0.1)
        n_errors = sum(1 for q in self.core_questions if not q.is_correct)
        
        print(f"\n总体统计:")
        print(f"  核心问题总数: {n_total}")
        print(f"  不确定问题数: {n_uncertain} ({n_uncertain/n_total*100:.1f}%)")
        print(f"  错误问题数: {n_errors} ({n_errors/n_total*100:.1f}%)")
        
        # Margin统计
        margins = [q.margin for q in self.core_questions]
        print(f"\nMargin分布:")
        print(f"  最小值: {min(margins):.4f}")
        print(f"  最大值: {max(margins):.4f}")
        print(f"  均值: {np.mean(margins):.4f}")
        print(f"  中位数: {np.median(margins):.4f}")
        print(f"  标准差: {np.std(margins):.4f}")
        
        # 按意图类别统计
        print(f"\n意图类别分布:")
        intent_counts = Counter(q.expected_intent for q in self.core_questions)
        for intent, count in intent_counts.most_common():
            print(f"  {intent}: {count}条")
        
        # 按Agent统计
        print(f"\nAgent分布:")
        agent_counts = Counter(q.expected_agent for q in self.core_questions)
        for agent, count in agent_counts.most_common():
            print(f"  {agent}: {count}条")
    
    def show_uncertain_questions(self, n: int = 10):
        """显示不确定问题（Margin小）"""
        uncertain = [q for q in self.core_questions if q.margin < 0.1]
        uncertain.sort(key=lambda x: x.margin)
        
        print("\n" + "="*80)
        print(f"不确定问题示例 (Top-{n}, Margin < 0.1)")
        print("="*80)
        
        for i, q in enumerate(uncertain[:n], 1):
            print(f"\n[{i}] {q.query}")
            print(f"    预期: {q.expected_intent} → {q.expected_agent}")
            print(f"    预测: {q.predicted_agent}")
            print(f"    Top-1: {q.top1_prob:.4f} | Top-2: {q.top2_prob:.4f} | Margin: {q.margin:.4f}")
            print(f"    结果: {'✓ 正确' if q.is_correct else '✗ 错误'}")
            
            # 显示Top-3概率
            top3 = sorted(q.all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top-3概率分布:")
            for agent, prob in top3:
                marker = " ← 预测" if agent == q.predicted_agent else ""
                print(f"      {agent}: {prob:.4f}{marker}")
    
    def show_error_questions(self, n: int = 10):
        """显示错误分类问题"""
        errors = [q for q in self.core_questions if not q.is_correct]
        errors.sort(key=lambda x: x.margin)
        
        print("\n" + "="*80)
        print(f"错误分类问题示例 (Top-{n})")
        print("="*80)
        
        for i, q in enumerate(errors[:n], 1):
            print(f"\n[{i}] {q.query}")
            print(f"    预期: {q.expected_intent} → {q.expected_agent}")
            print(f"    预测: {q.predicted_agent}")
            print(f"    Top-1: {q.top1_prob:.4f} | Top-2: {q.top2_prob:.4f} | Margin: {q.margin:.4f}")
            
            # 显示Top-3概率
            top3 = sorted(q.all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top-3概率分布:")
            for agent, prob in top3:
                marker = " ← 预测" if agent == q.predicted_agent else ""
                correct_marker = " ← 正确" if agent == q.expected_agent else ""
                print(f"      {agent}: {prob:.4f}{marker}{correct_marker}")
    
    def show_correct_questions(self, n: int = 5):
        """显示正确分类问题"""
        correct = [q for q in self.core_questions if q.is_correct]
        correct.sort(key=lambda x: x.margin)
        
        print("\n" + "="*80)
        print(f"正确分类问题示例 (Top-{n})")
        print("="*80)
        
        for i, q in enumerate(correct[:n], 1):
            print(f"\n[{i}] {q.query}")
            print(f"    意图: {q.expected_intent} → Agent: {q.predicted_agent}")
            print(f"    Margin: {q.margin:.4f}")
    
    def analyze_margin_distribution(self):
        """分析Margin分布"""
        margins = [q.margin for q in self.core_questions]
        
        # 创建histogram
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        hist, _ = np.histogram(margins, bins=bins)
        
        print("\n" + "="*80)
        print("Margin分布直方图")
        print("="*80)
        
        for i in range(len(bins)-1):
            count = hist[i]
            bar = "█" * count
            print(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {bar} ({count})")
        
        # 显示分布比例
        print(f"\n分布比例:")
        total = len(margins)
        print(f"  Margin < 0.05: {sum(1 for m in margins if m < 0.05)} ({sum(1 for m in margins if m < 0.05)/total*100:.1f}%)")
        print(f"  Margin < 0.10: {sum(1 for m in margins if m < 0.10)} ({sum(1 for m in margins if m < 0.10)/total*100:.1f}%)")
        print(f"  Margin < 0.15: {sum(1 for m in margins if m < 0.15)} ({sum(1 for m in margins if m < 0.15)/total*100:.1f}%)")
        print(f"  Margin > 0.30: {sum(1 for m in margins if m > 0.30)} ({sum(1 for m in margins if m > 0.30)/total*100:.1f}%)")
    
    def analyze_agent_confusion(self):
        """分析Agent混淆矩阵"""
        print("\n" + "="*80)
        print("Agent混淆分析")
        print("="*80)
        
        # 统计每个agent被误分到哪些agent
        confusion = Counter((q.expected_agent, q.predicted_agent) for q in self.core_questions if not q.is_correct)
        
        print("\n最常见的误分类模式:")
        for (true_agent, pred_agent), count in confusion.most_common(20):
            print(f"  {true_agent} → {pred_agent}: {count}次")
    
    def analyze_by_intent(self, intent: str):
        """分析特定意图类别的问题"""
        intent_questions = [q for q in self.core_questions if q.expected_intent == intent]
        
        if not intent_questions:
            print(f"没有找到意图为 '{intent}' 的问题")
            return
        
        print("\n" + "="*80)
        print(f"意图类别分析: {intent}")
        print("="*80)
        
        n_total = len(intent_questions)
        n_correct = sum(1 for q in intent_questions if q.is_correct)
        n_uncertain = sum(1 for q in intent_questions if q.margin < 0.1)
        
        print(f"\n统计:")
        print(f"  总数: {n_total}")
        print(f"  正确: {n_correct} ({n_correct/n_total*100:.1f}%)")
        print(f"  错误: {n_total-n_correct} ({(n_total-n_correct)/n_total*100:.1f}%)")
        print(f"  不确定: {n_uncertain} ({n_uncertain/n_total*100:.1f}%)")
        
        # 显示该意图下的问题
        print(f"\n示例问题:")
        for i, q in enumerate(intent_questions[:5], 1):
            print(f"  [{i}] {q.query}")
            print(f"      预测: {q.predicted_agent} | Margin: {q.margin:.4f} | {'✓' if q.is_correct else '✗'}")
    
    def compare_with_new_questions(self, new_questions: List[CoreQuestion]):
        """与新的核心问题列表对比"""
        print("\n" + "="*80)
        print("核心问题对比分析")
        print("="*80)
        
        # 计算overlap
        old_queries = {q.query for q in self.core_questions}
        new_queries = {q.query for q in new_questions}
        
        overlap = old_queries & new_queries
        only_old = old_queries - new_queries
        only_new = new_queries - old_queries
        
        print(f"\n重叠分析:")
        print(f"  当前核心问题数: {len(old_queries)}")
        print(f"  新的核心问题数: {len(new_queries)}")
        print(f"  重叠问题数: {len(overlap)}")
        print(f"  仅在当前: {len(only_old)}")
        print(f"  仅在新列表: {len(only_new)}")
        print(f"  重叠率: {len(overlap)/len(old_queries)*100:.1f}%")
    
    def verify_selection_quality(self):
        """验证核心问题选择质量"""
        print("\n" + "="*80)
        print("核心问题质量验证")
        print("="*80)
        
        checks = []
        
        # 检查1: 是否有足够的不确定性
        n_uncertain = sum(1 for q in self.core_questions if q.margin < 0.1)
        uncertain_ratio = n_uncertain / len(self.core_questions)
        check1 = uncertain_ratio >= 0.1  # 至少10%不确定
        checks.append(("不确定性比例", f"{uncertain_ratio*100:.1f}% (>= 10%)", check1))
        
        # 检查2: 是否有足够的错误案例
        n_errors = sum(1 for q in self.core_questions if not q.is_correct)
        error_ratio = n_errors / len(self.core_questions)
        check2 = error_ratio >= 0.2  # 至少20%错误
        checks.append(("错误案例比例", f"{error_ratio*100:.1f}% (>= 20%)", check2))
        
        # 检查3: Margin分布是否合理
        margins = [q.margin for q in self.core_questions]
        margin_mean = np.mean(margins)
        check3 = 0.05 <= margin_mean <= 0.3  # 均值在合理范围
        checks.append(("Margin均值", f"{margin_mean:.4f} (0.05-0.3)", check3))
        
        # 检查4: 意图类别覆盖
        intent_count = len(set(q.expected_intent for q in self.core_questions))
        check4 = intent_count >= 8  # 至少覆盖8个意图类别
        checks.append(("意图类别覆盖", f"{intent_count}/12 (>= 8)", check4))
        
        # 打印结果
        print("\n质量检查:")
        for name, value, passed in checks:
            status = "✓ 通过" if passed else "✗ 未通过"
            print(f"  {name}: {value} - {status}")
        
        overall = all(check[2] for check in checks)
        print(f"\n总体评估: {'✓ 质量良好' if overall else '⚠ 需要优化'}")
        
        return overall
    
    def export_for_evaluation(self) -> Tuple[List[str], List[str], List[bool]]:
        """
        导出用于评估的数据格式
        
        Returns:
            (queries, intents, is_boundary)
        """
        queries = [q.query for q in self.core_questions]
        intents = [q.expected_intent for q in self.core_questions]
        is_boundary = [False] * len(self.core_questions)
        
        print(f"\n导出完成: {len(queries)} 条数据")
        return queries, intents, is_boundary
    
    def show_sample_with_details(self, n: int = 5, filter_type: str = 'all'):
        """
        显示带详细信息的示例
        
        Args:
            n: 显示数量
            filter_type: 'all', 'uncertain', 'errors', 'correct'
        """
        if filter_type == 'uncertain':
            questions = [q for q in self.core_questions if q.margin < 0.1]
        elif filter_type == 'errors':
            questions = [q for q in self.core_questions if not q.is_correct]
        elif filter_type == 'correct':
            questions = [q for q in self.core_questions if q.is_correct]
        else:
            questions = self.core_questions
        
        questions = questions[:n]
        
        print("\n" + "="*80)
        print(f"核心问题详细示例 ({filter_type}, n={len(questions)})")
        print("="*80)
        
        for i, q in enumerate(questions, 1):
            print(f"\n{'─'*80}")
            print(f"问题 [{i}]: {q.query}")
            print(f"{'─'*80}")
            print(f"预期意图: {q.expected_intent}")
            print(f"预期Agent: {q.expected_agent}")
            print(f"预测Agent: {q.predicted_agent}")
            print(f"分类结果: {'✓ 正确' if q.is_correct else '✗ 错误'}")
            print(f"\n概率分布:")
            print(f"  Top-1: {q.top1_prob:.4f}")
            print(f"  Top-2: {q.top2_prob:.4f}")
            print(f"  Margin: {q.margin:.4f} {'⚠ 不确定' if q.margin < 0.1 else '✓ 确定'}")
            print(f"\n完整概率分布:")
            sorted_probs = sorted(q.all_probs.items(), key=lambda x: x[1], reverse=True)
            for agent, prob in sorted_probs:
                markers = []
                if agent == q.predicted_agent:
                    markers.append("[预测]")
                if agent == q.expected_agent:
                    markers.append("[预期]")
                marker_str = " ".join(markers)
                print(f"  {agent:20s}: {prob:.4f} {marker_str}")


def analyze_core_questions(filepath: str = None):
    """
    便捷函数：分析核心问题文件
    
    Args:
        filepath: 核心问题文件路径，None则自动查找最新的
    """
    import glob
    
    # 如果没有指定路径，查找最新的
    if filepath is None:
        result_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        files = glob.glob(os.path.join(result_dir, 'core_questions_*.json'))
        if not files:
            print("未找到核心问题文件！")
            return None
        filepath = max(files, key=os.path.getctime)
        print(f"自动选择最新文件: {filepath}")
    
    # 创建分析器
    analyzer = CoreQuestionAnalyzer(filepath)
    
    # 显示各种分析
    analyzer.show_summary()
    analyzer.analyze_margin_distribution()
    analyzer.show_uncertain_questions(n=10)
    analyzer.show_error_questions(n=10)
    analyzer.analyze_agent_confusion()
    analyzer.verify_selection_quality()
    
    return analyzer


if __name__ == "__main__":
    import sys
    
    # 如果提供了命令行参数，使用指定文件
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = None
    
    analyze_core_questions(filepath)
