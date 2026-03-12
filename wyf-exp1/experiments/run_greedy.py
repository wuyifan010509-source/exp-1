"""
贪心算法实验 - 迭代优化智能体画像
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime

from config import (
    BACKBONE_API_URL, RESULTS_DIR, GOLDEN_TEST_PATH, HISTORICAL_LOGS_PATH
)
from structured_profile import ProfileSet
from evaluation.classifier import QwenClassifier, MockClassifier
from evaluation import EvaluationPipeline
from evaluation.core_question_integration import (
    setup_core_question_evaluation,
    CoreQuestionEvaluator
)
from whitebox_init import generate_initial_population
from evolution import LLMMutator
from greedy import GreedyOptimizer

# 实验配置
MAX_ITERATIONS = 100
CANDIDATES_PER_SLOT = 1  # 每槽位只生成1个候选
PATIENCE = 10
TRAIN_SIZE = 100  # 每轮固定100条数据


def run_greedy_experiment(use_mock: bool = False, allowed_slots: list = None):
    """
    运行贪心算法实验
    
    Args:
        use_mock: 是否使用Mock分类器
        allowed_slots: 允许优化的槽位列表，如 ['C','B','R'], ['B','R'] 等
    """
    print("=" * 80)
    print("贪心算法优化实验（高效版本）")
    print(f"配置: 最大{MAX_ITERATIONS}轮, 每槽位{CANDIDATES_PER_SLOT}候选, 耐心值{PATIENCE}")
    print(f"优化策略: 每轮只优化最差的3个槽位，每槽位只生成1个候选")
    print(f"评估策略: 改完3个槽位后统一评估1次（大幅减少评估开销）")
    print(f"数据策略: 每轮固定100条数据，不同轮次使用不同数据")
    print(f"滑动窗口: 最近3轮平均作为基线，改进需超阈值0.001")
    print(f"预估每轮: ~3次LLM调用 + 1次评估")
    print("=" * 80)
    
    # 调试：检查当前目录和文件
    print(f"\n[Debug] Current directory: {os.getcwd()}")
    print(f"[Debug] Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 1. 初始化分类器
    if use_mock:
        print("\n[1/4] 使用Mock分类器")
        classifier = MockClassifier(seed=42)
    else:
        print(f"\n[1/4] 连接到GPU模型: {BACKBONE_API_URL}")
        classifier = QwenClassifier(api_url=BACKBONE_API_URL)
        print("[1/4] 连接成功")
    
    # 2. 白盒初始化 - 只生成1个初始解（使用置信度低的样本优化B槽位）
    print(f"\n[2/4] 白盒初始化 - 生成初始解（使用低置信度样本）")
    initial_population = generate_initial_population(
        pop_size=1, 
        classifier=classifier,
        sample_size=200
    )
    
    # 检查是否生成成功
    if not initial_population:
        print("[Error] 初始种群生成失败！请检查:")
        print("  1. data/agents/tools_descriptions.json 文件是否存在")
        print("  2. 文件内容格式是否正确")
        return None
    
    initial_profile = initial_population[0]
    print(f"[2/4] 初始解生成完成，共{len(initial_profile.profiles)}个智能体")
    
    # 3. 创建评估流水线
    print("\n[3/4] 创建评估流水线")
    pipeline = EvaluationPipeline(classifier)
    
    # 4. 创建贪心优化器和LLM变异器
    print("\n[4/4] 开始贪心优化")
    print(f"[4/4] 评估策略: 每轮固定100条数据，该轮所有候选使用相同数据")
    print(f"[4/4] 不同轮次使用不同的100条数据，避免过拟合")
    print(f"[4/4] 滑动窗口: 最近3轮平均作为基线，改进需超阈值0.001")
    print(f"[4/4] 候选策略: 每槽位只生成1个候选（LLM调用最少化）")
    print(f"[4/4] 评估策略: 改完3个槽位后统一评估1次（而非3次）")
    # 【槽位生成策略配置】
    # allowed_slots 参数控制允许生成的槽位组合：
    #   - ['C', 'B', 'R'] : 同时生成 C(核心能力)、B(处理边界)、R(拒绝范围) - 全优化
    #   - ['B', 'R']      : 只生成 B 和 R（默认，C固定不变）
    #   - ['C', 'B']      : 只生成 C 和 B
    #   - ['C', 'R']      : 只生成 C 和 R
    #   - ['C']           : 只优化 C（核心能力）
    #   - ['B']           : 只优化 B（处理边界）
    #   - ['R']           : 只优化 R（拒绝范围）
    if allowed_slots is None:
        allowed_slots = ['B', 'R']  # 默认：不优化C，只优化B和R
    
    print(f"\n[Config] 槽位生成策略: {allowed_slots}")
    slot_name_map = {'C': '核心能力', 'B': '处理边界', 'R': '拒绝范围'}
    for slot in allowed_slots:
        print(f"  - {slot}: {slot_name_map[slot]}")
    
    optimizer = GreedyOptimizer(
        max_iterations=MAX_ITERATIONS,
        candidates_per_slot=CANDIDATES_PER_SLOT,
        patience=PATIENCE,
        slots_per_iteration=2,  # 每轮只优化2个槽位
        window_size=3,  # 滑动窗口大小
        improvement_threshold=0.001,  # 改进阈值
        allowed_slots=allowed_slots  # 槽位生成策略
    )
    
    # 加载训练数据用于获取正例
    import pandas as pd
    train_df = pd.read_csv('data/HISTORICAL_LOGS.csv')
    
    # 创建获取正例的函数
    def get_positive_examples(agent_name, n=3):
        """获取指定智能体的正例查询"""
        target_intents = [intent for intent, name in pipeline.intent_to_agent.items() if name == agent_name]
        
        if not target_intents:
            return []
        
        mask = train_df['预期意图'].isin(target_intents)
        positive_queries = train_df[mask]['问题'].dropna().tolist()
        
        if len(positive_queries) > n:
            import random
            return random.sample(positive_queries, n)
        return positive_queries
    
    mutator = LLMMutator(positive_examples_func=get_positive_examples)
    
    # 【核心问题评估】使用从历史数据中提取的核心问题替代黄金测试集
    print("\n" + "="*80)
    print("初始化核心问题评估集")
    print("="*80)
    evaluator = setup_core_question_evaluation(
        classifier=classifier,
        profiles=initial_profile,
        force_regenerate=True,   # 强制重新生成（使用新的logprobs逻辑）
        total_size=500           # 核心问题总数
    )
    
    # 获取核心问题统计
    stats = evaluator.get_stats()
    print(f"\n[CoreQuestion] 评估集统计:")
    print(f"  总数: {stats.get('total', 0)}")
    print(f"  不确定: {stats.get('uncertain', 0)} (Margin < 0.1)")
    print(f"  错误: {stats.get('errors', 0)} (用于挖掘改进空间)")
    
    # 【评估函数】使用核心问题计算准确率
    def fitness_func(profile_set: ProfileSet, test_data=None) -> float:
        """使用核心问题评估准确率"""
        try:
            # 使用核心问题评估（替代黄金测试集）
            queries, intents, is_boundary = evaluator.get_evaluation_data()
            result = pipeline.evaluate(profile_set, test_data=(queries, intents, is_boundary))
            
            fitness = result['fitness']
            print(f"    Fitness: {fitness:.4f} (Acc: {result['accuracy']:.2%}, Boundary Acc: {result.get('boundary_accuracy', 0):.2%})")
            return fitness
        except Exception as e:
            print(f"    [Error] {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    # 【错误案例获取函数】使用固定的核心问题集（500条）
    core_queries, core_intents, _ = evaluator.get_evaluation_data()
    print(f"[Setup] 加载核心问题集用于错误挖掘: {len(core_queries)} 条")
    
    def get_bad_cases_func(profile_set: ProfileSet, sample_size: int = None) -> list:
        """使用固定的核心问题集返回分类结果（用于挖掘错误案例）"""
        try:
            # 使用固定的核心问题集（500条），不再采样
            queries = core_queries
            intents = core_intents
            
            print(f"[Bad Cases] 使用核心问题集 {len(queries)} 条（固定）")
            
            # 分类这些样本
            results = pipeline.classifier.classify_batch(queries, profile_set)
            
            all_cases = []
            for intent, result, query in zip(intents, results, queries):
                expected = pipeline.intent_to_agent.get(intent)
                is_correct = (result.predicted_agent == expected)
                
                case = {
                    'query': query,
                    'expected_agent': expected,
                    'predicted_agent': result.predicted_agent,
                    'margin': result.get_margin(),
                    'is_correct': is_correct
                }
                all_cases.append(case)
            
            # 统计错误案例数
            error_count = sum(1 for c in all_cases if not c['is_correct'])
            print(f"[Bad Cases] 发现 {error_count} 个错误案例")
            
            return all_cases
        except Exception as e:
            print(f"    [Warning] 获取分类结果失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # 运行贪心优化
    print(f"\n开始贪心优化 (最大{MAX_ITERATIONS}轮)...")
    print("-" * 80)
    
    try:
        result = optimizer.optimize(
            initial_profile_set=initial_profile,
            fitness_func=fitness_func,
            llm_mutator=mutator,
            save_dir=os.path.join(RESULTS_DIR, "greedy_checkpoints"),
            get_bad_cases_func=get_bad_cases_func
        )
        
        # 输出结果
        print("\n" + "=" * 80)
        print("贪心优化完成！")
        print("=" * 80)
        
        print(f"\n[结果摘要]")
        print(f"总迭代次数: {result.total_iterations}")
        print(f"总评估次数: {result.total_evaluations}")
        print(f"初始适应度: {result.fitness_history[0]:.4f}")
        print(f"最终适应度: {result.best_fitness:.4f}")
        print(f"提升幅度: {result.best_fitness - result.fitness_history[0]:.4f}")
        
        if len(result.improvement_history) > 0:
            print(f"\n[改进历史]")
            for i, imp in enumerate(result.improvement_history, 1):
                print(f"  {i}. 第{imp['eval_num']}次评估: {imp['agent']}的{imp['slot']}改进")
        
        print(f"\n[适应度曲线]")
        for i, fitness in enumerate(result.fitness_history):
            if i == 0:
                print(f"  初始: {fitness:.4f}")
            else:
                print(f"  第{i}次改进后: {fitness:.4f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"greedy_{timestamp}.json")
        
        save_data = {
            "config": {
                "max_iterations": MAX_ITERATIONS,
                "candidates_per_slot": CANDIDATES_PER_SLOT,
                "patience": PATIENCE,
                "train_size": TRAIN_SIZE
            },
            "optimization": {
                "total_iterations": result.total_iterations,
                "total_evaluations": result.total_evaluations,
                "fitness_history": result.fitness_history,
                "initial_fitness": result.fitness_history[0],
                "final_fitness": result.best_fitness,
                "improvement": result.best_fitness - result.fitness_history[0]
            },
            "improvement_history": result.improvement_history,
            "best_profile": result.best_profile_set.to_dict()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[保存] 结果已保存到: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"\n[Error] 优化过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='贪心算法优化实验')
    parser.add_argument('--mock', action='store_true', help='使用Mock分类器')
    parser.add_argument('--slots', type=str, default='BR', 
                       help='槽位生成策略: CBR(全优化), BR(默认,不优化C), CB, CR, C, B, R')
    
    args = parser.parse_args()
    
    # 解析槽位参数
    slot_map = {
        'CBR': ['C', 'B', 'R'],
        'BR': ['B', 'R'],
        'CB': ['C', 'B'],
        'CR': ['C', 'R'],
        'C': ['C'],
        'B': ['B'],
        'R': ['R']
    }
    allowed_slots = slot_map.get(args.slots.upper(), ['B', 'R'])
    
    result_file = run_greedy_experiment(use_mock=args.mock, allowed_slots=allowed_slots)
    
    if result_file:
        print("\n✓ 实验完成！")
        print(f"\n接下来可以在GOLDEN_TEST.csv上测试:")
        print(f"  python -m experiments.final_test --result {result_file}")
    else:
        print("\n✗ 实验失败")


if __name__ == "__main__":
    main()
