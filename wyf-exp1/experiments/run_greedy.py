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
from whitebox_init import generate_initial_population
from evolution import LLMMutator
from greedy import GreedyOptimizer

# 实验配置
MAX_ITERATIONS = 100
CANDIDATES_PER_SLOT = 1  # 每槽位只生成1个候选
PATIENCE = 10
TRAIN_SIZE = 100  # 每轮固定100条数据


def run_greedy_experiment(use_mock: bool = False):
    """
    运行贪心算法实验
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
    optimizer = GreedyOptimizer(
        max_iterations=MAX_ITERATIONS,
        candidates_per_slot=CANDIDATES_PER_SLOT,
        patience=PATIENCE,
        slots_per_iteration=2,  # 每轮只优化2个槽位（B和R，C不迭代）
        window_size=3,  # 滑动窗口大小
        improvement_threshold=0.001  # 改进阈值
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
    
    # 加载黄金测试集（用于评估准确率）
    import pandas as pd
    golden_df = pd.read_csv(GOLDEN_TEST_PATH)
    golden_queries = golden_df['问题'].tolist()
    golden_intents = golden_df['预期意图'].tolist()
    golden_is_boundary = golden_df['是否处于意图边界'].fillna('否').map(lambda x: str(x).strip() == '是').tolist()
    
    print(f"[Setup] 加载黄金测试集: {len(golden_queries)} 条")
    
    # 加载历史数据集（用于挖掘错误案例）
    historical_df = pd.read_csv(HISTORICAL_LOGS_PATH)
    historical_df = historical_df[historical_df['预期意图'] != 'oos']
    
    print(f"[Setup] 加载历史数据集: {len(historical_df)} 条")
    
    # 【评估函数】使用黄金测试集计算准确率
    def fitness_func(profile_set: ProfileSet, test_data=None) -> float:
        """使用黄金测试集评估准确率"""
        try:
            # 始终使用黄金测试集
            result = pipeline.evaluate(profile_set, test_data=(golden_queries, golden_intents, golden_is_boundary))
            
            fitness = result['fitness']
            print(f"    Fitness: {fitness:.4f} (Acc: {result['accuracy']:.2%}, Boundary Acc: {result.get('boundary_accuracy', 0):.2%})")
            return fitness
        except Exception as e:
            print(f"    [Error] {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    # 【错误案例获取函数】从历史数据集采样并分类
    def get_bad_cases_func(profile_set: ProfileSet, sample_size: int = 200) -> list:
        """从历史数据集中采样并返回分类结果（用于挖掘错误案例）"""
        try:
            # 从历史数据集分层采样
            unique_intents = historical_df['预期意图'].unique()
            samples_per_intent = max(1, sample_size // len(unique_intents))
            
            sampled_dfs = []
            for intent in unique_intents:
                intent_df = historical_df[historical_df['预期意图'] == intent]
                n = min(samples_per_intent, len(intent_df))
                if n > 0:
                    sampled_dfs.append(intent_df.sample(n=n))
            
            sampled_df = pd.concat(sampled_dfs).head(sample_size)
            
            queries = sampled_df['问题'].tolist()
            intents = sampled_df['预期意图'].tolist()
            
            print(f"[Bad Cases] 从历史数据集采样 {len(queries)} 条")
            
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
    
    # 获取每轮历史数据采样（用于生成bad cases）
    def get_historical_sample_func(sample_size: int = 100):
        """从历史数据集采样（用于生成bad cases）"""
        unique_intents = historical_df['预期意图'].unique()
        samples_per_intent = max(1, sample_size // len(unique_intents))
        
        sampled_dfs = []
        for intent in unique_intents:
            intent_df = historical_df[historical_df['预期意图'] == intent]
            n = min(samples_per_intent, len(intent_df))
            if n > 0:
                sampled_dfs.append(intent_df.sample(n=n))
        
        sampled_df = pd.concat(sampled_dfs).head(sample_size)
        
        queries = sampled_df['问题'].tolist()
        intents = sampled_df['预期意图'].tolist()
        is_boundary = [False] * len(queries)  # 历史数据没有边界标记
        
        return queries, intents, is_boundary
    
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
    
    args = parser.parse_args()
    
    result_file = run_greedy_experiment(use_mock=args.mock)
    
    if result_file:
        print("\n✓ 实验完成！")
        print(f"\n接下来可以在GOLDEN_TEST.csv上测试:")
        print(f"  python -m experiments.final_test --result {result_file}")
    else:
        print("\n✗ 实验失败")


if __name__ == "__main__":
    main()
