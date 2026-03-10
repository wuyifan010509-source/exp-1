"""
完整MVE实验 - 5代 x 5个体 x 50条样本
带显式Few-shot优化
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime

from config import (
    BACKBONE_API_URL, RESULTS_DIR
)
from structured_profile import ProfileSet
from evaluation.classifier import QwenClassifier, MockClassifier
from evaluation import EvaluationPipeline
from whitebox_init import generate_initial_population
from evolution import GeneticAlgorithm, LLMMutator

# 实验配置
POP_SIZE = 5
N_GENERATIONS = 10
TRAIN_SIZE = 50


def run_full_experiment(use_mock: bool = False):
    """
    运行完整实验（5代 x 5个体 x 50条）
    """
    print("=" * 80)
    print("完整MVE实验")
    print(f"配置: {N_GENERATIONS}代 x {POP_SIZE}个体 x {TRAIN_SIZE}条训练")
    print("=" * 80)
    
    # 1. 初始化
    if use_mock:
        print("\n[1/4] 使用Mock分类器")
        classifier = MockClassifier(seed=42)
    else:
        print(f"\n[1/4] 连接到GPU模型: {BACKBONE_API_URL}")
        classifier = QwenClassifier(api_url=BACKBONE_API_URL)
        print("[1/4] 连接成功")
    
    # 2. 白盒初始化
    print(f"\n[2/4] 白盒初始化 - 生成{POP_SIZE}个初始个体")
    initial_population = generate_initial_population(pop_size=POP_SIZE)
    print(f"[2/4] 成功生成{len(initial_population)}个个体")
    
    # 3. 创建评估流水线
    print("\n[3/4] 创建评估流水线")
    pipeline = EvaluationPipeline(classifier)
    
    # 4. 创建GA组件
    print("\n[4/4] 开始遗传演化")
    ga = GeneticAlgorithm(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        crossover_rate=0.7,
        mutation_rate=0.8,  # 50%变异率
        elite_count=2,
        tournament_k=3
    )
    
    # 加载训练数据用于获取正例
    import pandas as pd
    train_df = pd.read_csv('data/HISTORICAL_LOGS.csv')
    
    # 创建获取正例的函数
    def get_positive_examples(agent_name, n=3):
        """获取指定智能体的正例查询"""
        # 从训练集中找预期是该智能体的查询
        # CSV列名是中文: '问题' 和 '预期意图'
        target_intents = [intent for intent, name in pipeline.intent_to_agent.items() if name == agent_name]
        
        if not target_intents:
            return []
        
        # 筛选出这些意图的查询
        mask = train_df['预期意图'].isin(target_intents)
        positive_queries = train_df[mask]['问题'].dropna().tolist()
        
        # 随机采样n个
        if len(positive_queries) > n:
            import random
            return random.sample(positive_queries, n)
        return positive_queries
    
    mutator = LLMMutator(positive_examples_func=get_positive_examples)
    
    # 适应度函数
    def fitness_func(profile_set: ProfileSet) -> float:
        try:
            result = pipeline.evaluate(profile_set, sample_size=TRAIN_SIZE)
            fitness = result['fitness']
            print(f"    Fitness: {fitness:.4f} (Acc: {result['accuracy']:.2%})")
            return fitness
        except Exception as e:
            print(f"    [Error] {e}")
            return 0.0
    
    # Bad cases获取函数
    def get_bad_cases_func(profile_set: ProfileSet) -> list:
        try:
            queries, intents, _ = pipeline.load_test_data(sample_size=TRAIN_SIZE)
            results = pipeline.classifier.classify_batch(queries, profile_set)
            
            bad_cases = []
            for intent, result, query in zip(intents, results, queries):
                expected = pipeline.intent_to_agent.get(intent)
                if result.predicted_agent != expected:
                    bad_cases.append({
                        'query': query,
                        'expected_agent': expected,
                        'predicted_agent': result.predicted_agent,
                        'margin': result.get_margin()
                    })
            
            bad_cases.sort(key=lambda x: x['margin'])
            return bad_cases[:10]
        except Exception as e:
            print(f"    [Warning] 获取bad cases失败: {e}")
            return []
    
    # 运行演化
    print(f"\n开始演化 ({N_GENERATIONS}代 x {POP_SIZE}个体 x {TRAIN_SIZE}条)...")
    print("-" * 80)
    
    try:
        result = ga.evolve(
            initial_population=initial_population,
            fitness_func=fitness_func,
            mutator=mutator,
            save_dir=RESULTS_DIR,
            get_bad_cases_func=get_bad_cases_func
        )
        
        # 输出结果
        print("\n" + "=" * 80)
        print("演化完成！")
        print("=" * 80)
        
        print(f"\n[结果摘要]")
        print(f"总代数: {result.total_generations}")
        print(f"初始适应度: {result.best_fitness_history[0]:.4f}")
        print(f"最终适应度: {result.best_fitness_history[-1]:.4f}")
        print(f"提升幅度: {result.best_fitness_history[-1] - result.best_fitness_history[0]:.4f}")
        
        print(f"\n[演化曲线]")
        for i, (best, avg) in enumerate(zip(result.best_fitness_history, result.avg_fitness_history), 1):
            print(f"  Gen {i}: Best={best:.4f}, Avg={avg:.4f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"mve_5x5x50_{timestamp}.json")
        
        save_data = {
            "config": {
                "pop_size": POP_SIZE,
                "generations": N_GENERATIONS,
                "train_size": TRAIN_SIZE,
                "mutation_rate": 0.5
            },
            "evolution": {
                "best_fitness_history": result.best_fitness_history,
                "avg_fitness_history": result.avg_fitness_history,
                "final_best": result.best_fitness_history[-1],
                "final_avg": result.avg_fitness_history[-1]
            },
            "best_profile": result.best_individual.profile_set.to_dict()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[保存] 结果已保存到: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"\n[Error] 演化过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='完整MVE实验 (5x5x50)')
    parser.add_argument('--mock', action='store_true', help='使用Mock分类器')
    
    args = parser.parse_args()
    
    result_file = run_full_experiment(use_mock=args.mock)
    
    if result_file:
        print("\n✓ 实验完成！")
        print(f"\n接下来可以在GOLDEN_TEST.csv上测试:")
        print(f"  python -m experiments.final_test --result {result_file}")
    else:
        print("\n✗ 实验失败")


if __name__ == "__main__":
    main()
