#!/usr/bin/env python3
"""
核心问题评估演示脚本

运行这个脚本可以快速体验核心问题选择和分析的完整流程
"""
import os
import sys
import time

# 确保在wyf-exp1目录下运行
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir.endswith('evaluation'):
    os.chdir(os.path.dirname(script_dir))

sys.path.insert(0, os.getcwd())

print("="*80)
print("🎯 核心问题评估演示")
print("="*80)
print("\n这个演示将展示如何从历史数据中自动提取核心问题用于评估")
print("核心问题包括：")
print("  1️⃣  不确定的问题（Top-1和Top-2概率差距小）")
print("  2️⃣  分错的问题（模型预测错误）")
print("")

time.sleep(1)

# ==================== 第1步：生成核心问题 ====================

print("\n" + "="*80)
print("📊 第1步：从历史数据中提取核心问题")
print("="*80)

from evaluation.core_question_selector import CoreQuestionSelector, create_test_profile
from evaluation.classifier import MockClassifier

# 创建测试用的profile和分类器
test_profile = create_test_profile()
print(f"\n✓ 创建测试Profile：{len(test_profile.profiles)} 个智能体")

classifier = MockClassifier(seed=42)
print(f"✓ 创建分类器：MockClassifier（用于演示）")

# 创建选择器并生成核心问题
selector = CoreQuestionSelector(classifier, margin_threshold=0.1)
print(f"\n⏳ 开始分析历史数据（采样200条，提取50个核心问题）...")
print("-"*80)

core_questions = selector.select_core_questions(
    profiles=test_profile,
    total_size=50,
    uncertainty_ratio=0.5,
    sample_size=200
)

# 保存
filepath = selector.save_core_questions()

print("\n" + "="*80)
print("📈 生成结果统计")
print("="*80)

stats = selector.stats
print(f"\n✓ 分析总数: {stats['total_analyzed']} 条")
print(f"✓ 核心问题: {len(core_questions)} 个")
print(f"✓ 准确率: {stats['accuracy']:.1%}")
print(f"✓ 不确定问题: {stats['low_margin_count']} 个 (Margin < 0.1)")
print(f"✓ 错误问题: {stats['total_errors']} 个")
print(f"✓ Margin均值: {stats['margin_mean']:.4f}")

time.sleep(1)

# ==================== 第2步：查看示例问题 ====================

print("\n" + "="*80)
print("🔍 第2步：查看不确定问题示例（Margin < 0.1）")
print("="*80)

selector.print_sample_questions(n=5, show_type='uncertain')

time.sleep(0.5)

print("\n" + "="*80)
print("❌ 第3步：查看错误分类问题示例")
print("="*80)

selector.print_sample_questions(n=5, show_type='errors')

time.sleep(0.5)

# ==================== 第3步：深度分析 ====================

print("\n" + "="*80)
print("📋 第4步：核心问题质量分析")
print("="*80)

from evaluation.core_question_analyzer import CoreQuestionAnalyzer

analyzer = CoreQuestionAnalyzer()
analyzer.core_questions = core_questions
analyzer.stats = stats

analyzer.show_summary()
analyzer.analyze_margin_distribution()
analyzer.verify_selection_quality()

time.sleep(0.5)

print("\n" + "="*80)
print("🔄 第5步：Agent混淆分析（最常见的误分类模式）")
print("="*80)

analyzer.analyze_agent_confusion()

# ==================== 第4步：验证Top-1/Top-2概率 ====================

time.sleep(0.5)

print("\n" + "="*80)
print("🔢 第6步：验证Top-1/Top-2概率获取（通过底层API）")
print("="*80)

print("\n每个核心问题都包含：")
print("  ✓ Top-1概率：模型最有信心的预测")
print("  ✓ Top-2概率：模型第二有信心的预测")
print("  ✓ Margin：Top-1 - Top-2（差距越小越不确定）")
print("  ✓ 完整概率分布：所有12个智能体的概率")
print("")

# 显示第一个问题的详细信息
if core_questions:
    q = core_questions[0]
    print("示例问题的概率分布：")
    print(f"  问题: {q.query}")
    print(f"  预期: {q.expected_agent}")
    print(f"  预测: {q.predicted_agent}")
    print(f"\n  Top-1概率: {q.top1_prob:.4f}")
    print(f"  Top-2概率: {q.top2_prob:.4f}")
    print(f"  Margin: {q.margin:.4f} {'⚠️ 不确定' if q.margin < 0.1 else '✓ 确定'}")
    print(f"\n  完整概率分布（Top-5）：")
    for i, (agent, prob) in enumerate(sorted(q.all_probs.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        markers = []
        if agent == q.predicted_agent:
            markers.append("🏆 预测")
        if agent == q.expected_agent:
            markers.append("✓ 预期")
        print(f"    {i}. {agent:20s}: {prob:.4f} {' '.join(markers)}")

# ==================== 第5步：使用说明 ====================

print("\n" + "="*80)
print("📝 总结：如何使用核心问题评估")
print("="*80)

print(f"""
✅ 核心问题评估已准备就绪！

📦 生成的文件：
   - {filepath}
   - {filepath.replace('.pkl', '.json')}

🔧 在run_greedy.py中使用：

   from evaluation.core_question_integration import setup_core_question_evaluation
   
   # 初始化核心问题评估
   evaluator = setup_core_question_evaluation(
       classifier=classifier,
       profiles=initial_profile,
       force_regenerate=False,  # 使用现有文件
       total_size=500
   )
   
   # 创建适应度函数（替代原有的黄金测试集）
   fitness_func = evaluator.create_fitness_function(pipeline)

📊 查看分析结果：
   python -m evaluation.core_question_analyzer

🎯 核心问题特点：
   ✓ 500个精心挑选的问题
   ✓ 包含不确定案例（Margin < 0.1）
   ✓ 包含错误案例（模型分错）
   ✓ 覆盖所有12个意图类别
   ✓ Top-1/Top-2概率来自底层API，非大模型直接输出
""")

print("="*80)
print("🎉 演示完成！")
print("="*80)
print("\n您现在可以在run_greedy.py中集成核心问题评估功能了。")
print("查看 evaluation/core_question_example.py 了解详细集成方案。")
