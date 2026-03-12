"""
核心问题评估集成使用示例

这个文件展示了如何在run_greedy.py中使用新的核心问题评估功能
替代原有的黄金测试集
"""


# ==================== 使用方式1：简单替换（推荐） ====================

def example_simple_replace():
    """
    简单替换示例：在run_greedy.py中，只需要修改fitness_func即可
    """
    from evaluation.core_question_integration import setup_core_question_evaluation
    
    # ... 原有的初始化代码 ...
    # classifier = QwenClassifier(api_url=BACKBONE_API_URL)
    # pipeline = EvaluationPipeline(classifier)
    
    # 新增：设置核心问题评估
    evaluator = setup_core_question_evaluation(
        classifier=classifier,
        profiles=initial_profile,  # 初始画像
        force_regenerate=False,     # 第一次运行后设为False，使用已有核心问题
        total_size=500              # 核心问题总数
    )
    
    # 替代原有的fitness_func
    fitness_func = evaluator.create_fitness_function(pipeline)
    
    # ... 原有的优化代码 ...
    # optimizer.optimize(...)


# ==================== 使用方式2：手动控制生成 ====================

def example_manual_control():
    """
    手动控制生成：更灵活的控制核心问题的生成和加载
    """
    from evaluation.core_question_integration import CoreQuestionEvaluator
    from evaluation.core_question_analyzer import CoreQuestionAnalyzer
    
    # ... 原有的初始化代码 ...
    
    # 创建评估器
    evaluator = CoreQuestionEvaluator(classifier)
    
    # 第一次运行：生成核心问题
    core_file = evaluator.generate_core_questions(
        profiles=initial_profile,
        total_size=500,
        margin_threshold=0.1,
        sample_size=None,  # 使用全部历史数据
        force_regenerate=True
    )
    print(f"核心问题已保存到: {core_file}")
    
    # 分析核心问题质量
    analyzer = evaluator.analyze_current_core_questions()
    
    # 打印示例问题
    evaluator.print_sample_questions(n=5)
    
    # 创建适应度函数
    fitness_func = evaluator.create_fitness_function(pipeline)
    
    # ... 后续优化 ...


# ==================== 使用方式3：使用现有文件（推荐用于多次运行） ====================

def example_use_existing():
    """
    使用已存在的核心问题文件（多次运行时使用，避免重复生成）
    """
    from evaluation.core_question_integration import (
        CoreQuestionEvaluator, 
        create_fitness_function_with_core_questions
    )
    
    # ... 原有的初始化代码 ...
    
    # 方式A：自动查找最新的核心问题文件
    evaluator = CoreQuestionEvaluator(classifier)
    if not evaluator.core_questions:
        # 如果没有找到，生成新的
        evaluator.generate_core_questions(
            profiles=initial_profile,
            total_size=500,
            force_regenerate=True
        )
    
    fitness_func = evaluator.create_fitness_function(pipeline)
    
    # 方式B：指定具体文件路径
    # core_file = "results/core_questions_20260311_105242.pkl"
    # evaluator = CoreQuestionEvaluator(classifier, core_file)
    # fitness_func = evaluator.create_fitness_function(pipeline)
    
    # 方式C：便捷函数一键创建
    # fitness_func = create_fitness_function_with_core_questions(
    #     classifier=classifier,
    #     pipeline=pipeline,
    #     profiles=initial_profile,
    #     force_regenerate=False  # 使用已有文件
    # )


# ==================== 使用方式4：查看和分析核心问题 ====================

def example_analyze_questions():
    """
    查看和分析已生成的核心问题
    """
    from evaluation.core_question_analyzer import analyze_core_questions
    
    # 自动分析最新的核心问题文件
    analyzer = analyze_core_questions()
    
    # 或指定文件分析
    # analyzer = analyze_core_questions("results/core_questions_20260311_105242.json")
    
    # 查看特定意图类别的问题
    analyzer.analyze_by_intent("选股类")
    
    # 查看不确定问题
    analyzer.show_uncertain_questions(n=10)
    
    # 查看错误问题
    analyzer.show_error_questions(n=10)
    
    # 导出评估数据
    queries, intents, is_boundary = analyzer.export_for_evaluation()


# ==================== 在run_greedy.py中的具体修改方案 ====================

"""
修改步骤：

1. 在文件顶部添加导入（约第15行）
   
   from evaluation.core_question_integration import (
       setup_core_question_evaluation,
       CoreQuestionEvaluator
   )


2. 替换原有的黄金测试集加载（约第117-121行）
   
   # 原有的代码：
   # golden_df = pd.read_csv(GOLDEN_TEST_PATH)
   # golden_queries = golden_df['问题'].tolist()
   # golden_intents = golden_df['预期意图'].tolist()
   # golden_is_boundary = golden_df['是否处于意图边界'].fillna('否').map(lambda x: str(x).strip() == '是').tolist()
   
   # 新代码：
   evaluator = setup_core_question_evaluation(
       classifier=classifier,
       profiles=initial_profile,
       force_regenerate=False,  # 第一次设为True，之后设为False
       total_size=500
   )


3. 替换fitness_func（约第132-145行）
   
   # 原有的代码：
   # def fitness_func(profile_set: ProfileSet, test_data=None) -> float:
   #     result = pipeline.evaluate(profile_set, test_data=(golden_queries, golden_intents, golden_is_boundary))
   #     ...
   
   # 新代码：
   fitness_func = evaluator.create_fitness_function(pipeline)


4. （可选）在优化结束后分析核心问题
   
   在main函数末尾添加：
   
   # 分析核心问题质量
   print("\n" + "="*80)
   print("核心问题分析")
   print("="*80)
   evaluator.analyze_current_core_questions()
   evaluator.print_sample_questions(n=5)


就是这样！只需4步，就可以将黄金测试集替换为核心问题评估集。
"""


# ==================== 运行示例 ====================

if __name__ == "__main__":
    print("="*80)
    print("核心问题评估集成使用示例")
    print("="*80)
    
    print("\n这个文件展示了如何在run_greedy.py中使用核心问题评估功能")
    print("\n主要步骤:")
    print("1. 导入: from evaluation.core_question_integration import setup_core_question_evaluation")
    print("2. 初始化: evaluator = setup_core_question_evaluation(classifier, profiles, force_regenerate=True)")
    print("3. 使用: fitness_func = evaluator.create_fitness_function(pipeline)")
    print("4. （可选）分析: evaluator.analyze_current_core_questions()")
    
    print("\n" + "="*80)
    print("命令行查看核心问题")
    print("="*80)
    print("\n分析最新的核心问题文件:")
    print("  python -m evaluation.core_question_analyzer")
    
    print("\n生成新的核心问题（测试用）:")
    print("  python -m evaluation.core_question_selector")
    
    print("\n测试集成模块:")
    print("  python -m evaluation.core_question_integration")
    
    print("\n" + "="*80)
    print("完成！请查看上面的修改方案来集成到run_greedy.py")
    print("="*80)
