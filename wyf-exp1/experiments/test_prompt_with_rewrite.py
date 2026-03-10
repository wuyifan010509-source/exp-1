"""
测试 gen_10_no_oos.txt prompt 在 GOLDEN_TEST.csv 上的准确率
带查询改写模块：将问题改写成完整句子后再分类
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import requests
from datetime import datetime
from collections import defaultdict

# 配置
BACKBONE_API_URL = "http://172.17.160.46:8080/v1"
BACKBONE_MODEL = "Qwen2.5-32B-Instruct"
GOLDEN_TEST_PATH = "/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/data/GOLDEN_TEST.csv"
PROMPT_PATH = "/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/data/baselines/greedy.txt"

# 意图类别
INTENT_CLASSES = [
    "选股类", "诊股类", "预测类", "知识库类", "新闻类",
    "通用类", "推荐类", "策略类", "指标查询类", "身份类",
    "分时图类", "K线图类"
]

# 查询改写Prompt（用户可以修改这里）
REWRITE_PROMPT = """
# 角色设定
你是一个专业的金融问答“意图清洗与改写专家”。用户的原始提问往往非常简略、口语化或语义模糊。你的任务是根据预设的【5条核心映射规则】，识别用户提问的隐性诉求，并将其改写为意图明确、主谓宾齐全的标准金融提问，以便下游系统进行处理。
**注意，只有当用户问题模糊（如只有一个宾语，没有谓语)时，才触发改写，否则必须保持愿意！
# 核心映射与改写规则

【规则 1：诊股类 - 纯标的映射】
- 触发条件：输入内容只能是一个股票名称、股票代码（无其他疑问词或动词，句子只能有股票名称本身）。如“300803”，不要有其他主语、谓语。**如果包含"后期"、”下一步"、"未来"则属于预测类**
- 改写动作：将其改写为请求系统给出该股票综合评价的标准句式。
- 改写模板：“请对 [股票名称/代码] 这只股票进行综合评价、基本面和技术面的诊断。”

【规则 2：选股类 - 纯板块映射】
- 触发条件：输入内容纯粹是一个板块名称或概念名称。如“计算机板块”，"因用股票",不要有其他主语、谓语
- 改写动作：将其改写为请求系统列出该板块成分股的标准句式。
- 改写模板：“请按条件筛选并列出 [板块/概念名称] 板块下的所有成分股名单。”

【规则 3：预测类 - 时空穿梭映射】
- 触发条件：问题中带有“往后”、“未来”、“明天”、“下一步”、“前景”等时间推演词汇，且提及了具体搭股票。**记住:如果未提及具体股票那就不是预测类，如"明天买什么","明天哪个核聚变股”好是推荐类**
- 改写动作：突出预见性判断的诉求，将其改写为对未来的预测请求。
- 改写模板：“请预测和推演 [原问题标的] [明天/未来/下一步] 的前景。”

【规则 4：选股类 - 裸条件映射】
- 触发条件：纯粹提供了一个或多个选股条件（如技术形态“一阳穿三线的”、数值门槛“股价大于300”、市场行为“今天底位首板股票”、”xxx持仓的股票“），无具体股票主语。**必须包含的是客观条件，如“赚钱的“、"推荐的"这类主观的不行**
- 改写动作：将其改写为明确的结构化数据筛选指令。
- 改写模板：“请按客观条件筛选出满足 [原问题中的所有客观条件] 的股票列表。”

【规则 5：策略类 - 策略查询】
- 触发条件:问题中包含"xx策略是什么"这种句式
- 改写模块:"请处理xx投资策略的解析"

【规则 6：指标查询类 - 指标查询】
- 触发条件:问题中包含"xxxx(股票名称)的yy(指标名称)"，如"300803的概念板块是什么",这种句式。注意"趋势"这种词不是指标，不属于指标查询类
- 改写模块:"请问查询xxxx(股票名称)的yy（指标名称）这一指标"

【规则 7：知识库类 - 概念询问】
- 触发条件:如果问题只有某某概念，如"大盘量能","价值决策"
- 改写模块:"什么是[提及的概念]"

【规则 8：知识库类 - 软件操作】
- 触发条件:提及软件私有知识，如"擒龙平台"、"黄金坑"
- 改写模块: 告诉我以下私有知识:[原问题]

# 处理要求
1. 阅读用户的原始输入，匹配最符合的一条规则。注意：必须要严格符合要求才能改写，**如果都不符合，则必须严格保持原意不要进行任何改写**，不强制套用模板。
2. 根据匹配到的规则改写问题，补全缺失的金融语境。
3. 严格输出为 JSON 格式，包含 `matched_rule`（匹配的规则编号/名称）和 `rewritten_query`（改写后的标准问题）。不允许输出任何其他解释性文字。

# 示例参考
原始输入：华西股份
输出：
{{
  "matched_rule": "规则 1：诊股类 - 纯标的映射",
  "rewritten_query": "请对华西股份这只股票进行综合评价、基本面和技术面的诊断。"
}}

原始输入：一阳穿三线的
输出：
{{
  "matched_rule": "规则 4：选股类 - 裸条件映射",
  "rewritten_query": "请按客观条件筛选出满足一阳穿三线形态的股票列表。"
}}

原始输入：航空发展往后怎么走
输出：
{{
  "matched_rule": "规则 3：预测类 - 时空穿梭映射",
  "rewritten_query": "请预测和推演航空发展往后的走势、表现或前景。"
}}

# 当前用户输入
{query}

"""


def load_prompt():
    """加载分类prompt文件"""
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def load_test_data():
    """加载测试数据,去除oos"""
    df = pd.read_csv(GOLDEN_TEST_PATH)
    df = df[df['预期意图'] != 'oos']  # 去除oos
    
    queries = df['问题'].tolist()
    intents = df['预期意图'].tolist()
    is_boundary = df['是否处于意图边界'].fillna('否').map(lambda x: str(x).strip() == '是').tolist()
    
    return queries, intents, is_boundary


def rewrite_query(api_url, model, query):
    """
    使用LLM将查询改写成完整句子，并解析JSON格式的输出
    
    Args:
        api_url: API地址
        model: 模型名称
        query: 原始查询
        
    Returns:
        rewritten_query: 改写后的完整句子
        matched_rule: 匹配的规则
        raw_response: 原始响应（用于调试）
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy"
    }
    
    # 构建改写prompt
    rewrite_prompt = REWRITE_PROMPT.format(query=query)
    
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": rewrite_prompt}
                ],
                "temperature": 0.0,  # 使用确定性输出
                "max_tokens": 300
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"].strip()
        
        # 解析JSON输出
        try:
            # 提取JSON部分（处理可能的多余文本）
            json_str = raw_response
            if '```json' in raw_response:
                # 提取代码块中的JSON
                json_str = raw_response.split('```json')[1].split('```')[0].strip()
            elif '```' in raw_response:
                json_str = raw_response.split('```')[1].split('```')[0].strip()
            
            parsed = json.loads(json_str)
            rewritten = parsed.get('rewritten_query', query)
            matched_rule = parsed.get('matched_rule', '未匹配规则')
            
            return rewritten, matched_rule, raw_response
            
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON解析失败: {e}")
            print(f"[Warning] 原始响应: {raw_response[:100]}...")
            # JSON解析失败时返回原始查询
            return query, "JSON解析失败", raw_response
        
    except Exception as e:
        print(f"[Error] Rewrite API call failed: {e}")
        # 改写失败时返回原始查询
        return query, "API调用失败", str(e)


def classify_query(api_url, model, system_prompt, query):
    """使用prompt对单个query进行分类"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy"
    }
    
    full_user_prompt = f"""{system_prompt}

用户问题：{query}

请直接输出意图类别（只输出类别名称，如"选股类"、"诊股类"等，不要解释）："""
    
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": full_user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 100
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"].strip()
        
        # 提取意图
        predicted = extract_intent(raw_response)
        
        return predicted, raw_response
        
    except Exception as e:
        print(f"[Error] Classify API call failed: {e}")
        return "", str(e)


def extract_intent(text):
    """从响应中提取意图类别"""
    text = text.strip()
    
    for intent in INTENT_CLASSES:
        if intent in text:
            return intent
    
    # 尝试匹配简写形式
    for intent in INTENT_CLASSES:
        short_name = intent.replace("类", "")
        if short_name in text:
            return intent
    
    return text[:20] if len(text) > 20 else text


def calculate_metrics(all_predictions, all_labels):
    """计算 Precision, Recall, F1 和混淆矩阵"""
    # 构建混淆矩阵
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for pred, true in zip(all_predictions, all_labels):
        confusion_matrix[true][pred] += 1
    
    # 计算每个类别的指标
    metrics_per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for intent in INTENT_CLASSES:
        tp = confusion_matrix[intent][intent]  # True Positive
        fp = sum(confusion_matrix[other][intent] for other in INTENT_CLASSES if other != intent)  # False Positive
        fn = sum(confusion_matrix[intent][other] for other in INTENT_CLASSES if other != intent)  # False Negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        support = tp + fn  # 该类的样本数
        
        metrics_per_class[intent] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support
        }
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # 计算宏平均
    macro_precision = sum(m["precision"] for m in metrics_per_class.values()) / len(INTENT_CLASSES)
    macro_recall = sum(m["recall"] for m in metrics_per_class.values()) / len(INTENT_CLASSES)
    macro_f1 = sum(m["f1"] for m in metrics_per_class.values()) / len(INTENT_CLASSES)
    
    # 计算加权平均
    total_samples = sum(m["support"] for m in metrics_per_class.values())
    weighted_precision = sum(m["precision"] * m["support"] for m in metrics_per_class.values()) / total_samples if total_samples > 0 else 0
    weighted_recall = sum(m["recall"] * m["support"] for m in metrics_per_class.values()) / total_samples if total_samples > 0 else 0
    weighted_f1 = sum(m["f1"] * m["support"] for m in metrics_per_class.values()) / total_samples if total_samples > 0 else 0
    
    # 转换混淆矩阵为普通dict
    cm_dict = {}
    for true_intent in INTENT_CLASSES:
        cm_dict[true_intent] = {}
        for pred_intent in INTENT_CLASSES:
            cm_dict[true_intent][pred_intent] = confusion_matrix[true_intent][pred_intent]
    
    return {
        "per_class": metrics_per_class,
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4)
        },
        "weighted_avg": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1": round(weighted_f1, 4)
        },
        "confusion_matrix": cm_dict
    }


def evaluate_prompt():
    """评估prompt（带查询改写）"""
    print("=" * 80)
    print("测试 gen_10_no_oos.txt prompt (带查询改写)")
    print("=" * 80)
    print("\n[改写Prompt]")
    print(REWRITE_PROMPT)
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1/4] 加载测试数据...")
    queries, intents, is_boundary = load_test_data()
    print(f"  测试样本: {len(queries)}条 (已去除oos)")
    print(f"  边界样本: {sum(is_boundary)}条")
    
    # 2. 加载prompt
    print("\n[2/4] 加载分类prompt...")
    system_prompt = load_prompt()
    print(f"  Prompt长度: {len(system_prompt)}字符")
    
    # 3. 评估
    print("\n[3/4] 开始评估（先改写，再分类）...")
    print("=" * 80)
    
    correct = 0
    boundary_correct = 0
    total = len(queries)
    boundary_total = sum(is_boundary)
    
    results = []
    all_predictions = []
    all_labels = []
    
    for i, (query, true_intent, is_bound) in enumerate(zip(queries, intents, is_boundary)):
        print(f"\n{'='*80}")
        print(f"题目 {i+1}/{total}")
        print(f"{'='*80}")
        
        # 步骤1：改写查询
        print(f"[改写前] {query}")
        rewritten_query, matched_rule, rewrite_raw = rewrite_query(BACKBONE_API_URL, BACKBONE_MODEL, query)
        print(f"[匹配规则] {matched_rule}")
        print(f"[改写后] {rewritten_query}")
        
        # 步骤2：对改写后的查询进行分类
        predicted, classify_raw = classify_query(BACKBONE_API_URL, BACKBONE_MODEL, system_prompt, rewritten_query)
        
        is_correct = (predicted == true_intent)
        if is_correct:
            correct += 1
            if is_bound:
                boundary_correct += 1
        
        results.append({
            'index': i + 1,
            'original_query': query,
            'rewritten_query': rewritten_query,
            'matched_rule': matched_rule,
            'expected': true_intent,
            'predicted': predicted,
            'correct': is_correct,
            'is_boundary': is_bound,
            'rewrite_raw': rewrite_raw[:200] if len(rewrite_raw) > 200 else rewrite_raw,
            'classify_raw': classify_raw[:100] if len(classify_raw) > 100 else classify_raw
        })
        
        all_predictions.append(predicted)
        all_labels.append(true_intent)
        
        # 实时输出结果
        status = "✓ 正确" if is_correct else "✗ 错误"
        boundary_mark = " [边界]" if is_bound else ""
        print(f"\n结果: {status}{boundary_mark}")
        print(f"期望: {true_intent} -> 预测: {predicted}")
        if not is_correct:
            print(f"分类原始输出: {classify_raw[:80]}{'...' if len(classify_raw) > 80 else ''}")
    
    # 统计结果
    print(f"\n{'='*80}")
    print("评估结果汇总")
    print(f"{'='*80}")
    
    accuracy = correct / total if total > 0 else 0
    boundary_accuracy = boundary_correct / boundary_total if boundary_total > 0 else 0
    
    # 计算详细指标
    detailed_metrics = calculate_metrics(all_predictions, all_labels)
    
    print(f"\n总体准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"边界准确率: {boundary_accuracy:.2%} ({boundary_correct}/{boundary_total})")
    
    # 输出 Macro Average
    print(f"\n[Macro Average]")
    print(f"  Precision: {detailed_metrics['macro_avg']['precision']:.4f}")
    print(f"  Recall:    {detailed_metrics['macro_avg']['recall']:.4f}")
    print(f"  F1-Score:  {detailed_metrics['macro_avg']['f1']:.4f}")
    
    # 输出 Weighted Average
    print(f"\n[Weighted Average]")
    print(f"  Precision: {detailed_metrics['weighted_avg']['precision']:.4f}")
    print(f"  Recall:    {detailed_metrics['weighted_avg']['recall']:.4f}")
    print(f"  F1-Score:  {detailed_metrics['weighted_avg']['f1']:.4f}")
    
    # 统计各类别表现
    print("\n各类别详细指标:")
    print(f"{'类别':<12} {'Support':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 55)
    
    intent_stats = {}
    for intent in INTENT_CLASSES:
        metrics = detailed_metrics['per_class'][intent]
        if metrics['support'] > 0:
            intent_stats[intent] = metrics
            print(f"{intent:<12} {metrics['support']:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    # 输出混淆矩阵（简化版 - 只显示主要混淆）
    print("\n混淆矩阵 - 主要错误分类:")
    cm = detailed_metrics['confusion_matrix']
    confusion_pairs = []
    for true_intent in cm:
        for pred_intent in cm[true_intent]:
            if true_intent != pred_intent and cm[true_intent][pred_intent] > 0:
                confusion_pairs.append((true_intent, pred_intent, cm[true_intent][pred_intent]))
    
    # 按混淆数量排序
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"{'真实类别':<12} -> {'预测类别':<12} {'数量':<6}")
    print("-" * 35)
    for true_intent, pred_intent, count in confusion_pairs[:15]:
        print(f"{true_intent:<12} -> {pred_intent:<12} {count:<6}")
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"gen_10_with_rewrite_{timestamp}.json"
    
    output_data = {
        'prompt_file': PROMPT_PATH,
        'test_file': GOLDEN_TEST_PATH,
        'rewrite_prompt': REWRITE_PROMPT,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'boundary_samples': boundary_total,
        'boundary_correct': boundary_correct,
        'boundary_accuracy': boundary_accuracy,
        'macro_avg': detailed_metrics['macro_avg'],
        'weighted_avg': detailed_metrics['weighted_avg'],
        'per_intent_stats': intent_stats,
        'confusion_matrix': detailed_metrics['confusion_matrix'],
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存: {output_file}")
    
    return results


if __name__ == "__main__":
    evaluate_prompt()
