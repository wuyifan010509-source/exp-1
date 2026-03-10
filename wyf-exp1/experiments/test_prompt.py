"""
测试 gen_10_no_oos.txt prompt 在 GOLDEN_TEST.csv 上的准确率
输出每个题目的对错情况，包含 precision, recall, f1 和混淆矩阵
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
PROMPT_PATH = "/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/data/baselines/gen_10.txt"

# 意图类别
INTENT_CLASSES = [
    "选股类", "诊股类", "预测类", "知识库类", "新闻类",
    "通用类", "推荐类", "策略类", "指标查询类", "身份类",
    "分时图类", "K线图类"
]


def load_prompt():
    """加载prompt文件"""
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
        print(f"[Error] API call failed: {e}")
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
    """评估prompt"""
    print("=" * 80)
    print("测试 gen_10_no_oos.txt prompt")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1/3] 加载测试数据...")
    queries, intents, is_boundary = load_test_data()
    print(f"  测试样本: {len(queries)}条 (已去除oos)")
    print(f"  边界样本: {sum(is_boundary)}条")
    
    # 2. 加载prompt
    print("\n[2/3] 加载prompt...")
    system_prompt = load_prompt()
    print(f"  Prompt长度: {len(system_prompt)}字符")
    
    # 3. 评估
    print("\n[3/3] 开始评估...")
    print("=" * 80)
    
    correct = 0
    boundary_correct = 0
    total = len(queries)
    boundary_total = sum(is_boundary)
    
    results = []
    all_predictions = []
    all_labels = []
    
    for i, (query, true_intent, is_bound) in enumerate(zip(queries, intents, is_boundary)):
        predicted, raw = classify_query(BACKBONE_API_URL, BACKBONE_MODEL, system_prompt, query)
        
        is_correct = (predicted == true_intent)
        if is_correct:
            correct += 1
            if is_bound:
                boundary_correct += 1
        
        results.append({
            'index': i + 1,
            'query': query,
            'expected': true_intent,
            'predicted': predicted,
            'correct': is_correct,
            'is_boundary': is_bound,
            'raw_response': raw[:100] if len(raw) > 100 else raw
        })
        
        all_predictions.append(predicted)
        all_labels.append(true_intent)
        
        # 实时输出每个题目的结果
        status = "✓" if is_correct else "✗"
        boundary_mark = "[边界]" if is_bound else ""
        print(f"{i+1:3d}. {status} {boundary_mark}")
        print(f"     问题: {query[:50]}{'...' if len(query) > 50 else ''}")
        print(f"     期望: {true_intent} -> 预测: {predicted}")
        if not is_correct:
            print(f"     原始输出: {raw[:80]}{'...' if len(raw) > 80 else ''}")
        print()
    
    # 统计结果
    accuracy = correct / total if total > 0 else 0
    boundary_accuracy = boundary_correct / boundary_total if boundary_total > 0 else 0
    
    # 计算详细指标
    detailed_metrics = calculate_metrics(all_predictions, all_labels)
    
    print("=" * 80)
    print("评估结果汇总")
    print("=" * 80)
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
    output_file = f"gen_10_{timestamp}.json"
    
    output_data = {
        'prompt_file': PROMPT_PATH,
        'test_file': GOLDEN_TEST_PATH,
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
