"""
测试 gen_10_no_oos.txt prompt 在 GOLDEN_TEST.csv 上的准确率
输出每个题目的对错情况，包含 precision, recall, f1 和混淆矩阵
支持并发测试以提高速度
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import datetime

# 设置中文字体 - 优先使用文泉驿微米黑
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 配置
BACKBONE_API_URL = "http://172.17.160.42:8080/v1"
BACKBONE_MODEL = "Qwen2.5-32B-Instruct"
# BACKBONE_API_URL="http://127.0.0.1:3002/v1"
# BACKBONE_MODEL = "/Data1/wz_workspace/Qwen2.5-32B-Instruct"
GOLDEN_TEST_PATH = "/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/data/GOLDEN_TEST.csv"
PROMPT_PATH = "/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/data/baselines/intent_gpt.txt"

# 意图类别（中英文对照）- 12类
INTENT_CLASSES = [
    "Stock Selection", "Stock Diagnosis", "Forecast", "Knowledge Base", "News",
    "General Chat", "Recommendation", "Strategy", "Indicator Query",
    "K-line", "Time-sharing", "Identity"
]

# 中文类别名称（用于图表显示）
INTENT_CLASSES_CN = [
    "选股", "诊股", "预测", "知识库", "新闻",
    "通用聊天", "推荐", "策略", "指标查询",
    "K线图", "分时图", "身份"
]

# 英文到中文的映射
EN_TO_CN = {en: cn for en, cn in zip(INTENT_CLASSES, INTENT_CLASSES_CN)}


def load_prompt():
    """加载prompt文件"""
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def load_test_data():
    """加载测试数据，将中文意图映射为英文"""
    # 中文到英文的意图映射
    CHINESE_TO_ENGLISH = {
        "选股类": "Stock Selection",
        "诊股类": "Stock Diagnosis",
        "预测类": "Forecast",
        "知识库类": "Knowledge Base",
        "新闻类": "News",
        "通用类": "General Chat",
        "推荐类": "Recommendation",
        "策略类": "Strategy",
        "指标查询类": "Indicator Query",
        "K线图类": "K-line",
        "分时图类": "Time-sharing",
        "身份类": "Identity",
    }
    
    df = pd.read_csv(GOLDEN_TEST_PATH)
    df = df[df['预期意图'] != 'oos']  # 去除oos
    
    queries = df['问题'].tolist()
    # 将中文意图转换为英文
    intents = [CHINESE_TO_ENGLISH.get(intent, intent) for intent in df['预期意图'].tolist()]
    is_boundary = df['是否处于意图边界'].fillna('否').map(lambda x: str(x).strip() == '是').tolist()
    
    return queries, intents, is_boundary


def classify_query_single(args):
    """单个query分类（用于并发）"""
    idx, query, true_intent, is_bound, api_url, model, system_prompt = args
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy"
    }

    context = """
    用户输入股票名称、股票代码，是在请求系统给出该股票综合评价
    用户输入板块名称，是在请求系统列出该板块成分股
    用户问题带有“往后”、“未来”、“明天”、“下一步”、“前景”等词汇，且提及了具体的股票。是在对该股票未来的预测请求。如果没有具体股票则肯定不是预测类！
    用户只输入选股条件，是在请求筛选该条件的股票
    用户提及“大盘量能“、”价值决策”，是在询问这个概念的含义
    用户提及“擒龙平台”、“黄金坑”，是在询问软件操作知识
    用户提及“xx策略”，要转为策略类。
    用户提及“xx(股票）的yy(指标名称）”，是在查询这指标

    """
    
    full_user_prompt = f"""
    
{system_prompt}



User Query: {query}

Please output the intent class name directly (only output the class name, e.g., "Stock Selection", "Stock Diagnosis", etc., no explanation):

Intent Class:"""
    
    try:
        import math
        start_time = time.time()
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": full_user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 20,
                "logprobs": True,
                "top_logprobs": 20
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        raw_response = choice["message"]["content"].strip()
        end_time = time.time()
        
        latency = end_time - start_time
        predicted = extract_intent(raw_response)
        
        # 计算 margin (Top1 - Top2 概率差距)
        margin = 0.0
        top1_prob = 0.0
        top2_prob = 0.0
        all_token_probs = []
        
        logprobs_content = choice.get("logprobs", {}).get("content", [])
        if logprobs_content and len(logprobs_content) > 0:
            # 获取第一个token的top logprobs
            first_token_data = logprobs_content[0]
            first_token_logprobs = first_token_data.get("top_logprobs", [])
            
            if first_token_logprobs:
                # 直接取 top 2 的 token 概率
                sorted_tokens = sorted(first_token_logprobs, 
                                     key=lambda x: x.get("logprob", -float('inf')), 
                                     reverse=True)
                
                if len(sorted_tokens) >= 1:
                    top1_prob = math.exp(sorted_tokens[0].get("logprob", -float('inf')))
                if len(sorted_tokens) >= 2:
                    top2_prob = math.exp(sorted_tokens[1].get("logprob", -float('inf')))
                
                margin = top1_prob - top2_prob
                
                # 保存所有 token 概率用于调试
                all_token_probs = [
                    {"token": t.get("token", ""), "prob": round(math.exp(t.get("logprob", -float('inf'))), 4)}
                    for t in sorted_tokens[:5]
                ]
        
        # 调试：如果 margin 为 0，打印 logprobs 信息
        if margin == 0.0 and idx == 0:
            print(f"[Debug] First query logprobs check:")
            print(f"  Raw response: {raw_response}")
            print(f"  Logprobs available: {bool(logprobs_content)}")
            if logprobs_content:
                print(f"  First token top_logprobs: {logprobs_content[0].get('top_logprobs', [])[:3]}")
        
        return {
            'index': idx + 1,
            'query': query,
            'expected': true_intent,
            'predicted': predicted,
            'raw_response': raw_response,
            'latency': latency,
            'margin': round(margin, 4),
            'top1_prob': round(top1_prob, 4),
            'top2_prob': round(top2_prob, 4),
            'token_probs': all_token_probs,
            'success': True
        }
        
    except Exception as e:
        return {
            'index': idx + 1,
            'query': query,
            'expected': true_intent,
            'predicted': "",
            'raw_response': str(e),
            'latency': 0.0,
            'success': False
        }


def classify_queries_concurrent(queries, intents, is_boundary, api_url, model, system_prompt, max_workers=10):
    """
    并发分类所有query
    
    Args:
        max_workers: 并发线程数（默认10）
    """
    results = [None] * len(queries)
    
    # 准备参数
    args_list = [
        (i, queries[i], intents[i], is_boundary[i], api_url, model, system_prompt)
        for i in range(len(queries))
    ]
    
    print(f"\n[Concurrent] Starting concurrent classification with {max_workers} workers...")
    print(f"[Concurrent] Total queries: {len(queries)}")
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(classify_query_single, args): args[0] 
            for args in args_list
        }
        
        # 收集结果
        for future in as_completed(future_to_idx):
            result = future.result()
            idx = result['index'] - 1
            results[idx] = result
            
            completed += 1
            if completed % 10 == 0 or completed == len(queries):
                print(f"[Progress] {completed}/{len(queries)} completed ({completed/len(queries)*100:.1f}%)")
    
    return results


def extract_intent(text):
    """从响应中提取意图类别"""
    text = text.strip()
    
    # 直接匹配英文类别
    for intent in INTENT_CLASSES:
        if intent in text:
            return intent
    
    # 中文到英文的映射（如果模型输出中文）
    CHINESE_TO_ENGLISH = {
        "选股类": "Stock Selection",
        "诊股类": "Stock Diagnosis",
        "预测类": "Forecast",
        "知识库类": "Knowledge Base",
        "新闻类": "News",
        "通用类": "General Chat",
        "推荐类": "Recommendation",
        "策略类": "Strategy",
        "指标查询类": "Indicator Query",
        "K线图类": "K-line",
        "分时图类": "Time-sharing",
        "身份类": "Identity",
    }
    
    # 尝试直接匹配中文类别
    for cn_intent, en_intent in CHINESE_TO_ENGLISH.items():
        if cn_intent in text:
            return en_intent
    
    # 尝试匹配简写形式
    chinese_mapping = {
        "Stock Selection": ["选股"],
        "Stock Diagnosis": ["诊股"],
        "Forecast": ["预测"],
        "Knowledge Base": ["知识库"],
        "News": ["新闻"],
        "General Chat": ["通用"],
        "Recommendation": ["推荐"],
        "Strategy": ["策略"],
        "Indicator Query": ["指标查询", "指标"],
        "K-line": ["K线图", "K线"],
        "Time-sharing": ["分时图", "分时"],
        "Identity": ["身份"],
    }
    
    for intent, chinese_names in chinese_mapping.items():
        for cn_name in chinese_names:
            if cn_name in text:
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


def evaluate_prompt(max_workers=10):
    """
    Evaluate prompt with concurrent requests
    
    Args:
        max_workers: Number of concurrent threads (default: 10)
    """
    print("=" * 80)
    print("Testing Prompt on GOLDEN_TEST (Concurrent Mode)")
    print("=" * 80)
    print(f"[Config] Concurrent workers: {max_workers}")
    
    # 1. 加载数据
    print("\n[1/3] Loading test data...")
    queries, intents, is_boundary = load_test_data()
    print(f"  Test samples: {len(queries)} (oos removed)")
    print(f"  Boundary samples: {sum(is_boundary)}")
    
    # 2. 加载prompt
    print("\n[2/3] Loading prompt...")
    system_prompt = load_prompt()
    print(f"  Prompt length: {len(system_prompt)} chars")
    
    # 3. 并发评估
    print("\n[3/3] Starting concurrent evaluation...")
    print("=" * 80)
    
    start_time = time.time()
    
    # 使用并发分类所有query
    raw_results = classify_queries_concurrent(
        queries, intents, is_boundary,
        BACKBONE_API_URL, BACKBONE_MODEL, system_prompt,
        max_workers=max_workers
    )
    
    total_elapsed = time.time() - start_time
    
    # 处理结果
    correct = 0
    boundary_correct = 0
    total = len(queries)
    boundary_total = sum(is_boundary)
    
    results = []
    all_predictions = []
    all_labels = []
    latency_list = []
    margin_list = []
    
    print("\n[Results] Processing results...")
    for result in raw_results:
        idx = result['index']
        predicted = result['predicted']
        true_intent = result['expected']
        is_bound = is_boundary[idx - 1]
        query = result['query']
        raw = result['raw_response']
        latency = result['latency']
        
        is_correct = (predicted == true_intent)
        if is_correct:
            correct += 1
            if is_bound:
                boundary_correct += 1
        
        # 更新结果字典
        result['correct'] = is_correct
        result['is_boundary'] = is_bound
        result['raw_response'] = raw[:100] if len(raw) > 100 else raw
        result['latency'] = round(latency, 3)
        
        results.append(result)
        all_predictions.append(predicted)
        all_labels.append(true_intent)
        latency_list.append(latency)
        margin_list.append(result.get('margin', 0.0))
        
        # 实时输出（可选，只打印错误的）
        if not is_correct:
            status = "✗"
            boundary_mark = "[Boundary]" if is_bound else ""
            print(f"{idx:3d}. {status} {boundary_mark}")
            print(f"     Query: {query[:50]}{'...' if len(query) > 50 else ''}")
            print(f"     Expected: {true_intent} -> Predicted: {predicted}")
            print(f"     Raw output: {raw[:80]}{'...' if len(raw) > 80 else ''}")
    
    print(f"\n[Speed] Total time: {total_elapsed:.1f}s ({total/total_elapsed:.1f} queries/sec)")
    
    # 统计结果
    accuracy = correct / total if total > 0 else 0
    boundary_accuracy = boundary_correct / boundary_total if boundary_total > 0 else 0
    
    # 计算详细指标
    detailed_metrics = calculate_metrics(all_predictions, all_labels)
    
    # 计算推理延迟统计
    total_latency = sum(latency_list) if latency_list else 0
    avg_latency = total_latency / total if total > 0 else 0
    min_latency = min(latency_list) if latency_list else 0
    max_latency = max(latency_list) if latency_list else 0
    
    print("=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Boundary Accuracy: {boundary_accuracy:.2%} ({boundary_correct}/{boundary_total})")
    
    # 输出推理延迟统计
    print(f"\n[Inference Latency]")
    print(f"  Average: {avg_latency:.3f}s")
    print(f"  Min:     {min_latency:.3f}s")
    print(f"  Max:     {max_latency:.3f}s")
    print(f"  Total:   {total_latency:.3f}s (wall clock)")
    
    # 计算 margin 统计
    # 注意：margin=0 可能是因为 API 没有返回 logprobs，这种情况仍然要统计
    valid_margins = [m for m in margin_list if m is not None]
    non_zero_margins = [m for m in valid_margins if m > 0]
    
    if valid_margins:
        avg_margin = sum(valid_margins) / len(valid_margins)
        min_margin = min(valid_margins)
        max_margin = max(valid_margins)
        low_margin_count = sum(1 for m in valid_margins if m < 0.1)  # margin < 0.1 认为是低置信度
        
        print(f"\n[Margin Statistics (Top1 - Top2 Probability)]")
        print(f"  Average Margin: {avg_margin:.4f}")
        print(f"  Min Margin:     {min_margin:.4f}")
        print(f"  Max Margin:     {max_margin:.4f}")
        print(f"  Low Margin (<0.1): {low_margin_count} ({low_margin_count/len(valid_margins):.1%})")
        
        # 按 margin 分组统计准确率
        high_margin_correct = sum(1 for i, r in enumerate(results) 
                                  if r.get('margin', 0) >= 0.5 and r.get('correct', False))
        high_margin_total = sum(1 for r in results if r.get('margin', 0) >= 0.5)
        
        low_margin_correct = sum(1 for i, r in enumerate(results) 
                                 if r.get('margin', 0) < 0.5 and r.get('correct', False))
        low_margin_total = sum(1 for r in results if r.get('margin', 0) < 0.5)
        
        if high_margin_total > 0:
            print(f"\n[Accuracy by Margin]")
            print(f"  High Margin (≥0.5): {high_margin_correct}/{high_margin_total} ({high_margin_correct/high_margin_total:.2%})")
        if low_margin_total > 0:
            print(f"  Low Margin (<0.5):  {low_margin_correct}/{low_margin_total} ({low_margin_correct/low_margin_total:.2%})")
    
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
    print("\nPer-Class Detailed Metrics:")
    print(f"{'Class':<20} {'Support':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 65)
    
    intent_stats = {}
    for intent in INTENT_CLASSES:
        metrics = detailed_metrics['per_class'][intent]
        if metrics['support'] > 0:
            intent_stats[intent] = metrics
            print(f"{intent:<20} {metrics['support']:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    # 输出混淆矩阵（简化版 - 只显示主要混淆）
    print("\nConfusion Matrix - Top Misclassifications:")
    cm = detailed_metrics['confusion_matrix']
    confusion_pairs = []
    for true_intent in cm:
        for pred_intent in cm[true_intent]:
            if true_intent != pred_intent and cm[true_intent][pred_intent] > 0:
                confusion_pairs.append((true_intent, pred_intent, cm[true_intent][pred_intent]))
    
    # 按混淆数量排序
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"{'True Class':<20} -> {'Predicted Class':<20} {'Count':<6}")
    print("-" * 55)
    for true_intent, pred_intent, count in confusion_pairs[:15]:
        print(f"{true_intent:<20} -> {pred_intent:<20} {count:<6}")
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"gen_10_{timestamp}.json"
    
    # 准备 margin 统计
    margin_stats = {
        'total_samples': len(valid_margins),
        'non_zero_margin_samples': len(non_zero_margins),
        'logprobs_availability': f"{len(non_zero_margins)}/{len(valid_margins)} ({len(non_zero_margins)/len(valid_margins)*100:.1f}%)" if valid_margins else "0/0"
    }
    
    if non_zero_margins:
        margin_stats.update({
            'average': round(sum(non_zero_margins) / len(non_zero_margins), 4),
            'min': round(min(non_zero_margins), 4),
            'max': round(max(non_zero_margins), 4),
            'low_margin_count': sum(1 for m in non_zero_margins if m < 0.1),
            'low_margin_ratio': round(sum(1 for m in non_zero_margins if m < 0.1) / len(non_zero_margins), 4),
        })
    
    # 按 margin 分组的准确率统计（使用所有有 margin 的样本）  
    if valid_margins:
        margin_stats.update({
            'high_margin_accuracy': round(high_margin_correct / high_margin_total, 4) if high_margin_total > 0 else 0,
            'low_margin_accuracy': round(low_margin_correct / low_margin_total, 4) if low_margin_total > 0 else 0
        })
    
    output_data = {
        'prompt_file': PROMPT_PATH,
        'test_file': GOLDEN_TEST_PATH,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'boundary_samples': boundary_total,
        'boundary_correct': boundary_correct,
        'boundary_accuracy': boundary_accuracy,
        'latency_stats': {
            'average': round(avg_latency, 3),
            'min': round(min_latency, 3),
            'max': round(max_latency, 3),
            'total': round(total_latency, 3)
        },
        'margin_stats': margin_stats,
        'macro_avg': detailed_metrics['macro_avg'],
        'weighted_avg': detailed_metrics['weighted_avg'],
        'per_intent_stats': intent_stats,
        'confusion_matrix': detailed_metrics['confusion_matrix'],
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved: {output_file}")
    
    # Plot confusion matrix heatmap
    plot_confusion_heatmap(cm, output_file.replace('.json', '_confusion_matrix.png'))
    
    return results


def plot_confusion_heatmap(confusion_matrix, output_path):
    """Plot confusion matrix heatmap (absolute counts, diagonal masked)"""
    print("\nPlotting confusion matrix heatmap (absolute counts, diagonal masked)...")
    
    # Build confusion matrix array (raw counts)
    cm_array = np.zeros((len(INTENT_CLASSES), len(INTENT_CLASSES)))
    for i, true_intent in enumerate(INTENT_CLASSES):
        for j, pred_intent in enumerate(INTENT_CLASSES):
            cm_array[i, j] = confusion_matrix[true_intent][pred_intent]
    
    # Create mask for diagonal (hide diagonal color)
    mask = np.eye(len(INTENT_CLASSES), dtype=bool)
    
    # Calculate max value excluding diagonal for better color contrast
    cm_array_no_diag = cm_array.copy()
    np.fill_diagonal(cm_array_no_diag, 0)
    vmax_non_diag = cm_array_no_diag.max()
    
    # Create figure
    plt.figure(figsize=(14, 12))

    # Draw heatmap with diagonal masked (no color on diagonal)
    ax = sns.heatmap(cm_array,
                     mask=mask,  # Hide diagonal
                     annot=True,  # Show values
                     fmt='g',     # Integer format
                     cmap='Blues',  # Blue gradient
                     xticklabels=INTENT_CLASSES_CN,
                     yticklabels=INTENT_CLASSES_CN,
                     cbar_kws={'label': 'Count', 'shrink': 0.8},
                     square=True,
                     linewidths=0.5,  # Add grid lines
                     linecolor='white',
                     vmin=0,  # Set color scale minimum
                     vmax=vmax_non_diag,  # Use non-diagonal max for better contrast
                     annot_kws={'size': 18, 'weight': 'bold'})  # 非对角线数字字体大小和对角线一致

    # Add diagonal values manually (black text, no color)
    for i in range(len(INTENT_CLASSES)):
        value = int(cm_array[i, i])
        ax.text(i + 0.5, i + 0.5, str(value),
                ha='center', va='center',
                fontsize=18, fontweight='bold', color='black')

    # plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    # plt.ylabel('True Label', fontsize=18, fontweight='bold')
    # plt.title('Confusion Matrix',
    #           fontsize=20, fontweight='bold', pad=20)

    # Rotate labels to avoid overlap
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved: {output_path}")
    
    # Also save raw data as CSV
    csv_path = output_path.replace('.png', '_counts.csv')
    cm_df = pd.DataFrame(cm_array,
                         index=INTENT_CLASSES,
                         columns=INTENT_CLASSES)
    cm_df.to_csv(csv_path)
    print(f"Raw count data saved: {csv_path}")
    
    plt.close()


if __name__ == "__main__":
    evaluate_prompt(max_workers=1)
