"""
评估指标计算 - 完整版（含Precision/Recall/F1）
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    precision_score, recall_score, f1_score
)

from evaluation.classifier import ClassifyResult


def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    计算准确率
    """
    if not predictions or not labels or len(predictions) != len(labels):
        return 0.0
    
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)


def compute_boundary_accuracy(predictions: List[str], labels: List[str],
                               is_boundary_list: List[bool]) -> float:
    """计算边界样本的准确率"""
    boundary_preds = [p for p, is_b in zip(predictions, is_boundary_list) if is_b]
    boundary_labels = [l for l, is_b in zip(labels, is_boundary_list) if is_b]
    
    if not boundary_preds:
        return 0.0
    
    return compute_accuracy(boundary_preds, boundary_labels)


def compute_precision_recall_f1(predictions: List[str], labels: List[str],
                                average: str = 'weighted') -> Tuple[float, float, float]:
    """
    计算精确率、召回率、F1分数
    
    Returns:
        (precision, recall, f1)
    """
    if not predictions or not labels:
        return 0.0, 0.0, 0.0
    
    try:
        precision = float(precision_score(labels, predictions, average=average))
        recall = float(recall_score(labels, predictions, average=average))
        f1 = float(f1_score(labels, predictions, average=average))
        
        return precision, recall, f1
    except Exception as e:
        print(f"[Warning] Error computing precision/recall/f1: {e}")
        return 0.0, 0.0, 0.0


def compute_per_class_metrics(predictions: List[str], labels: List[str],
                              class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    计算每个类别的精确率、召回率、F1
    
    Returns:
        {类别名称: {'precision': x, 'recall': x, 'f1': x, 'support': x}}
    """
    if not predictions or not labels:
        return {}
    
    try:
        # 只计算存在的类别
        existing_classes = list(set(labels) | set(predictions))
        existing_classes = [c for c in existing_classes if c in class_names]
        
        if not existing_classes:
            return {}
        
        # 使用返回值解包
        result = precision_recall_fscore_support(
            labels, predictions, labels=existing_classes
        )
        
        # result是4个数组的元组
        precision_arr = result[0]
        recall_arr = result[1]
        f1_arr = result[2]
        support_arr = result[3]
        
        metrics = {}
        for i, class_name in enumerate(existing_classes):
            metrics[class_name] = {
                'precision': round(float(precision_arr[i]), 4),
                'recall': round(float(recall_arr[i]), 4),
                'f1': round(float(f1_arr[i]), 4),
                'support': int(support_arr[i])
            }
        return metrics
    except Exception as e:
        print(f"[Warning] Error computing per-class metrics: {e}")
        return {}


def compute_margins(results: List[ClassifyResult]) -> List[float]:
    """计算所有结果的Margin"""
    return [r.get_margin() for r in results]


def compute_average_margin(results: List[ClassifyResult]) -> float:
    """计算平均Margin"""
    margins = compute_margins(results)
    return float(np.mean(margins)) if margins else 0.0


def compute_average_margin_from_scores(predictions: List[str],
                                       confidence_scores: List[Dict[str, float]]) -> float:
    """
    从置信度分数计算平均Margin
    """
    margins = []
    for pred, scores in zip(predictions, confidence_scores):
        if scores and pred in scores:
            top1_score = scores[pred]
            other_scores = [s for k, s in scores.items() if k != pred]
            if other_scores:
                top2_score = max(other_scores)
                margins.append(top1_score - top2_score)
            else:
                margins.append(1.0)
        else:
            margins.append(0.0)
    
    return float(np.mean(margins)) if margins else 0.0


def compute_fitness(accuracy: float, avg_length: float, max_length: float = 200,
                   lambda_penalty: float = 100) -> float:
    """
    计算适应度函数
    F(P) = Accuracy - λ · max(0, avg_len - max_len)
    """
    length_penalty = lambda_penalty * max(0, avg_length - max_length)
    return accuracy - length_penalty


def compute_confusion_matrix(predictions: List[str], labels: List[str],
                            class_names: List[str]) -> np.ndarray:
    """计算混淆矩阵"""
    if not predictions or not labels:
        return np.zeros((len(class_names), len(class_names)))
    
    try:
        # 映射到索引
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        y_true_idx = [class_to_idx.get(y, -1) for y in labels]
        y_pred_idx = [class_to_idx.get(y, -1) for y in predictions]
        
        # 过滤无效值
        valid_pairs = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if t >= 0 and p >= 0]
        if not valid_pairs:
            return np.zeros((len(class_names), len(class_names)))
        
        y_true_idx, y_pred_idx = zip(*valid_pairs)
        
        return confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(class_names)))
    except Exception as e:
        print(f"[Warning] Error computing confusion matrix: {e}")
        return np.zeros((len(class_names), len(class_names)))


def get_bad_cases(predictions: List[str], labels: List[str], queries: List[str],
                 confidence_scores: Optional[List[Dict[str, float]]] = None,
                 top_k: int = 20) -> List[Dict[str, Any]]:
    """
    获取分类错误的案例
    """
    bad_cases: List[Dict[str, Any]] = []
    
    for i, (pred, label, query) in enumerate(zip(predictions, labels, queries)):
        if pred != label:
            case: Dict[str, Any] = {
                "query": query,
                "expected": label,
                "predicted": pred
            }
            
            if confidence_scores and i < len(confidence_scores):
                scores = confidence_scores[i]
                case["confidence"] = scores.get(pred, 0)
                # 计算margin
                if scores and pred in scores:
                    top1 = scores[pred]
                    others = [s for k, s in scores.items() if k != pred]
                    case["margin"] = top1 - max(others) if others else 1.0
            
            bad_cases.append(case)
    
    # 按置信度排序
    if confidence_scores:
        bad_cases.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    return bad_cases[:top_k]


def compute_complete_metrics(predictions: List[str], labels: List[str],
                            queries: List[str],
                            is_boundary_list: List[bool],
                            class_names: List[str],
                            confidence_scores: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    生成完整的评估报告（包含所有指标）
    
    Returns:
        包含所有指标的字典
    """
    # 基础指标
    accuracy = compute_accuracy(predictions, labels)
    boundary_accuracy = compute_boundary_accuracy(predictions, labels, is_boundary_list)
    
    # Precision, Recall, F1 (weighted average)
    precision_w, recall_w, f1_w = compute_precision_recall_f1(predictions, labels, average='weighted')
    
    # Precision, Recall, F1 (macro average)
    precision_m, recall_m, f1_m = compute_precision_recall_f1(predictions, labels, average='macro')
    
    # Per-class metrics
    per_class = compute_per_class_metrics(predictions, labels, class_names)
    
    # Margin
    if confidence_scores:
        avg_margin = compute_average_margin_from_scores(predictions, confidence_scores)
    else:
        avg_margin = 0.0
    
    # Confusion matrix
    cm = compute_confusion_matrix(predictions, labels, class_names)
    
    # Bad cases
    bad_cases = get_bad_cases(predictions, labels, queries, confidence_scores)
    
    # 构建结果字典
    results: Dict[str, Any] = {
        # 总体指标
        "accuracy": round(accuracy, 4),
        "accuracy_percent": f"{accuracy:.2%}",
        
        # 边界样本指标
        "boundary_accuracy": round(boundary_accuracy, 4),
        "boundary_accuracy_percent": f"{boundary_accuracy:.2%}",
        "boundary_samples": int(sum(is_boundary_list)),
        
        # Weighted Precision/Recall/F1
        "precision_weighted": round(precision_w, 4),
        "recall_weighted": round(recall_w, 4),
        "f1_weighted": round(f1_w, 4),
        
        # Macro Precision/Recall/F1
        "precision_macro": round(precision_m, 4),
        "recall_macro": round(recall_m, 4),
        "f1_macro": round(f1_m, 4),
        
        # Margin
        "average_margin": round(avg_margin, 4),
        
        # 混淆矩阵
        "confusion_matrix": cm.tolist(),
        
        # 每类别指标
        "per_class_metrics": per_class,
        
        # 样本统计
        "total_samples": len(predictions),
        "correct_samples": sum(1 for p, l in zip(predictions, labels) if p == l),
        "error_samples": sum(1 for p, l in zip(predictions, labels) if p != l),
        
        # 错误案例
        "bad_cases": bad_cases,
        "bad_cases_count": len(bad_cases)
    }
    
    return results


# 向后兼容的函数
def compute_metrics_report(results: List[Tuple[str, ClassifyResult]],
                          is_boundary_list: List[bool],
                          intent_to_agent: Dict[str, str],
                          agent_names: List[str]) -> Dict[str, Any]:
    """
    兼容旧接口的评估报告生成
    """
    predictions = []
    labels = []
    queries = []
    confidence_scores: List[Dict[str, float]] = []
    
    for expected_intent, result in results:
        predictions.append(result.predicted_agent)
        labels.append(intent_to_agent.get(expected_intent, ""))
        queries.append(result.raw_response[:100])  # 使用raw_response作为query占位
        confidence_scores.append(result.confidence_scores)
    
    return compute_complete_metrics(
        predictions, labels, queries, is_boundary_list,
        agent_names, confidence_scores
    )
