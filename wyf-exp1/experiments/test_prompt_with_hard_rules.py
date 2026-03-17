"""
测试 gen_10_no_oos.txt prompt 在 GOLDEN_TEST.csv 上的准确率
使用硬规则引擎替换 LLM 的改写模块，当硬规则未命中时，兜底使用 LLM 进行意图分类。
"""
import sys
import os
import re
import json
import pandas as pd
import requests
from datetime import datetime
from collections import defaultdict

# 配置
BACKBONE_API_URL = "http://172.17.160.42:8080/v1"
BACKBONE_MODEL = "Qwen2.5-32B-Instruct"
GOLDEN_TEST_PATH = "/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/data/GOLDEN_TEST.csv"
PROMPT_PATH = "/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/data/baselines/greedy.txt"

# 意图类别
INTENT_CLASSES = [
    "选股类", "诊股类", "预测类", "知识库类", "新闻类",
    "通用类", "推荐类", "策略类", "指标查询类", "身份类",
    "分时图类", "K线图类"
]

# ==========================================
# 硬规则引擎 (包含词典与匹配逻辑)
# ==========================================
STOCK_DICT = {"华西股份", "索通发展", "楚江新材", "航天发展", "旋极信息", "301299", "603228", "300803", "创新医疗","159218ETF","301299","601992",}
INDICATOR_DICT = {"概念板块", "市盈率", "换手率", "涨跌幅", "成交量", "主力资金","名字","资金流入"}
SOFTWARE_TERMS = {"擒龙", "擒龙平台", "指南针", "黄金坑", "圆圈战", "积分"}
CONCEPT_DICT = {"计算机板块", "核聚变", "成份股", "因用股票", "多模态大模型", "京津冀一体化","Ai企业应用"}
FINANCE_TERMS = {"大盘量能", "价值决策", "锁定因子", "筹码集中度", "北证股票", "破净股", "严重异动", "重合度","70%筹码","游资","机构扫货"}

def exact_match(query: str) -> dict:
    query = query.strip()
    if not query: 
        return None
        
    clean_query = re.sub(r'[？。，！?.,!]+$', '', query)
    predict_keywords = ["后期", "下一步", "未来", "明天", "走势", "前景", "还会涨吗", "怎么走"]
    contains_predict = any(kw in query for kw in predict_keywords)
    
    # 规则1
    if not contains_predict:
        is_pure_code = re.match(r'^(sh|sz)?\d{6}$', clean_query, re.IGNORECASE)
        if is_pure_code or clean_query in STOCK_DICT:
            rule_short_name = "纯标的映射"
            return {
                "matched_rule": "规则 1：诊股类 - 纯标的映射", 
                "rule_short_name": rule_short_name,
                "rewritten_query": f"{rule_short_name}：{clean_query}"
            }

    # 规则5
    if re.search(r'(.*?)策略', query):
        match = re.search(r'(.*?)策略', query)
        target = match.group(1).strip()
        if target:
            rule_short_name = "策略查询"
            return {
                "matched_rule": "规则 5：策略类 - 策略查询", 
                "rule_short_name": rule_short_name,
                "rewritten_query": f"{rule_short_name}：{target}"
            }

    # 规则8
    if any(term in query for term in SOFTWARE_TERMS) or re.match(r'^(怎么查|在哪里查|在哪里查看|如何查|怎么看|在哪看|如何判断)', clean_query):
        rule_short_name = "软件操作"
        return {
            "matched_rule": "规则 8：知识库类 - 软件操作", 
            "rule_short_name": rule_short_name,
            "rewritten_query": f"{rule_short_name}：{query}"
        }

    # 规则3
    if contains_predict:
        mentioned_stocks = [s for s in STOCK_DICT if s in query]
        if mentioned_stocks:
            stock = mentioned_stocks[0]
            match = re.search(r'(明天|未来|下一步|往后)', query)
            time_word = match.group(1) if match else "未来的走势"
            rule_short_name = "时空穿梭映射"
            return {
                "matched_rule": "规则 3：预测类 - 时空穿梭映射", 
                "rule_short_name": rule_short_name,
                "rewritten_query": f"{rule_short_name}：{stock}{time_word}"
            }

    # 规则6 (指标查询)
    for stock in STOCK_DICT:
        if stock in clean_query:
            remaining = clean_query.replace(stock, '').replace('股票', '').replace('的', '').strip()
            indicator_clean = re.sub(r'(是什么|是多少|怎么看|情况|还是流出)$', '', remaining).strip()
            
            if indicator_clean in INDICATOR_DICT or any(ind in remaining for ind in INDICATOR_DICT):
                rule_short_name = "指标查询"
                return {
                    "matched_rule": "规则 6：指标查询类 - 指标查询", 
                    "rule_short_name": rule_short_name,
                    "rewritten_query": f"{rule_short_name}：{stock}的{indicator_clean}"
                }
            
            if re.search(r'板块', remaining):
                rule_short_name = "板块查询"
                return {
                    "matched_rule": "规则 6：指标查询类 - 板块查询", 
                    "rule_short_name": rule_short_name,
                    "rewritten_query": f"{rule_short_name}：{stock}的概念板块"
                }

    # 针对未在股票字典中但明显是股票代码的情况
    code_match = re.search(r'(\d{6})', clean_query)
    if code_match:
        stock_code = code_match.group(1)
        remaining = clean_query.replace(stock_code, '').replace('股票', '').replace('的', '').strip()
        indicator_clean = re.sub(r'(是什么|是多少|怎么看|情况|还是流出)$', '', remaining).strip()
        
        if indicator_clean in INDICATOR_DICT or any(ind in remaining for ind in INDICATOR_DICT):
            rule_short_name = "指标查询"
            return {
                "matched_rule": "规则 6：指标查询类 - 代码指标查询", 
                "rule_short_name": rule_short_name,
                "rewritten_query": f"{rule_short_name}：{stock_code}的{indicator_clean}"
            }
        
        if re.search(r'板块', remaining):
            rule_short_name = "板块查询"
            return {
                "matched_rule": "规则 6：指标查询类 - 代码板块查询", 
                "rule_short_name": rule_short_name,
                "rewritten_query": f"{rule_short_name}：{stock_code}的概念板块"
            }

    # 规则2
    r2_clean = re.sub(r'(有哪些|是什么|标的股|的股票|股票)?$', '', clean_query).strip()
    if r2_clean in CONCEPT_DICT:
        rule_short_name = "纯板块映射"
        return {
            "matched_rule": "规则 2：选股类 - 纯板块映射", 
            "rule_short_name": rule_short_name,
            "rewritten_query": f"{rule_short_name}：{r2_clean}"
        }
        
    # 规则7
    if re.search(r'(有什么意义|是什么意思|说明什么|是啥意思)$', clean_query):
        concept = re.sub(r'(有什么意义|是什么意思|说明什么|是啥意思)$', '', clean_query).strip()
        concept = re.sub(r'^(什么是|什么叫|怎么查|如何判断)', '', concept).strip()
        if concept:
            rule_short_name = "概念询问"
            return {
                "matched_rule": "规则 7：知识库类 - 概念询问", 
                "rule_short_name": rule_short_name,
                "rewritten_query": f"{rule_short_name}：{concept}"
            }

    r7_clean = re.sub(r'^(什么是|什么叫|怎么查)', '', clean_query)
    r7_clean = re.sub(r'(是什么意思|是什么|有什么意义|说明什么|怎么查|动向)?$', '', r7_clean).strip()
    if r7_clean in FINANCE_TERMS:
        rule_short_name = "金融术语"
        return {
            "matched_rule": "规则 7：知识库类 - 金融术语", 
            "rule_short_name": rule_short_name,
            "rewritten_query": f"{rule_short_name}：{r7_clean}"
        }

    return None

def extract_intent_from_rule(rule_name: str) -> str:
    """根据硬规则名称，直接提取对应的意图类别"""
    for intent in INTENT_CLASSES:
        if intent in rule_name:
            return intent
    return "未知分类"

def load_prompt():
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()

def load_test_data():
    df = pd.read_csv(GOLDEN_TEST_PATH)
    df = df[df['预期意图'] != 'oos']
    queries = df['问题'].tolist()
    intents = df['预期意图'].tolist()
    is_boundary = df['是否处于意图边界'].fillna('否').map(lambda x: str(x).strip() == '是').tolist()
    return queries, intents, is_boundary

def classify_query(api_url, model, system_prompt, query):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}
    full_user_prompt = f"{system_prompt}\n\n用户问题：{query}\n\n请直接输出意图类别（只输出类别名称，如'选股类'、'诊股类'等，不要解释）："
    
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": full_user_prompt}],
                "temperature": 0.0,
                "max_tokens": 100
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"].strip()
        predicted = extract_intent(raw_response)
        return predicted, raw_response
    except Exception as e:
        print(f"[Error] Classify API call failed: {e}")
        return "", str(e)

def extract_intent(text):
    text = text.strip()
    for intent in INTENT_CLASSES:
        if intent in text: return intent
    for intent in INTENT_CLASSES:
        short_name = intent.replace("类", "")
        if short_name in text: return intent
    return text[:20] if len(text) > 20 else text

def calculate_metrics(all_predictions, all_labels):
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for pred, true in zip(all_predictions, all_labels):
        confusion_matrix[true][pred] += 1
    
    metrics_per_class = {}
    for intent in INTENT_CLASSES:
        tp = confusion_matrix[intent][intent]
        fp = sum(confusion_matrix[other][intent] for other in INTENT_CLASSES if other != intent)
        fn = sum(confusion_matrix[intent][other] for other in INTENT_CLASSES if other != intent)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[intent] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": tp + fn
        }
        
    macro_precision = sum(m["precision"] for m in metrics_per_class.values()) / len(INTENT_CLASSES)
    macro_recall = sum(m["recall"] for m in metrics_per_class.values()) / len(INTENT_CLASSES)
    macro_f1 = sum(m["f1"] for m in metrics_per_class.values()) / len(INTENT_CLASSES)
    
    total_samples = sum(m["support"] for m in metrics_per_class.values())
    weighted_precision = sum(m["precision"] * m["support"] for m in metrics_per_class.values()) / total_samples if total_samples > 0 else 0
    weighted_recall = sum(m["recall"] * m["support"] for m in metrics_per_class.values()) / total_samples if total_samples > 0 else 0
    weighted_f1 = sum(m["f1"] * m["support"] for m in metrics_per_class.values()) / total_samples if total_samples > 0 else 0
    
    cm_dict = {}
    for true_intent in INTENT_CLASSES:
        cm_dict[true_intent] = {}
        for pred_intent in INTENT_CLASSES:
            cm_dict[true_intent][pred_intent] = confusion_matrix[true_intent][pred_intent]
            
    return {
        "per_class": metrics_per_class,
        "macro_avg": {"precision": round(macro_precision, 4), "recall": round(macro_recall, 4), "f1": round(macro_f1, 4)},
        "weighted_avg": {"precision": round(weighted_precision, 4), "recall": round(weighted_recall, 4), "f1": round(weighted_f1, 4)},
        "confusion_matrix": cm_dict
    }

def evaluate_prompt():
    print("=" * 80)
    print("测试 gen_10_no_oos.txt prompt (硬规则匹配 + 改写后送入LLM)")
    print("=" * 80)
    
    queries, intents, is_boundary = load_test_data()
    print(f"  测试样本: {len(queries)}条")
    system_prompt = load_prompt()
    
    correct, boundary_correct = 0, 0
    total, boundary_total = len(queries), sum(is_boundary)
    
    results = []
    all_predictions, all_labels = [], []
    hard_rule_hit_count = 0
    
    for i, (query, true_intent, is_bound) in enumerate(zip(queries, intents, is_boundary)):
        # print(f"\n{'='*80}\n题目 {i+1}/{total}\n{'='*80}")
        print(f"[原问题] {query}")
        
        # 使用硬规则引擎进行匹配
        rule_match = exact_match(query)
        
        if rule_match:
            # 命中硬规则：改写成"规则名：问题"格式，然后送入LLM
            rewritten_query = rule_match["rewritten_query"]
            print(f"[硬规则命中] {rule_match['matched_rule']}")
            print(f"[原问题] {query}")
            print(f"[改写后] {rewritten_query}")
            
            # 将改写后的问题送入LLM进行分类
            predicted, classify_raw = classify_query(BACKBONE_API_URL, BACKBONE_MODEL, system_prompt, rewritten_query)
            classify_raw = f"[硬规则改写] {rule_match['matched_rule']} -> {classify_raw}"
            hard_rule_hit_count += 1
        else:
            # 未命中：走大模型分类兜底（使用原始问题）
            # print(f"[匹配规则] 未命中，进入 LLM 兜底分类...")
            predicted, classify_raw = classify_query(BACKBONE_API_URL, BACKBONE_MODEL, system_prompt, query)
        
        is_correct = (predicted == true_intent)
        if is_correct:
            correct += 1
            if is_bound: boundary_correct += 1
        
        results.append({
            'index': i + 1, 'original_query': query,
            'matched_rule': rule_match['matched_rule'] if rule_match else None,
            'rewritten_query': rule_match['rewritten_query'] if rule_match else None,
            'expected': true_intent, 'predicted': predicted,
            'correct': is_correct, 'is_boundary': is_bound,
            'classify_raw': classify_raw
        })
        all_predictions.append(predicted)
        all_labels.append(true_intent)
        
        status = "✓ 正确" if is_correct else "✗ 错误"
        if is_correct==False:
            print(f"[原问题] {query}")
            if rule_match:
                print(f"[匹配规则] {rule_match['matched_rule']}")
                print(f"[改写后] {rule_match['rewritten_query']}")
            else:
                print(f"[匹配规则] 未命中，LLM兜底")
        print(f"结果: {status} | 期望: {true_intent} -> 预测: {predicted}")
    
    print(f"\n{'='*80}\n评估结果汇总\n{'='*80}")
    
    accuracy = correct / total if total > 0 else 0
    boundary_accuracy = boundary_correct / boundary_total if boundary_total > 0 else 0
    
    detailed_metrics = calculate_metrics(all_predictions, all_labels)
    
    print(f"硬规则命中率: {hard_rule_hit_count/total:.2%} ({hard_rule_hit_count}/{total})")
    print(f"总体准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"边界准确率: {boundary_accuracy:.2%} ({boundary_correct}/{boundary_total})")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/experiments/eval_hard_rules_{timestamp}.json"
    
    output_data = {
        'total_samples': total, 'correct': correct, 'accuracy': accuracy,
        'hard_rule_hits': hard_rule_hit_count,
        'macro_avg': detailed_metrics['macro_avg'],
        'weighted_avg': detailed_metrics['weighted_avg'],
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_file}")

if __name__ == "__main__":
    evaluate_prompt()
