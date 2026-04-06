import json

# Load both result files
with open('exp_1_result/sbx&context.json', 'r') as f:
    context_data = json.load(f)

with open('exp_1_result/sbx.json', 'r') as f:
    sbx_data = json.load(f)

# Extract detailed results
context_results = {r['index']: r for r in context_data['detailed_results']}
sbx_results = {r['index']: r for r in sbx_data['detailed_results']}

# Find questions that context got right but sbx got wrong
improved = []
for idx in context_results:
    if idx in sbx_results:
        ctx = context_results[idx]
        sbx = sbx_results[idx]
        if ctx['correct'] and not sbx['correct']:
            improved.append({
                'index': idx,
                'query': ctx['query'],
                'expected': ctx['expected'],
                'sbx_predicted': sbx['predicted'],
                'context_predicted': ctx['predicted'],
                'sbx_margin': sbx['margin'],
                'context_margin': ctx['margin']
            })

print(f"共找到 {len(improved)} 个问题，sbx&context 答对但 sbx 答错：\n")
print("=" * 80)

for item in improved:
    print(f"\n【序号 {item['index']}】")
    print(f"问题：{item['query']}")
    print(f"正确答案：{item['expected']}")
    print(f"  sbx 预测：{item['sbx_predicted']} (margin: {item['sbx_margin']:.4f})")
    print(f"  sbx&context 预测：{item['context_predicted']} (margin: {item['context_margin']:.4f})")
    print("-" * 80)

# 按预期类别统计
from collections import Counter
intent_counts = Counter([item['expected'] for item in improved])
print(f"\n按意图类别统计：")
for intent, count in intent_counts.most_common():
    print(f"  {intent}: {count} 个")
