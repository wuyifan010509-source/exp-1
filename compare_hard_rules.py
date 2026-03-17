import json

def compare_results():
    # 读取原始 LLM 改写结果
    llm_file = '/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/experiments/rewrite_io_20260313_234406.json'
    with open(llm_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
        
    # 构建包含我们要比较的内容的列表
    queries = []
    llm_rules = []
    
    for item in llm_data:
        queries.append(item['original_query'])
        rule = item.get('matched_rule', '')
        if rule is None:
            rule = ''
        llm_rules.append(rule.strip())
        
    print(f"Total samples to compare: {len(queries)}")
    
    # 导入我们写的硬规则引擎来测试这些 queries
    import sys
    sys.path.append('/home/iilab9/scholar-papers/experiments/intention/exp-1')
    from hard_rules_engine import exact_match
    
    matches = 0
    mismatches = 0
    hard_rule_hits = 0
    llm_empty_hard_hits = 0
    
    print("\n--- 差异对比 (Hard Rule 命中 vs LLM 结果) ---")
    
    for q, llm_rule in zip(queries, llm_rules):
        hard_res = exact_match(q)
        
        # 提取 LLM 规则的主体部分进行比较
        clean_llm_rule = llm_rule
        if "规则" in llm_rule:
            # 统一格式，例如 "规则 1：诊股类 - 纯标的映射"
            clean_llm_rule = llm_rule.split('：')[0].strip() if '：' in llm_rule else llm_rule
            
        hard_rule_name = ''
        clean_hard_rule = ''
        if hard_res:
            hard_rule_hits += 1
            hard_rule_name = hard_res['matched_rule']
            clean_hard_rule = hard_rule_name.split('：')[0].strip() if '：' in hard_rule_name else hard_rule_name
            
            # 如果硬规则命中了，我们比较一下它和LLM的判断是否一致
            if clean_hard_rule in clean_llm_rule or clean_llm_rule in clean_hard_rule:
                matches += 1
            else:
                mismatches += 1
                if llm_rule == '' or llm_rule == '无':
                    llm_empty_hard_hits += 1
                    print(f"[LLM 漏判 / 硬规则拦截] Query: {q}")
                    print(f"  -> 硬规则判定: {hard_rule_name}")
                    print(f"  -> LLM 判定: {llm_rule}")
                else:
                    print(f"[分类冲突] Query: {q}")
                    print(f"  -> 硬规则判定: {hard_rule_name}")
                    print(f"  -> LLM 判定: {llm_rule}")
    
    print("\n" + "="*40)
    print("对比统计结果:")
    print("="*40)
    print(f"总测试样本数: {len(queries)}")
    print(f"硬规则总命中数: {hard_rule_hits} ({hard_rule_hits/len(queries):.1%})")
    print(f"硬规则与 LLM 结论一致: {matches}")
    print(f"硬规则与 LLM 结论冲突: {mismatches}")
    print(f"其中 LLM 判为'无'但被硬规则成功拦截: {llm_empty_hard_hits}")

if __name__ == '__main__':
    compare_results()
