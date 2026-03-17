import re

# 模拟的知识库词典（实际生产中应从数据库或文件中加载）
STOCK_DICT = {"华西股份", "索通发展", "楚江新材", "航天发展", "旋极信息", "301299", "603228", "300803", "创新医疗"}
INDICATOR_DICT = {"概念板块", "市盈率", "换手率", "涨跌幅", "成交量", "主力资金"}
SOFTWARE_TERMS = {"擒龙", "擒龙平台", "指南针", "黄金坑", "圆圈战", "积分"}
CONCEPT_DICT = {"计算机板块", "核聚变", "成份股", "因用股票", "多模态大模型", "京津冀一体化"}
FINANCE_TERMS = {"大盘量能", "价值决策", "锁定因子", "筹码集中度", "北证股票", "破净股", "严重异动", "重合度"}

def exact_match(query: str) -> dict:
    """
    硬规则匹配引擎 (第一梯队 & 第二梯队)
    返回格式: {"matched_rule": "规则X", "rewritten_query": "改写后的文本"}
    如果未命中任何硬规则，返回 None
    """
    query = query.strip()
    if not query:
        return None
        
    # 去除句尾常见标点符号，方便完全匹配
    clean_query = re.sub(r'[？。，！?.,!]+$', '', query)

    # ==========================================
    # 第一梯队：极其适合硬规则（极速拦截）
    # ==========================================

    # 【规则 1：诊股类 - 纯标的映射】
    # 条件：完全由6位数字组成（可能是代码），或完全等于股票名称。且不能包含预测词。
    predict_keywords = ["后期", "下一步", "未来", "明天", "走势", "前景", "还会涨吗", "怎么走"]
    contains_predict = any(kw in query for kw in predict_keywords)
    
    if not contains_predict:
        is_pure_code = re.match(r'^(sh|sz)?\d{6}$', clean_query, re.IGNORECASE)
        if is_pure_code or clean_query in STOCK_DICT:
            return {
                "matched_rule": "规则 1：诊股类 - 纯标的映射",
                "rewritten_query": f"请对 {clean_query} 这只股票进行综合评价、基本面和技术面的诊断。"
            }

    # 【规则 5：策略类 - 策略查询】
    # 条件：包含"策略"，并且前面有标的或者特定的策略名
    if re.search(r'(.*?)交易策略', query):
        match = re.search(r'(.*?)交易策略', query)
        target = match.group(1).strip()
        if target:
            return {
                "matched_rule": "规则 5：策略类 - 策略查询",
                "rewritten_query": f"请处理{target}交易策略的解析"
            }

    # 【规则 8：知识库类 - 软件操作】
    # 条件：包含私有软件黑话实体
    if any(term in query for term in SOFTWARE_TERMS):
        return {
            "matched_rule": "规则 8：知识库类 - 软件操作",
            "rewritten_query": f"告诉我以下私有知识:{query}"
        }

    # ==========================================
    # 第二梯队：组合硬规则（实体 + 强句式）
    # ==========================================

    # 【规则 3：预测类 - 时空穿梭映射】
    # 条件：提及了具体股票，并且包含了时间推演词汇
    if contains_predict:
        # 提取句子中可能包含的股票名称（粗略提取，实际可使用 Aho-Corasick 自动机进行高效提取）
        mentioned_stocks = [s for s in STOCK_DICT if s in query]
        if mentioned_stocks:
            stock = mentioned_stocks[0] # 假设只取第一个
            # 提取推演词（这里简单写为“未来”，实际可以根据正则捕获的内容动态替换）
            match = re.search(r'(明天|未来|下一步|往后)', query)
            time_word = match.group(1) if match else "未来的走势"
            
            return {
                "matched_rule": "规则 3：预测类 - 时空穿梭映射",
                "rewritten_query": f"请预测和推演{stock}{time_word}的前景。"
            }

    # 【规则 6：指标查询类 - 指标查询】
    # 条件：正则匹配 "xxx的yyy"，且 xxx 是股票，yyy 是指标
    match = re.search(r'(.+?)的(.+)', clean_query)
    if match:
        target, indicator = match.group(1).strip(), match.group(2).strip()
        # 处理可能的后缀疑问词，例如 "概念板块是什么" -> "概念板块"
        indicator_clean = re.sub(r'(是什么|是多少|怎么看)$', '', indicator)
        
        if target in STOCK_DICT and indicator_clean in INDICATOR_DICT:
            return {
                "matched_rule": "规则 6：指标查询类 - 指标查询",
                "rewritten_query": f"请问查询{target}的{indicator_clean}这一指标"
            }

    # 【规则 2：选股类 - 纯板块映射】
    # 条件：几乎是纯实体，可能带有 "是什么"、"有哪些" 这种无关词
    # 比如 "计算机板块有哪些"，去掉"有哪些"后在字典中
    r2_clean = re.sub(r'(有哪些|是什么|标的股|的股票|股票)?$', '', clean_query).strip()
    if r2_clean in CONCEPT_DICT:
        return {
            "matched_rule": "规则 2：选股类 - 纯板块映射",
            "rewritten_query": f"请按条件筛选并列出{r2_clean}板块下的所有成分股名单。"
        }
        
    # 【规则 7：知识库类 - 概念询问】
    # 比如 "什么是锁定因子" 或 "破净股是什么意思"
    r7_clean = re.sub(r'^(什么是|什么叫|怎么查)', '', clean_query)
    r7_clean = re.sub(r'(是什么意思|是什么|有什么意义|说明什么|怎么查|动向)?$', '', r7_clean).strip()
    if r7_clean in FINANCE_TERMS:
        return {
            "matched_rule": "规则 7：知识库类 - 概念询问",
            "rewritten_query": f"什么是{r7_clean}"
        }

    # 未命中任何硬规则
    return None

def test_engine():
    test_cases = [
        "华西股份",
        "300803",
        "景旺电子交易策略。",
        "指南针擒龙版积分在哪里查看",
        "航空发展往后怎么走，是走还是留",
        "300803的概念板块是什么",
        "计算机板块",
        "成份股有哪些",
        "锁定因子是什么",
        "破净股是什么意思",
        # 以下不应命中硬规则
        "今天底位首板股票",
        "一阳穿三线的",
        "明天天气怎么样？"
    ]
    
    print("=== 硬规则匹配引擎测试 ===")
    for q in test_cases:
        res = exact_match(q)
        print(f"\n原句: {q}")
        if res:
            print(f"命中: {res['matched_rule']}")
            print(f"改写: {res['rewritten_query']}")
        else:
            print("命中: None (需转入大模型或意图分类器)")

if __name__ == "__main__":
    test_engine()
