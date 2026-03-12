#!/usr/bin/env python3
"""
延迟分析工具 - 详细分析不同Prompt长度对推理延迟的影响
"""
import sys
import os
import time
import json
import statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from config import BACKBONE_API_URL, BACKBONE_MODEL

def measure_latency_detailed(api_url, model, system_prompt, query, warmup=False):
    """
    详细测量延迟，区分不同阶段
    
    返回:
        {
            'total_latency': 总延迟（包含网络）,
            'ttft': Time To First Token（首Token延迟）,
            'prompt_tokens': Prompt的token数（估算）,
            'output_tokens': 输出token数
        }
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy"
    }
    
    full_prompt = f"""{system_prompt}

User Query: {query}

Intent Class:"""
    
    # 估算token数（粗略：1个中文字符≈1.5 tokens，英文≈1 token）
    prompt_tokens = len(full_prompt) * 1.2  # 粗略估算
    
    try:
        # 测量总延迟
        start_total = time.time()
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 10
            },
            timeout=60
        )
        end_total = time.time()
        
        result = response.json()
        
        # 尝试从响应中获取实际token数
        usage = result.get("usage", {})
        actual_prompt_tokens = usage.get("prompt_tokens", 0)
        actual_completion_tokens = usage.get("completion_tokens", 0)
        
        total_latency = end_total - start_total
        
        return {
            "total_latency": total_latency,
            "prompt_tokens": actual_prompt_tokens or prompt_tokens,
            "output_tokens": actual_completion_tokens or 1,
            "prompt_length": len(full_prompt),
            "success": True
        }
        
    except Exception as e:
        return {
            "total_latency": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "prompt_length": len(full_prompt),
            "success": False,
            "error": str(e)
        }


def analyze_prompt_length_impact():
    """分析不同Prompt长度对延迟的影响"""
    
    # 定义不同长度的system prompt
    test_prompts = {
        "100字": "你是一个意图分类助手。" + "负责分类用户查询的意图。" * 5,
        "500字": "你是一个意图分类助手。" + "负责分类用户查询的意图到正确类别。" * 25,
        "1000字": "你是一个意图分类助手。" + "根据用户输入的问题，判断其意图属于12个类别之一。" * 30,
        "2000字": "你是一个专业的股票领域意图分类助手。" + "根据用户输入的问题，准确判断其意图类别。" * 60,
        "4000字": "你是一个专业的股票领域意图分类助手，精通各类投资相关问题。" + "能够准确识别用户查询意图。" * 120,
    }
    
    test_query = " quantum tech stocks "
    
    print("=" * 80)
    print("Prompt长度对推理延迟的影响分析")
    print("=" * 80)
    print(f"\nAPI: {BACKBONE_API_URL}")
    print(f"Model: {BACKBONE_MODEL}")
    print(f"测试Query: {test_query}")
    print(f"\n{'Prompt长度':<12} {'字符数':<8} {'估算Tokens':<12} {'平均延迟(ms)':<15} {'样本数'}")
    print("-" * 80)
    
    results = {}
    
    for name, prompt in test_prompts.items():
        latencies = []
        prompt_chars = len(prompt)
        
        # 预热（1次）
        measure_latency_detailed(BACKBONE_API_URL, BACKBONE_MODEL, prompt, test_query, warmup=True)
        time.sleep(0.5)
        
        # 正式测试10次
        for i in range(10):
            result = measure_latency_detailed(BACKBONE_API_URL, BACKBONE_MODEL, prompt, test_query)
            if result["success"]:
                latencies.append(result["total_latency"])
            time.sleep(0.2)
        
        if latencies:
            avg_latency = statistics.mean(latencies) * 1000  # 转为ms
            std_latency = statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0
            min_latency = min(latencies) * 1000
            max_latency = max(latencies) * 1000
            
            results[name] = {
                "chars": prompt_chars,
                "tokens": result.get("prompt_tokens", 0),
                "avg_ms": avg_latency,
                "std_ms": std_latency,
                "min_ms": min_latency,
                "max_ms": max_latency,
                "samples": len(latencies)
            }
            
            print(f"{name:<12} {prompt_chars:<8} {result.get('prompt_tokens', 0):<12} "
                  f"{avg_latency:>10.1f}±{std_latency:<4.1f}  {len(latencies)}")
        else:
            print(f"{name:<12} {prompt_chars:<8} {'ERROR':<12} {'N/A':<15} 0")
    
    # 计算延迟增长比例
    print("\n" + "=" * 80)
    print("延迟增长分析")
    print("=" * 80)
    
    baseline = None
    for name, data in results.items():
        if baseline is None:
            baseline = data["avg_ms"]
            print(f"{name}: {data['avg_ms']:.1f}ms (baseline)")
        else:
            ratio = data["avg_ms"] / baseline
            increase = data["avg_ms"] - baseline
            print(f"{name}: {data['avg_ms']:.1f}ms (×{ratio:.2f}, +{increase:.1f}ms)")
    
    # 保存结果
    with open("latency_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n详细结果已保存到: latency_analysis.json")
    
    return results


if __name__ == "__main__":
    analyze_prompt_length_impact()
