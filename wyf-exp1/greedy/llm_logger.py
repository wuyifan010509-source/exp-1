"""
LLM交互日志记录器
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


class LLMInteractionLogger:
    """记录LLM所有交互的日志器"""
    
    def __init__(self, log_dir: str = "logs/llm_interactions"):
        # 使用绝对路径，确保在正确位置创建
        if not os.path.isabs(log_dir):
            # 从当前文件位置计算项目根目录
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            log_dir = os.path.join(project_root, log_dir)
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        print(f"[Logger] Creating log directory: {log_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"greedy_llm_log_{timestamp}.jsonl")
        self.interactions = []
        
        print(f"[Logger] LLM交互日志将保存到: {self.log_file}")
    
    def log_interaction(self,
                       iteration: int,
                       agent: str,
                       slot: str,
                       prompt: str,
                       response: str,
                       fitness_before: float,
                       fitness_after: float,
                       accepted: bool,
                       candidate_idx: int = 0,
                       bad_cases: Optional[List[Dict]] = None) -> None:
        """
        记录一次LLM交互
        
        Args:
            iteration: 迭代轮次
            agent: 智能体名称
            slot: 槽位名称
            prompt: 发送给LLM的prompt
            response: LLM返回的结果
            fitness_before: 改进前适应度
            fitness_after: 改进后适应度
            accepted: 是否接受了这次改进
            candidate_idx: 候选编号
            bad_cases: 使用的错例
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "agent": agent,
            "slot": slot,
            "prompt": prompt,
            "response": response,
            "fitness_before": fitness_before,
            "fitness_after": fitness_after,
            "improvement": fitness_after - fitness_before,
            "accepted": accepted,
            "candidate_idx": candidate_idx,
            "bad_cases_count": len(bad_cases) if bad_cases else 0,
            "bad_cases": bad_cases if bad_cases else []
        }
        
        self.interactions.append(interaction)
        
        # 立即写入文件（追加模式）
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
                f.flush()  # 强制刷新缓冲区
            # print(f"[Logger] Logged interaction to {self.log_file}")  # 调试用
        except Exception as e:
            print(f"[Logger Error] Failed to write log: {e}")
    
    def log_summary(self, total_iterations: int, 
                   total_interactions: int,
                   accepted_count: int,
                   final_fitness: float):
        """记录实验总结"""
        summary = {
            "type": "summary",
            "timestamp": datetime.now().isoformat(),
            "total_iterations": total_iterations,
            "total_llm_calls": total_interactions,
            "accepted_improvements": accepted_count,
            "rejected_improvements": total_interactions - accepted_count,
            "acceptance_rate": accepted_count / total_interactions if total_interactions > 0 else 0,
            "final_fitness": final_fitness
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.interactions:
            return {}
        
        total = len(self.interactions)
        accepted = sum(1 for i in self.interactions if i['accepted'])
        
        improvements = [i['improvement'] for i in self.interactions if i['accepted']]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        
        # 统计每个槽位的改进次数
        slot_stats = {}
        for i in self.interactions:
            slot = i['slot']
            if slot not in slot_stats:
                slot_stats[slot] = {'total': 0, 'accepted': 0}
            slot_stats[slot]['total'] += 1
            if i['accepted']:
                slot_stats[slot]['accepted'] += 1
        
        return {
            "total_llm_calls": total,
            "accepted_count": accepted,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "average_improvement": avg_improvement,
            "slot_statistics": slot_stats
        }
