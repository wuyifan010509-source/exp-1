"""
贪心优化算法
基于LLM定向变异的贪心搜索策略
"""
import json
import copy
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MAX_C_LENGTH, MAX_B_LENGTH, MAX_R_LENGTH
)
from structured_profile import StructuredProfile, ProfileSet
from greedy.llm_logger import LLMInteractionLogger


@dataclass
class GreedyResult:
    """贪心优化结果"""
    best_profile_set: ProfileSet
    best_fitness: float
    fitness_history: List[float]
    improvement_history: List[Dict]
    total_iterations: int
    total_evaluations: int


class GreedyOptimizer:
    """
    贪心优化器
    
    策略：
    1. 从初始解开始
    2. 对每个槽位生成多个候选变异
    3. 选择最佳改进（如果存在）
    4. 重复直到收敛或达到最大迭代次数
    """
    
    def __init__(self, 
                 max_iterations: int = 100,
                 candidates_per_slot: int = 3,
                 patience: int = 10,
                 enable_logging: bool = True,
                 slots_per_iteration: int = 3,
                 window_size: int = 3,
                 improvement_threshold: float = 0.001):
        """
        Args:
            max_iterations: 最大迭代次数
            candidates_per_slot: 每个槽位生成的候选数量
            patience: 早停耐心值（连续多少轮无改进则停止）
            enable_logging: 是否启用LLM交互日志
            slots_per_iteration: 每轮优化的槽位数量（选择性优化）
            window_size: 滑动窗口大小（用于计算基线适应度）
            improvement_threshold: 接受改进的最小阈值
        """
        self.max_iterations = max_iterations
        self.candidates_per_slot = candidates_per_slot
        self.patience = patience
        self.slots_per_iteration = slots_per_iteration
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.logger = LLMInteractionLogger() if enable_logging else None
        self.iteration = 0  # 当前迭代计数
        self.slot_selection_history = {}  # 记录槽位被选中次数：{(agent, slot): count}
        self.round_prompts = []  # 记录每轮的所有prompts
        
    def optimize(self,
                 initial_profile_set: ProfileSet,
                 fitness_func: Callable[[ProfileSet], float],
                 llm_mutator,
                 save_dir: Optional[str] = None,
                 get_bad_cases_func: Optional[Callable] = None) -> GreedyResult:
        """
        执行贪心优化
        
        每轮流程：
        1. 从历史数据集采样100条，分类得到错误案例
        2. 基于错误案例生成新描述
        3. 用黄金测试集评估新描述的准确率
        
        Args:
            initial_profile_set: 初始ProfileSet
            fitness_func: 适应度函数（评估黄金测试集）
            llm_mutator: LLM变异器
            save_dir: 保存结果的目录
            get_bad_cases_func: 从历史数据集获取错误案例的函数
            
        Returns:
            GreedyResult: 优化结果
        """
        # 当前最优解
        current = initial_profile_set.copy()
        current_fitness = fitness_func(current)
        
        # 历史记录
        fitness_history = [current_fitness]
        improvement_history = []
        total_evaluations = 1  # 已经评估了初始解
        
        # 滑动窗口基线记录
        window_fitness_history = [current_fitness]  # 用于滑动窗口的原始适应度（不累积）
        
        # 早停计数器
        no_improvement_count = 0
        
        print(f"[Greedy] Starting optimization...")
        print(f"[Greedy] Initial fitness: {current_fitness:.4f}")
        print(f"[Greedy] Window size: {self.window_size}, Threshold: {self.improvement_threshold}")
        
        for iteration in range(self.max_iterations):
            self.iteration = iteration + 1
            print(f"\n{'='*60}")
            print(f"[Iteration {self.iteration}/{self.max_iterations}]")
            print(f"{'='*60}")
            print(f"[Greedy] Current fitness: {current_fitness:.4f}")
            
            # 【Step 1】从历史数据集采样并分类，获取错误案例（用于指导优化）
            print(f"\n[Step 1] Mining bad cases from historical data...")
            all_cases = None
            if get_bad_cases_func:
                try:
                    all_cases = get_bad_cases_func(current, sample_size=100)
                    if all_cases:
                        error_count = sum(1 for c in all_cases if not c.get('is_correct', False))
                        print(f"[Step 1] Mined {len(all_cases)} cases ({error_count} errors) from historical data")
                except Exception as e:
                    print(f"[Step 1] Warning: Failed to mine bad cases: {e}")
            
            # 【Step 2】基于错误案例生成新描述，并用黄金测试集评估
            print(f"\n[Step 2] Generating new descriptions based on bad cases...")
            is_initialization = (iteration == 0)
            best_improvement = self._find_best_improvement(
                current, current_fitness, fitness_func, 
                llm_mutator, all_cases, total_evaluations,
                is_initialization=is_initialization
            )
            
            improved_profile, improved_fitness, evaluations_made, improvement_info, llm_logs = best_improvement
            total_evaluations += evaluations_made
            
            # 【滑动窗口基线】计算最近N轮的平均适应度作为基线
            window = window_fitness_history[-self.window_size:]
            baseline = sum(window) / len(window)
            
            # 检查是否有改进（必须超过基线+阈值）
            improvement_over_baseline = improved_fitness - baseline
            accepted = improvement_over_baseline > self.improvement_threshold
            
            print(f"\n[Greedy] Baseline (window={len(window)}): {baseline:.4f}")
            print(f"[Greedy] Improved fitness: {improved_fitness:.4f}")
            print(f"[Greedy] Improvement over baseline: {improvement_over_baseline:+.4f} (threshold: {self.improvement_threshold})")
            
            if accepted:
                improvement = improved_fitness - current_fitness
                current = improved_profile
                current_fitness = improved_fitness
                no_improvement_count = 0
                
                # 记录改进（用于累积历史展示）
                fitness_history.append(current_fitness)
                improvement_history.append(improvement_info)
                
                # 【关键】更新滑动窗口历史（记录当前轮次的适应度，用于下一轮基线计算）
                window_fitness_history.append(current_fitness)
                
                print(f"[Greedy] ✓ Improvement accepted! +{improvement:.4f} over previous")
                print(f"[Greedy] New fitness: {current_fitness:.4f}")
                # 打印所有修改的槽位
                changes = improvement_info.get('changes', [])
                print(f"[Greedy] Changed {len(changes)} slot(s):")
                for i, change in enumerate(changes, 1):
                    print(f"  {i}. {change['agent']} - {change['slot']}")
            else:
                no_improvement_count += 1
                print(f"[Greedy] ✗ No improvement accepted (below threshold)")
                print(f"[Greedy] No improvement count: {no_improvement_count}/{self.patience}")
                
                # 【关键】即使未接受，也记录本轮适应度到窗口历史
                # 这样基线会反映真实的性能趋势（包括换数据导致的波动）
                window_fitness_history.append(improved_fitness)
            
            # 【保存检查点】在改进判断后保存，确保保存的是最新状态
            if save_dir:
                self._save_checkpoint(current, current_fitness, self.iteration, save_dir)
                
                # 早停检查
                if no_improvement_count >= self.patience:
                    print(f"[Greedy] Early stopping triggered after {self.iteration} iterations")
                    break
            
            # 【每轮记录】打印所有智能体的完整描述
            print(f"\n{'='*80}")
            print(f"[Round {self.iteration} Summary] All Agent Profiles:")
            print(f"{'='*80}")
            for agent_name in sorted(current.profiles.keys()):
                profile = current.get_profile(agent_name)
                print(f"\n【{agent_name}】")
                print(f"  C-核心能力: {profile.core_capability[:80]}{'...' if len(profile.core_capability) > 80 else ''}")
                print(f"  B-处理边界: {profile.boundary[:80]}{'...' if len(profile.boundary) > 80 else ''}")
                print(f"  R-拒绝范围: {profile.rejection_scope[:80]}{'...' if len(profile.rejection_scope) > 80 else ''}")
            print(f"\n{'='*80}")
            
            # 记录LLM交互日志（记录所有候选，标记是否被接受）
            if self.logger and llm_logs:
                # 获取所有被修改的槽位
                changes = improvement_info.get('changes', [])
                modified_slots = {(c['agent'], c['slot']) for c in changes}
                
                for log in llm_logs:
                    # 如果这个候选对应的槽位在被修改的列表中，则标记为接受
                    is_this_accepted = (accepted and 
                                      (log['agent'], log['slot']) in modified_slots)
                    
                    self.logger.log_interaction(
                        iteration=self.iteration,
                        agent=log['agent'],
                        slot=log['slot'],
                        prompt=log['prompt'],
                        response=log['response'],
                        fitness_before=current_fitness if not accepted else current_fitness - (improved_fitness - current_fitness),
                        fitness_after=log.get('fitness_after', current_fitness),
                        accepted=is_this_accepted,
                        candidate_idx=log['candidate_idx'],
                        bad_cases=log.get('bad_cases', [])
                    )
        
        print(f"\n{'='*60}")
        print(f"[Greedy] Optimization finished!")
        print(f"[Greedy] Final fitness: {current_fitness:.4f}")
        print(f"[Greedy] Total evaluations: {total_evaluations}")
        print(f"{'='*60}")
        
        # 记录实验总结
        if self.logger:
            accepted_count = len(improvement_history)
            total_llm_calls = len(self.logger.interactions)
            self.logger.log_summary(
                total_iterations=self.iteration,
                total_interactions=total_llm_calls,
                accepted_count=accepted_count,
                final_fitness=current_fitness
            )
            
            # 打印日志统计
            stats = self.logger.get_stats()
            print(f"\n[Logger] LLM交互统计:")
            print(f"  总调用次数: {stats.get('total_llm_calls', 0)}")
            print(f"  接受改进: {stats.get('accepted_count', 0)}")
            print(f"  接受率: {stats.get('acceptance_rate', 0)*100:.1f}%")
            print(f"  平均提升: {stats.get('average_improvement', 0):.4f}")
        
        # 保存所有轮次的prompts到文件
        if self.round_prompts and save_dir:
            self._save_round_prompts(save_dir)
        
        return GreedyResult(
            best_profile_set=current,
            best_fitness=current_fitness,
            fitness_history=fitness_history,
            improvement_history=improvement_history,
            total_iterations=len(fitness_history) - 1,
            total_evaluations=total_evaluations
        )
    
    def _select_worst_slots(self, 
                           profile_set: ProfileSet,
                           all_cases: Optional[List[Dict]],
                           top_k: int = 2) -> List[Tuple[str, str]]:
        """
        选择最差的top_k个槽位进行优化
        
        策略：
        1. 只优化 B（处理边界）和 R（拒绝范围），不优化 C（核心能力）
        2. 只根据修改历史选择（修改次数越少优先级越高）
        3. 同一个智能体每轮最多只选一个槽位
        
        Returns:
            List of (agent_name, slot_code) tuples
        """
        agent_slot_scores = {}
        
        # 为每个智能体的 B 和 R 槽位计算得分（C槽位不迭代）
        for agent_name in profile_set.profiles.keys():
            # 只考虑 B 和 R 槽位（不优化 C）
            for slot_code in ['B', 'R']:
                slot_key = (agent_name, slot_code)
                selection_count = self.slot_selection_history.get(slot_key, 0)
                # 修改次数越少得分越高（负数）
                agent_slot_scores[(agent_name, slot_code)] = -selection_count
        
        # 按智能体分组，每个智能体只选修改次数最少的一个槽位
        agent_slots = {}
        for (agent_name, slot_code), score in agent_slot_scores.items():
            if agent_name not in agent_slots:
                agent_slots[agent_name] = []
            agent_slots[agent_name].append((slot_code, score))
        
        # 为每个智能体选择修改次数最少的槽位
        selected = []
        for agent_name, slots in agent_slots.items():
            # 按得分排序（修改次数少的在前）
            slots.sort(key=lambda x: x[1], reverse=True)
            selected_slot = slots[0][0]  # 选得分最高的（修改次数最少的）
            selected.append((agent_name, selected_slot))
        
        # 从所有智能体中按修改次数选择top_k个（修改次数少的优先）
        selected.sort(key=lambda x: agent_slot_scores[x], reverse=True)
        selected = selected[:top_k]
        
        # 更新选中历史
        for agent, slot in selected:
            slot_key = (agent, slot)
            self.slot_selection_history[slot_key] = self.slot_selection_history.get(slot_key, 0) + 1
        
        print(f"\n[Greedy] Selected top {len(selected)} slots for optimization (B/R only, C frozen):")
        slot_name_map = {'C': '核心能力', 'B': '处理边界', 'R': '拒绝范围'}
        for i, (agent, slot) in enumerate(selected, 1):
            history_count = self.slot_selection_history.get((agent, slot), 0)
            print(f"  {i}. {agent} - {slot_name_map[slot]} (modified {history_count} times)")
        
        return selected
    
    def _find_best_improvement(self,
                               current: ProfileSet,
                               current_fitness: float,
                               fitness_func: Callable,
                               llm_mutator,
                               all_cases: Optional[List[Dict]],
                               eval_count_start: int,
                               is_initialization: bool = False) -> Tuple[ProfileSet, float, int, Dict, List[Dict]]:
        """
        在最差的槽位中寻找最佳改进（改完3个槽位后统一评估一次）
        
        流程：
        1. 基于all_cases（历史数据集的bad cases）生成候选描述
        2. 应用所有改进
        3. 用fitness_func（黄金测试集）统一评估
        4. 如果比基线好就接受，否则全部拒绝
        
        Args:
            all_cases: 从历史数据集挖掘的错误案例
            is_initialization: 是否是初始化阶段（影响字数限制和例子类型）
        
        Returns:
            (best_profile, best_fitness, evaluations_made, improvement_info, llm_logs)
        """
        evaluations_made = 0
        all_llm_logs = []  # 收集所有LLM交互日志
        
        # 选择最差的槽位进行优化
        selected_slots = self._select_worst_slots(
            current, all_cases, top_k=self.slots_per_iteration
        )
        
        print(f"\n[Greedy] Optimizing {len(selected_slots)} slots, evaluate ONCE after all changes")
        print(f"[Greedy] Estimated LLM calls: {len(selected_slots) * self.candidates_per_slot}")
        print(f"[Greedy] Estimated evaluations: 1 (instead of {len(selected_slots)})")
        
        # 步骤1：依次修改3个槽位（不评估，只生成）
        working_profile = current.copy()
        changes_made = []  # 记录所有修改
        
        for slot_idx, (agent_name, slot) in enumerate(selected_slots):
            profile = working_profile.get_profile(agent_name)
            
            slot_name_map = {'C': '核心能力', 'B': '处理边界', 'R': '拒绝范围'}
            slot_name = slot_name_map[slot]
            max_len_map = {'C': MAX_C_LENGTH, 'B': MAX_B_LENGTH, 'R': MAX_R_LENGTH}
            max_len = max_len_map[slot]
            
            current_text = profile.get_slot(slot)
            
            print(f"\n[Greedy] [{slot_idx+1}/{len(selected_slots)}] Modifying {agent_name} - {slot_name}")
            print(f"[Greedy] Current: {current_text[:50]}...")
            
            # 生成候选
            candidates, candidate_logs = self._generate_candidates(
                llm_mutator, agent_name, slot, slot_name, 
                current_text, max_len, all_cases, is_initialization
            )
            all_llm_logs.extend(candidate_logs)
            
            # 记录该轮的prompts
            for log in candidate_logs:
                self.round_prompts.append({
                    'iteration': self.iteration,
                    'agent': log['agent'],
                    'slot': log['slot'],
                    'prompt': log['prompt'],
                    'timestamp': datetime.now().isoformat()
                })
            
            # 只取第一个候选（因为candidates_per_slot=1）
            if candidates:
                selected_candidate = candidates[0]
                
                # 应用到 working_profile
                new_profile = profile.mutate_slot(slot, selected_candidate)
                working_profile.profiles[agent_name] = new_profile
                
                changes_made.append({
                    "agent": agent_name,
                    "slot": slot_name,
                    "slot_code": slot,
                    "before": current_text,
                    "after": selected_candidate
                })
                
                print(f"[Greedy] ✓ Modified (not evaluated yet): {selected_candidate[:40]}...")
        
        # 【Step 3】用黄金测试集评估改进后的描述
        print(f"\n[Step 3] Evaluating on golden test set...")
        new_fitness = fitness_func(working_profile)
        evaluations_made += 1
        
        print(f"[Greedy] Combined result: {new_fitness:.4f}")
        
        # 步骤3：判断是否接受（改进信息记录所有修改）
        improvement_info = {
            "agent": "multiple",
            "slot": f"{len(changes_made)} slots",
            "slot_code": "C/B/R",
            "candidate_idx": 0,
            "changes": changes_made,
            "eval_num": eval_count_start + evaluations_made
        }
        
        return working_profile, new_fitness, evaluations_made, improvement_info, all_llm_logs
    
    def _generate_candidates(self, llm_mutator, agent_name: str, 
                            slot_code: str, slot_name: str,
                            current_text: str, max_len: int,
                            all_cases: Optional[List[Dict]],
                            is_initialization: bool = False) -> Tuple[List[str], List[Dict]]:
        """
        使用LLM生成多个候选变异
        
        Args:
            is_initialization: 是否是初始化阶段（影响字数限制和例子类型）
        
        Returns:
            (candidates, llm_logs) - 候选列表和对应的LLM交互日志
        """
        candidates = []
        llm_logs = []  # 记录每个候选的LLM交互
        
        if not llm_mutator.use_llm:
            # LLM不可用，使用随机变异
            for i in range(self.candidates_per_slot):
                candidate = llm_mutator._random_mutate_slot(current_text, max_len)
                if candidate not in candidates:
                    candidates.append(candidate)
                    # 记录随机变异的"伪日志"
                    llm_logs.append({
                        'agent': agent_name,
                        'slot': slot_name,
                        'prompt': f'[Random Mutation] {current_text}',
                        'response': candidate,
                        'candidate_idx': i,
                        'bad_cases': []
                    })
            return candidates, llm_logs
        
        # 使用LLM生成多个候选
        temperatures = [0.3, 0.5, 0.7]  # 不同的创造性程度
        
        # 根据槽位类型筛选错例
        slot_bad_cases = self._filter_bad_cases(all_cases, agent_name, slot_code)
        
        for i, temp in enumerate(temperatures[:self.candidates_per_slot]):
            try:
                # 构建prompt
                prompt = self._build_prompt(
                    agent_name, slot_name, current_text, max_len, slot_bad_cases, is_initialization
                )
                
                # 调用LLM
                response = llm_mutator.client.chat.completions.create(
                    model=llm_mutator.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=200
                )
                
                result = response.choices[0].message.content.strip()
                result = result.replace(f"[{slot_name}]", "").strip()
                
                if result and result not in candidates:
                    candidates.append(result)
                    # 记录这次LLM交互
                    llm_logs.append({
                        'agent': agent_name,
                        'slot': slot_name,
                        'prompt': prompt,
                        'response': result,
                        'candidate_idx': i,
                        'bad_cases': slot_bad_cases
                    })
                    
            except Exception as e:
                print(f"  [Warning] LLM generation failed: {e}")
                continue
        
        # 如果LLM生成不足，用随机变异补充
        while len(candidates) < self.candidates_per_slot:
            idx = len(candidates)
            candidate = llm_mutator._random_mutate_slot(current_text, max_len)
            if candidate not in candidates:
                candidates.append(candidate)
                llm_logs.append({
                    'agent': agent_name,
                    'slot': slot_name,
                    'prompt': f'[Random Mutation Fallback] {current_text}',
                    'response': candidate,
                    'candidate_idx': idx,
                    'bad_cases': []
                })
        
        return candidates[:self.candidates_per_slot], llm_logs[:self.candidates_per_slot]
    
    def _filter_bad_cases(self, all_cases: Optional[List[Dict]], 
                         agent_name: str, slot_code: str) -> List[Dict]:
        """筛选与当前槽位相关的错例"""
        if not all_cases:
            return []
        
        filtered = []
        for case in all_cases:
            expected = case.get('expected_agent', '')
            predicted = case.get('predicted_agent', '')
            is_correct = case.get('is_correct', False)
            
            if slot_code in ['C', 'B']:
                # C/B槽位关注：本该是本智能体但被分走（分错的案例）
                if expected == agent_name and not is_correct:
                    filtered.append(case)
            else:  # R槽位
                # R槽位关注：不该是本智能体但被错收
                if predicted == agent_name and expected != agent_name:
                    filtered.append(case)
        
        return filtered[:5]  # 最多5个案例
    
    def _build_prompt(self, agent_name: str, slot_name: str,
                     current_text: str, max_len: int,
                      all_cases: List[Dict],
                     is_initialization: bool = False) -> str:
        """构建LLM Prompt（支持初始化和迭代的不同策略）"""
        
        # 字数限制：B和R统一≤70字，C在初始化时限制字数
        if slot_name in ['处理边界', '拒绝范围']:
            # B/R：不分初始化和后续，统一70字
            length_limit = "≤70字"
            length_constraint = 70
        else:
            # C：初始化限制字数，后续可以用更长的
            if is_initialization:
                length_limit = "≤50字"
                length_constraint = 50
            else:
                length_limit = "≤70字"
                length_constraint = 70
        
        # 根据槽位类型和阶段筛选例子
        # C槽位和B槽位：都给期望是本智能体的例子（同类例子）
        # R槽位：给错例（智能体和预期答案不一样），包含正确类别
        positive_examples = []  # 期望是本智能体的例子（用于C和B）
        negative_examples = []  # 错例（用于R），格式为 (query, correct_agent)
        
        if all_cases:
            for case in all_cases[:10]:  # 最多10个例子
                query = case.get('query', '')
                expected = case.get('expected_agent', '')
                predicted = case.get('predicted_agent', '')
                
                if not query:
                    continue
                
                # C和B槽位的例子：期望是本智能体（同类例子）
                if expected == agent_name:
                    positive_examples.append(query)
                # 负例：被错分为本智能体（不同类型），记录正确类别
                elif predicted == agent_name and expected != agent_name:
                    negative_examples.append((query, expected))
        
        # 根据槽位类型构建专属prompt
        if slot_name == '核心能力':
            # C槽位：给同类例子（期望是本智能体的例子）
            if positive_examples and not is_initialization:
                examples_str = "、".join([f'"{ex[:15]}"' for ex in positive_examples[:3]])
                example_section = f"\n同类例子（必须参考，描述这些查询的处理能力）：{examples_str}\n"
            else:
                example_section = ""
                
            prompt = f"""你是一个智能体描述优化专家。

当前智能体：{agent_name}
需要优化的槽位：[核心能力]
当前内容：{current_text}{example_section}

请生成[核心能力]描述：
1 字数：严格{length_limit}，超出字数则减少描述性语言，保留例子
2 内容：描述该智能体主要负责什么
3. **必须包含具体示例，例子越多越好，描述要少，不要改写例子要输出原文**
4. **边界范围、核心能力只说"包括"**

请直接输出（只输出核心能力，不要解释，不要输出B和R）：
[核心能力]"""

        elif slot_name == '处理边界':
            # B槽位：给同类型例子
            if positive_examples:
                examples_str = "、".join([f'"{ex[:15]}"' for ex in positive_examples[:3]])
                example_section = f"\n正例（必须参考，这些是应处理的查询）：{examples_str}\n"
            else:
                example_section = ""
                
            prompt = f"""你是一个智能体描述优化专家。

当前智能体：{agent_name}
需要优化的槽位：[处理边界]
当前内容：{current_text}{example_section}

请生成[处理边界]描述：
1 字数：严格{length_limit}，超出字数则减少描述性语言，保留例子
2 内容：描述该智能体的包括什么
3. **必须包含具体示例，例子越多越好，描述要少，不要改写例子要输出原文**
4. **边界范围、核心能力只说"包括"**


诊股智能体核心能力/能力边界举例：包括单只股票诊断。例如：“航天工程”诊股、“我的自选股表现”。
请直接输出（只输出核心能力，不要解释，不要输出C和R）：

[处理边界]"""

        else:  # 拒绝范围
            # R槽位：给错例（智能体和预期答案不一样），格式为 (query, correct_agent)
            if negative_examples:
                # 格式：'查询'（实际xxx类）
                examples_str = "、".join([f'"{ex[0][:10]}"（实际{ex[1][:6]}）' for ex in negative_examples[:3]])
                example_section = f"\n错例（这些是应拒绝的查询，括号内为实际应处理的智能体）：{examples_str}\n"
            else:
                example_section = ""

            prompt = f"""你是一个智能体描述优化专家。

当前智能体：{agent_name}
需要优化的槽位：[拒绝范围]
当前内容：{current_text}{example_section}

请生成[拒绝范围]描述：
1 字数：严格{length_limit}，超出字数则减少描述性语言，保留例子
2 内容：描述该智能体不包括什么，必须标注正确类别
3. **必须包含具体示例，格式为：不包括'查询'（实际xxx类），例子越多越好**
4. **不要改写例子要输出原文，必须加引号和括号标注**


诊股智能体拒绝范围举例：不包括'河钢资源最新新闻'（实际新闻类）、'潜伏吸筹什么意思'（实际知识库类）。
请直接输出（只输出拒绝范围，不要解释，不要输出B和C）：

[拒绝范围]"""
        
        return prompt
    
    def _save_checkpoint(self, profile_set: ProfileSet, 
                        fitness: float, iteration: int, save_dir: str):
        """保存检查点（包含所有智能体的完整描述）"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON格式
        json_filepath = os.path.join(save_dir, f"greedy_iter_{iteration}_{timestamp}.json")
        
        data = {
            "iteration": iteration,
            "fitness": fitness,
            "timestamp": timestamp,
            "profiles": profile_set.to_dict()
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 同时保存为人类可读的文本格式
        text_filepath = os.path.join(save_dir, f"greedy_iter_{iteration}_{timestamp}.txt")
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"Round {iteration} - Fitness: {fitness:.4f}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"=" * 80 + "\n\n")
            
            for agent_name in sorted(profile_set.profiles.keys()):
                profile = profile_set.get_profile(agent_name)
                f.write(f"【{agent_name}】\n")
                f.write(f"C-核心能力: {profile.core_capability}\n")
                f.write(f"B-处理边界: {profile.boundary}\n")
                f.write(f"R-拒绝范围: {profile.rejection_scope}\n")
                f.write(f"\n")
        
        print(f"[Greedy] Checkpoint saved: {json_filepath}")
        print(f"[Greedy] Text summary saved: {text_filepath}")
    
    def _save_round_prompts(self, save_dir: Optional[str]):
        """保存每轮的所有prompts到文件"""
        if not save_dir:
            save_dir = "logs/prompts"
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON格式
        json_filepath = os.path.join(save_dir, f"round_prompts_{timestamp}.json")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.round_prompts, f, ensure_ascii=False, indent=2)
        
        # 同时保存为人类可读的文本格式
        text_filepath = os.path.join(save_dir, f"round_prompts_{timestamp}.txt")
        with open(text_filepath, 'w', encoding='utf-8') as f:
            current_iteration = 0
            for prompt_data in self.round_prompts:
                if prompt_data['iteration'] != current_iteration:
                    current_iteration = prompt_data['iteration']
                    f.write(f"\n{'='*80}\n")
                    f.write(f"第 {current_iteration} 轮\n")
                    f.write(f"{'='*80}\n\n")
                
                f.write(f"【{prompt_data['agent']} - {prompt_data['slot']}】\n")
                f.write(f"时间: {prompt_data['timestamp']}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{prompt_data['prompt']}\n")
                f.write(f"\n")
        
        print(f"\n[Logger] 每轮Prompts已保存:")
        print(f"  JSON: {json_filepath}")
        print(f"  TXT:  {text_filepath}")
