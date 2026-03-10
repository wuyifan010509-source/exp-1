"""
遗传算法核心引擎
"""
import random
import copy
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    POPULATION_SIZE, N_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE,
    ELITE_COUNT, TOURNAMENT_K, MAX_C_LENGTH, MAX_B_LENGTH, MAX_R_LENGTH
)
from structured_profile import StructuredProfile, ProfileSet


@dataclass
class Individual:
    """GA个体"""
    profile_set: ProfileSet
    fitness: float = 0.0
    generation: int = 0
    
    def copy(self) -> 'Individual':
        """深拷贝"""
        return Individual(
            profile_set=self.profile_set.copy(),
            fitness=self.fitness,
            generation=self.generation
        )


@dataclass
class EvolutionResult:
    """演化结果"""
    best_individual: Individual
    best_fitness_history: List[float]
    avg_fitness_history: List[float]
    population_history: List[List[Individual]]
    total_generations: int


class GeneticAlgorithm:
    """遗传算法"""
    
    def __init__(self, 
                 pop_size: int = POPULATION_SIZE,
                 n_generations: int = N_GENERATIONS,
                 crossover_rate: float = CROSSOVER_RATE,
                 mutation_rate: float = MUTATION_RATE,
                 elite_count: int = ELITE_COUNT,
                 tournament_k: int = TOURNAMENT_K):
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.tournament_k = tournament_k
        
        # 演化历史
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.population_history = []
        
    def evolve(self, initial_population: List[ProfileSet], 
               fitness_func, 
               mutator,
               save_dir: Optional[str] = None,
               get_bad_cases_func = None) -> EvolutionResult:
        """
        执行遗传演化
        
        Args:
            initial_population: 初始种群（ProfileSet列表）
            fitness_func: 适应度函数，输入ProfileSet输出float
            mutator: 变异器，输入ProfileSet和bad_cases输出变异后的ProfileSet
            save_dir: 保存结果的目录
            get_bad_cases_func: 获取bad_cases的函数，输入ProfileSet输出bad_cases列表
        
        Returns:
            EvolutionResult: 演化结果
        """
        # 初始化种群
        population = [
            Individual(profile_set=ps, generation=0)
            for ps in initial_population
        ]
        
        print(f"[GA] Starting evolution: pop_size={self.pop_size}, generations={self.n_generations}")
        
        for gen in range(self.n_generations):
            print(f"\n{'='*60}")
            print(f"[Generation {gen+1}/{self.n_generations}]")
            print(f"{'='*60}")
            
            # 1. 评估适应度
            print(f"[Gen {gen+1}] Evaluating fitness...")
            for ind in population:
                if ind.fitness == 0.0:  # 避免重复评估
                    ind.fitness = fitness_func(ind.profile_set)
            
            # 按适应度排序
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 记录历史
            best_fitness = population[0].fitness
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.population_history.append([ind.copy() for ind in population])
            
            print(f"[Gen {gen+1}] Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}")
            print(f"[Gen {gen+1}] Best Length: {population[0].profile_set.average_length():.1f}")
            
            # 保存检查点（每代都保存）
            if save_dir:
                self._save_checkpoint(population, gen + 1, save_dir)
            
            # 检查是否最后一轮
            if gen == self.n_generations - 1:
                break
            
            # 2. 获取最佳个体的bad cases（用于变异时的定向优化）
            best_individual = population[0]
            bad_cases = None
            if get_bad_cases_func:
                try:
                    bad_cases = get_bad_cases_func(best_individual.profile_set)
                    if bad_cases:
                        print(f"[Gen {gen+1}] Using {len(bad_cases)} bad cases for mutation")
                except Exception as e:
                    print(f"[Gen {gen+1}] Warning: Failed to get bad cases: {e}")
            
            # 3. 精英保留
            new_population = [ind.copy() for ind in population[:self.elite_count]]
            
            # 4. 生成新一代
            while len(new_population) < self.pop_size:
                # 锦标赛选择
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, gen + 1)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异（传递bad_cases实现定向优化）
                if random.random() < self.mutation_rate:
                    child1.profile_set = mutator.mutate(child1.profile_set, gen + 1, bad_cases)
                if random.random() < self.mutation_rate:
                    child2.profile_set = mutator.mutate(child2.profile_set, gen + 1, bad_cases)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.pop_size]
        
        # 最终排序
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        return EvolutionResult(
            best_individual=population[0],
            best_fitness_history=self.best_fitness_history,
            avg_fitness_history=self.avg_fitness_history,
            population_history=self.population_history,
            total_generations=self.n_generations
        )
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """锦标赛选择"""
        contestants = random.sample(population, min(self.tournament_k, len(population)))
        return max(contestants, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual, 
                   generation: int) -> Tuple[Individual, Individual]:
        """
        结构保留交叉 - 在槽位级别交换
        交换父代的C/B/R槽位，而不是字符级别
        """
        child1_profiles = []
        child2_profiles = []
        
        # 对每个智能体进行槽位交叉
        for agent_name in parent1.profile_set.profiles.keys():
            profile1 = parent1.profile_set.get_profile(agent_name)
            profile2 = parent2.profile_set.get_profile(agent_name)
            
            # 随机决定交换哪些槽位
            slots = ['C', 'B', 'R']
            child1_slots = {}
            child2_slots = {}
            
            for slot in slots:
                if random.random() < 0.5:
                    # 交换
                    child1_slots[slot] = profile2.get_slot(slot)
                    child2_slots[slot] = profile1.get_slot(slot)
                else:
                    # 不交换
                    child1_slots[slot] = profile1.get_slot(slot)
                    child2_slots[slot] = profile2.get_slot(slot)
            
            # 创建新的Profile
            child1_profile = StructuredProfile(
                agent_name=agent_name,
                core_capability=child1_slots['C'],
                boundary=child1_slots['B'],
                rejection_scope=child1_slots['R']
            )
            child2_profile = StructuredProfile(
                agent_name=agent_name,
                core_capability=child2_slots['C'],
                boundary=child2_slots['B'],
                rejection_scope=child2_slots['R']
            )
            
            child1_profiles.append(child1_profile)
            child2_profiles.append(child2_profile)
        
        child1 = Individual(
            profile_set=ProfileSet(child1_profiles),
            generation=generation
        )
        child2 = Individual(
            profile_set=ProfileSet(child2_profiles),
            generation=generation
        )
        
        return child1, child2
    
    def _save_checkpoint(self, population: List[Individual], 
                        generation: int, save_dir: str):
        """保存检查点"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"gen_{generation}_{timestamp}.json")
        
        data = {
            "generation": generation,
            "population": [
                {
                    "fitness": ind.fitness,
                    "profiles": ind.profile_set.to_dict()
                }
                for ind in population[:5]  # 只保存Top5
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[GA] Checkpoint saved: {filepath}")
