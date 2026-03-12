"""
绘制贪心优化迭代过程中的Fitness变化折线图
从 greedy_checkpoints 目录读取所有JSON文件
"""
import json
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_fitness_data(checkpoints_dir):
    """从checkpoints目录加载所有iteration和fitness数据"""
    data = []
    
    # 查找所有JSON文件
    json_files = sorted(glob.glob(os.path.join(checkpoints_dir, 'greedy_iter_*.json')))
    
    print(f"Found {len(json_files)} checkpoint files")
    
    for filepath in json_files[:10]:  # 先读取前10个看看
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                iteration = checkpoint.get('iteration', 0)
                fitness = checkpoint.get('fitness', 0)
                data.append({
                    'iteration': iteration,
                    'fitness': fitness,
                    'file': os.path.basename(filepath)
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return data

def plot_fitness_curve(data, output_path='fitness_curve.png'):
    """绘制Fitness变化折线图"""
    if not data:
        print("No data to plot")
        return
    
    # 按iteration排序
    data = sorted(data, key=lambda x: x['iteration'])
    
    iterations = [d['iteration'] for d in data]
    fitnesses = [d['fitness'] for d in data]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制折线图
    ax.plot(iterations, fitnesses, 'b-', linewidth=2, marker='o', markersize=4, label='Fitness')
    
    # 设置标题和标签
    ax.set_title('Greedy Optimization: Fitness vs Iteration', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加初始和最终Fitness标注
    if fitnesses:
        initial_fitness = fitnesses[0]
        final_fitness = fitnesses[-1]
        improvement = final_fitness - initial_fitness
        
        ax.axhline(y=initial_fitness, color='r', linestyle='--', alpha=0.5, label=f'Initial: {initial_fitness:.3f}')
        ax.axhline(y=final_fitness, color='g', linestyle='--', alpha=0.5, label=f'Final: {final_fitness:.3f}')
        
        # 在图上添加文本说明
        textstr = f'Improvement: +{improvement:.3f} ({improvement/initial_fitness*100:.1f}%)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    ax.legend(loc='lower right', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # 打印统计信息
    print(f"\nStatistics:")
    print(f"  Total iterations: {len(data)}")
    print(f"  Initial fitness: {fitnesses[0]:.3f}")
    print(f"  Final fitness: {fitnesses[-1]:.3f}")
    print(f"  Improvement: +{fitnesses[-1] - fitnesses[0]:.3f}")
    print(f"  Max fitness: {max(fitnesses):.3f} (iter {iterations[fitnesses.index(max(fitnesses))]})")
    print(f"  Min fitness: {min(fitnesses):.3f} (iter {iterations[fitnesses.index(min(fitnesses))]})")
    
    plt.close()

if __name__ == "__main__":
    checkpoints_dir = '/home/iilab9/scholar-papers/experiments/intention/exp-1/wyf-exp1/results/greedy_checkpoints'
    
    # 加载数据
    data = load_fitness_data(checkpoints_dir)
    
    if data:
        # 绘制图表
        plot_fitness_curve(data, output_path=os.path.join(checkpoints_dir, 'fitness_curve.png'))
        
        # 打印前10条数据
        print("\nFirst 10 data points:")
        for d in data[:10]:
            print(f"  Iter {d['iteration']:2d}: Fitness = {d['fitness']:.3f}")
    else:
        print("No data loaded!")
