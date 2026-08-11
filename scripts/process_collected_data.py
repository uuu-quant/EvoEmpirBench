import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import Counter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_root)

from src.config.paths import COLLECTED_DATA_DIR, PROCESSED_DATA_DIR

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="处理收集的游戏数据，为训练奖励模型做准备")
    parser.add_argument("--data_dir", type=str, default=None,
                      help="数据目录路径，默认为'outputs/collected_data'")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="输出目录路径，默认为'outputs/processed_data'")
    parser.add_argument("--file", type=str, default=None,
                      help="指定要处理的单个数据文件名，默认处理所有文件")
    parser.add_argument("--visualize", action="store_true",
                      help="生成数据可视化图表")
    return parser.parse_args()

def load_data(file_path: str) -> Dict[str, Any]:
    """
    加载收集的数据文件
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        加载的数据字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {str(e)}")
        return {}

def extract_state_action_reward(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    提取状态-动作-奖励数据，用于训练奖励模型
    
    Args:
        data: 加载的数据字典
        
    Returns:
        状态-动作-奖励数据列表
    """
    sar_data = []
    
    for episode in data.get("episodes", []):
        map_index = episode.get("map_index", -1)
        outcome = episode.get("outcome", "unknown")
        
        for step in episode.get("steps", []):
            entry = {
                "map_index": map_index,
                "step_index": step.get("step_index", -1),
                "action": step.get("action", -1),
                "reward": step.get("reward", 0),
                "state_repr": _create_state_representation(step.get("state_before", {})),
                "reasoning": step.get("reasoning", ""),
                "immediate_outcome": _evaluate_step_outcome(step),
                "episode_outcome": 1 if outcome == "success" else 0,
                "done": step.get("done", False)
            }
            sar_data.append(entry)
    
    return sar_data

def _create_state_representation(state: Dict[str, Any]) -> str:
    """
    创建状态的文本表示，用于训练奖励模型
    
    Args:
        state: 游戏状态字典
        
    Returns:
        状态的文本表示
    """
    # 这个函数可以根据需要定制，创建适合奖励模型训练的状态表示
    # 例如，可以将网格转换为ASCII表示等
    
    # 简单示例：提取关键信息
    if not state:
        return "Unknown state"
    
    agent_pos = state.get("agent_pos", (-1, -1))
    lives = state.get("lives", 0)
    score = state.get("score", 0)
    coins_count = len(state.get("coins", []))
    
    return f"Position: {agent_pos}, Lives: {lives}, Score: {score}, Coins: {coins_count}"

def _evaluate_step_outcome(step: Dict[str, Any]) -> int:
    """
    评估单步的结果质量，用于训练奖励模型
    
    Args:
        step: 步骤数据
        
    Returns:
        结果评分 (0-5)
    """
    # 这是一个简单的评估函数，可以根据需要定制
    # 0: 非常差的结果 (如失去生命)
    # 1: 较差的结果 (如没有新探索，但消耗了步数)
    # 2: 一般的结果 (有少量新探索)
    # 3: 较好的结果 (有大量新探索)
    # 4: 很好的结果 (收集金币)
    # 5: 极好的结果 (到达目标)
    
    if step.get("done", False) and step.get("reward", 0) > 0:
        return 5  # 到达目标
    
    state_diff = step.get("state_diff", {})
    
    # 检查生命损失
    if state_diff.get("lives_delta", 0) < 0:
        return 0  # 失去生命
    
    # 检查金币收集
    if state_diff.get("coins_collected_count", 0) > 0:
        return 4  # 收集了金币
    
    # 检查探索情况
    newly_explored = state_diff.get("newly_explored_count", 0)
    if newly_explored > 5:
        return 3  # 大量新探索
    elif newly_explored > 0:
        return 2  # 少量新探索
    else:
        return 1  # 没有新探索，但消耗了步数

def create_paired_comparisons(sar_data: List[Dict[str, Any]], 
                             sample_count: int = 1000) -> List[Dict[str, Any]]:
    """
    创建成对比较数据，用于训练奖励模型
    
    Args:
        sar_data: 状态-动作-奖励数据
        sample_count: 采样的比较对数量
        
    Returns:
        成对比较数据列表
    """
    import random
    comparisons = []
    
    # 按地图索引和步骤索引对数据进行分组
    grouped_data = {}
    for entry in sar_data:
        map_idx = entry["map_index"]
        step_idx = entry["step_index"]
        key = f"{map_idx}_{step_idx}"
        
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(entry)
    
    # 提取有多个动作选择的步骤
    multi_choice_steps = [data for key, data in grouped_data.items() if len(data) > 1]
    
    # 如果没有足够的多选步骤，就从不同步骤之间创建比较
    if len(multi_choice_steps) < sample_count // 2:
        # 对每个条目，找到不太好的和比它好的例子
        for i, entry in enumerate(sar_data):
            if i % 10 != 0:  # 只采样10%的数据以提高效率
                continue
                
            cur_outcome = entry["immediate_outcome"]
            
            # 寻找比当前更好和更差的例子
            better_examples = [e for e in sar_data if e["immediate_outcome"] > cur_outcome]
            worse_examples = [e for e in sar_data if e["immediate_outcome"] < cur_outcome]
            
            if better_examples and random.random() < 0.7:  # 70%的概率创建更好比较
                better = random.choice(better_examples)
                comparisons.append({
                    "state": entry["state_repr"],
                    "action_a": entry["action"],
                    "reasoning_a": entry["reasoning"],
                    "action_b": better["action"],
                    "reasoning_b": better["reasoning"],
                    "better_choice": "B",  # B是更好的选择
                    "quality_diff": better["immediate_outcome"] - entry["immediate_outcome"]
                })
            
            if worse_examples and random.random() < 0.3:  # 30%的概率创建更差比较
                worse = random.choice(worse_examples)
                comparisons.append({
                    "state": entry["state_repr"],
                    "action_a": entry["action"],
                    "reasoning_a": entry["reasoning"],
                    "action_b": worse["action"],
                    "reasoning_b": worse["reasoning"],
                    "better_choice": "A",  # A是更好的选择
                    "quality_diff": entry["immediate_outcome"] - worse["immediate_outcome"]
                })
    else:
        # 从多选步骤中创建比较
        for step_entries in multi_choice_steps:
            if len(step_entries) < 2:
                continue
                
            # 按照结果质量排序
            sorted_entries = sorted(step_entries, key=lambda x: x["immediate_outcome"])
            
            if len(sorted_entries) >= 2:
                worst = sorted_entries[0]
                best = sorted_entries[-1]
                
                # 确保有质量差异
                if best["immediate_outcome"] > worst["immediate_outcome"]:
                    comparisons.append({
                        "state": best["state_repr"],
                        "action_a": worst["action"],
                        "reasoning_a": worst["reasoning"],
                        "action_b": best["action"],
                        "reasoning_b": best["reasoning"],
                        "better_choice": "B",  # B是更好的选择
                        "quality_diff": best["immediate_outcome"] - worst["immediate_outcome"]
                    })
    
    # 随机采样以控制数据量
    if len(comparisons) > sample_count:
        return random.sample(comparisons, sample_count)
    return comparisons

def generate_visualizations(sar_data: List[Dict[str, Any]], output_path: str) -> None:
    """
    生成数据可视化图表
    
    Args:
        sar_data: 状态-动作-奖励数据
        output_path: 输出路径
    """
    if not sar_data:
        print("没有数据可供可视化")
        return
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 1. 动作分布直方图
    action_counts = Counter([entry["action"] for entry in sar_data])
    
    plt.figure(figsize=(10, 6))
    actions = sorted(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    
    plt.bar(actions, counts)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Distribution of Actions')
    plt.xticks(actions)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'action_distribution.png'))
    plt.close()
    
    # 2. 步骤结果质量分布
    outcome_counts = Counter([entry["immediate_outcome"] for entry in sar_data])
    
    plt.figure(figsize=(10, 6))
    outcomes = sorted(outcome_counts.keys())
    counts = [outcome_counts[o] for o in outcomes]
    
    plt.bar(outcomes, counts)
    plt.xlabel('Outcome Quality')
    plt.ylabel('Count')
    plt.title('Distribution of Step Outcomes')
    plt.xticks(outcomes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'outcome_distribution.png'))
    plt.close()
    
    # 3. 每个回合的步数统计
    episodes = {}
    for entry in sar_data:
        map_idx = entry["map_index"]
        step_idx = entry["step_index"]
        
        if map_idx not in episodes:
            episodes[map_idx] = set()
        episodes[map_idx].add(step_idx)
    
    episode_steps = [len(steps) for steps in episodes.values()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(episode_steps, bins=10, edgecolor='black')
    plt.xlabel('Steps per Episode')
    plt.ylabel('Frequency')
    plt.title('Distribution of Steps per Episode')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'steps_per_episode.png'))
    plt.close()
    
    print(f"可视化图表已保存到: {output_path}")

def process_data_file(file_path: str, output_dir: str, visualize: bool = False) -> None:
    """
    处理单个数据文件
    
    Args:
        file_path: 数据文件路径
        output_dir: 输出目录
        visualize: 是否生成可视化
    """
    print(f"处理文件: {file_path}")
    
    # 加载数据
    data = load_data(file_path)
    if not data:
        return
    
    # 提取文件名（不含扩展名）作为数据集标识
    dataset_id = os.path.splitext(os.path.basename(file_path))[0]
    
    # 创建输出目录
    dataset_output_dir = os.path.join(output_dir, dataset_id)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 提取状态-动作-奖励数据
    sar_data = extract_state_action_reward(data)
    print(f"提取了 {len(sar_data)} 条状态-动作-奖励数据")
    
    # 保存SAR数据
    sar_path = os.path.join(dataset_output_dir, "sar_data.json")
    with open(sar_path, 'w', encoding='utf-8') as f:
        json.dump(sar_data, f, ensure_ascii=False, indent=2)
    print(f"SAR数据已保存到: {sar_path}")
    
    # 创建成对比较数据
    comparisons = create_paired_comparisons(sar_data)
    print(f"创建了 {len(comparisons)} 条成对比较数据")
    
    # 保存比较数据
    comparisons_path = os.path.join(dataset_output_dir, "comparisons.json")
    with open(comparisons_path, 'w', encoding='utf-8') as f:
        json.dump(comparisons, f, ensure_ascii=False, indent=2)
    print(f"比较数据已保存到: {comparisons_path}")
    
    # 生成训练集和验证集分割
    import random
    random.shuffle(comparisons)
    split_idx = int(len(comparisons) * 0.8)
    train_data = comparisons[:split_idx]
    val_data = comparisons[split_idx:]
    
    # 保存训练集和验证集
    train_path = os.path.join(dataset_output_dir, "train.json")
    val_path = os.path.join(dataset_output_dir, "val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练集 ({len(train_data)} 条) 已保存到: {train_path}")
    print(f"验证集 ({len(val_data)} 条) 已保存到: {val_path}")
    
    # 生成可视化图表
    if visualize:
        viz_dir = os.path.join(dataset_output_dir, "visualizations")
        generate_visualizations(sar_data, viz_dir)

def main():
    args = parse_args()
    
    if args.data_dir is None:
        args.data_dir = str(COLLECTED_DATA_DIR)
    
    if args.output_dir is None:
        args.output_dir = str(PROCESSED_DATA_DIR)
    
    # 确保目录存在
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 处理文件
    if args.file:
        file_path = os.path.join(args.data_dir, args.file)
        if os.path.exists(file_path):
            process_data_file(file_path, args.output_dir, args.visualize)
        else:
            print(f"Error: 文件 {file_path} 不存在")
    else:
        # 处理目录中的所有JSON文件
        files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
        if not files:
            print(f"Error: 在 {args.data_dir} 中未找到JSON文件")
            return
        
        print(f"发现 {len(files)} 个数据文件")
        for file in files:
            file_path = os.path.join(args.data_dir, file)
            process_data_file(file_path, args.output_dir, args.visualize)
    
    print("\n数据处理完成!")

if __name__ == "__main__":
    main() 
