import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.game.environment import PathFindingEnv
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from src.agent.learning_agent import LearningAgent
from src.agent.agent_interface import DeepSeekAgent
from src.game.map_generator import MapGenerator
from src.config.paths import AGENT_MEMORY_DIR, MAZE_EVAL_MAPS_DIR, RESULTS_DIR

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估学习型代理的性能")
    parser.add_argument("--api_key", type=str, default=None,
                      help="API密钥")
    parser.add_argument("--model", type=str, default=None,
                      help="使用的模型名称")
    parser.add_argument("--api_type", type=str, default="openai", choices=["deepseek", "openai"],
                      help="API类型，可选deepseek或openai")
    parser.add_argument("--base_url", type=str, default=None,
                      help="API基础URL，仅当api_type=openai时使用")
    parser.add_argument("--maps_dir", type=str, default=None,
                      help="评估地图目录，默认为data/levels/maze_eval")
    parser.add_argument("--memory_dir", type=str, default=None,
                      help="记忆目录，默认为outputs/memory/agent_memory")
    parser.add_argument("--results_dir", type=str, default=None,
                      help="结果保存目录，默认为outputs/results/learning_agent")
    parser.add_argument("--num_maps", type=int, default=10,
                      help="每个难度评估的地图数量，默认为10")
    parser.add_argument("--max_steps", type=int, default=200,
                      help="每个地图的最大步数，默认为200")
    parser.add_argument("--mode", type=str, default=None,
                      choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3],
                      help="只评估指定模式，默认评估所有模式")
    parser.add_argument("--baseline", action="store_true",
                      help="评估基线代理（不使用记忆）")
    parser.add_argument("--compare", action="store_true",
                      help="比较学习代理和基线代理")
    return parser.parse_args()

def load_maps(mode: str, maps_dir: str) -> List[Dict[str, Any]]:
    """加载评估地图"""
    # 设置地图文件路径
    collection_file = os.path.join(maps_dir, f"{mode.replace(' ', '_')}_collection.json")
    
    # 检查地图文件是否存在
    if not os.path.exists(collection_file):
        print(f"找不到地图文件 {collection_file}，生成新地图")
        MapGenerator.generate_maps(30, mode, maps_dir)
    
    # 加载地图
    try:
        with open(collection_file, 'r') as f:
            maps = json.load(f)
        print(f"已加载 {len(maps)} 个 {mode} 模式地图")
        return maps
    except Exception as e:
        print(f"加载地图失败: {str(e)}")
        return []

def run_evaluation_episode(agent, env: PathFindingEnv, map_data: Dict[str, Any], 
                         map_index: int, max_steps: int = 200, use_memory: bool = True) -> Dict[str, Any]:
    """运行一个评估回合"""
    # 加载地图
    env.load_map(map_data, map_index)
    
    # 如果是学习代理，开始会话并设置关卡ID
    if hasattr(agent, 'start_session'):
        agent.start_session(env)
        print(f"使用{'记忆增强' if use_memory else '基础'}代理评估地图 {map_index+1}")
    else:
        agent.set_mode(env.mode)
        print(f"使用基础代理评估地图 {map_index+1}")
    
    # 游戏循环
    done = False
    step_count = 0
    total_reward = 0
    api_calls = 0
    
    while not done and step_count < max_steps:
        # 获取当前状态
        current_state = env.get_state_dict()
        
        # 获取代理动作
        if hasattr(agent, 'get_action'):
            action, response = agent.get_action(current_state, with_memory=use_memory)
            api_calls += 1
            
            # 记录交互
            if hasattr(agent, 'log_interaction'):
                agent.log_interaction(current_state, action, response, 0)  # 奖励暂时为0
        else:
            action, response = agent.get_action(current_state)
            api_calls += 1
        
        # 执行动作
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        step_count += 1
    
    # 计算最终指标
    success = done and reward > 0  # 如果游戏结束且奖励为正，则视为成功
    
    # 计算探索率
    grid_size = env.grid.shape[0]
    total_cells = grid_size * grid_size
    obstacles_count = len(env.obstacles)
    explorable_cells = total_cells - obstacles_count
    discovered_count = np.sum(env.vision_map == DISCOVERED)
    exploration_rate = discovered_count / explorable_cells if explorable_cells > 0 else 0
    
    # 结束会话（如果是学习代理）
    if hasattr(agent, 'end_session'):
        agent.end_session(env, success, env.score)
    
    # 整理评估指标
    metrics = {
        "map_index": map_index,
        "mode": env.mode,
        "steps": step_count,
        "score": env.score,
        "success": success,
        "exploration_rate": exploration_rate,
        "lives_remaining": env.lives,
        "collected_coins": COINS_COUNT - len(env.coins),
        "api_calls": api_calls,
        "agent_type": "learning" if use_memory else "baseline"
    }
    
    return metrics

def evaluate_mode(agent, mode: str, maps_dir: str, results_dir: str, 
                num_maps: int = 10, max_steps: int = 200, use_memory: bool = True) -> List[Dict[str, Any]]:
    """评估指定模式"""
    print(f"\n=============== 评估 {mode} 模式 ===============")
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载地图
    maps = load_maps(mode, maps_dir)
    
    if not maps:
        print(f"无法加载 {mode} 模式的地图，跳过评估")
        return []
    
    # 限制地图数量
    maps = maps[:num_maps]
    
    # 创建环境
    env = PathFindingEnv(mode=mode)
    
    # 评估结果
    results = []
    
    # 使用tqdm显示进度条
    for i in tqdm(range(len(maps)), desc=f"评估 {mode}"):
        map_data = maps[i]
        metrics = run_evaluation_episode(
            agent, env, map_data, i, 
            max_steps=max_steps,
            use_memory=use_memory
        )
        
        results.append(metrics)
        
        # 输出当前结果
        print(f"地图 {i+1} 评估结果:")
        print(f"- 步数: {metrics['steps']}")
        print(f"- 得分: {metrics['score']}")
        print(f"- 通过: {'成功' if metrics['success'] else '失败'}")
        print(f"- 探索率: {metrics['exploration_rate']:.2%}")
    
    # 保存评估结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    agent_type = "learning" if use_memory else "baseline"
    results_file = os.path.join(results_dir, f"{agent_type}_{mode.replace(' ', '_')}_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"评估结果已保存到: {results_file}")
    
    # 计算汇总统计
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_exploration = sum(r["exploration_rate"] for r in results) / len(results)
    
    print(f"\n{mode} 评估汇总:")
    print(f"- 评估地图数: {len(results)}")
    print(f"- 通关率: {success_rate:.2%}")
    print(f"- 平均得分: {avg_score:.1f}")
    print(f"- 平均步数: {avg_steps:.1f}")
    print(f"- 平均探索率: {avg_exploration:.2%}")
    
    # 关闭环境
    env.close()
    
    return results

def compare_agents(learning_results: Dict[str, List[Dict[str, Any]]], 
                  baseline_results: Dict[str, List[Dict[str, Any]]], 
                  results_dir: str):
    """比较学习代理和基线代理的性能"""
    print("\n=============== 代理性能比较 ===============")
    
    # 创建比较结果目录
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 比较结果
    comparison = {}
    
    for mode in learning_results.keys():
        if mode not in baseline_results:
            continue
        
        learning = learning_results[mode]
        baseline = baseline_results[mode]
        
        # 确保两组结果可比较（地图索引相同）
        learning_map_indices = [r["map_index"] for r in learning]
        baseline_map_indices = [r["map_index"] for r in baseline]
        common_indices = set(learning_map_indices).intersection(set(baseline_map_indices))
        
        learning_filtered = [r for r in learning if r["map_index"] in common_indices]
        baseline_filtered = [r for r in baseline if r["map_index"] in common_indices]
        
        # 排序结果，确保相同索引的地图可以直接比较
        learning_filtered.sort(key=lambda r: r["map_index"])
        baseline_filtered.sort(key=lambda r: r["map_index"])
        
        # 计算每个地图的性能差异
        improvements = []
        
        for l, b in zip(learning_filtered, baseline_filtered):
            imp = {
                "map_index": l["map_index"],
                "score_diff": l["score"] - b["score"],
                "score_percent": (l["score"] / b["score"] * 100 - 100) if b["score"] > 0 else float('inf'),
                "steps_diff": b["steps"] - l["steps"],  # 步数减少是改进
                "steps_percent": (1 - l["steps"] / b["steps"]) * 100 if b["steps"] > 0 else 0,
                "exploration_diff": l["exploration_rate"] - b["exploration_rate"],
                "exploration_percent": (l["exploration_rate"] / b["exploration_rate"] * 100 - 100) if b["exploration_rate"] > 0 else float('inf'),
                "success_improved": l["success"] and not b["success"]
            }
            improvements.append(imp)
        
        # 计算汇总统计
        learning_success_rate = sum(r["success"] for r in learning_filtered) / len(learning_filtered)
        baseline_success_rate = sum(r["success"] for r in baseline_filtered) / len(baseline_filtered)
        
        learning_avg_score = sum(r["score"] for r in learning_filtered) / len(learning_filtered)
        baseline_avg_score = sum(r["score"] for r in baseline_filtered) / len(baseline_filtered)
        
        learning_avg_steps = sum(r["steps"] for r in learning_filtered) / len(learning_filtered)
        baseline_avg_steps = sum(r["steps"] for r in baseline_filtered) / len(baseline_filtered)
        
        learning_avg_exploration = sum(r["exploration_rate"] for r in learning_filtered) / len(learning_filtered)
        baseline_avg_exploration = sum(r["exploration_rate"] for r in baseline_filtered) / len(baseline_filtered)
        
        # 整理比较结果
        comparison[mode] = {
            "num_maps": len(common_indices),
            "learning_success_rate": learning_success_rate,
            "baseline_success_rate": baseline_success_rate,
            "success_rate_change": learning_success_rate - baseline_success_rate,
            "success_rate_percent": (learning_success_rate / baseline_success_rate * 100 - 100) if baseline_success_rate > 0 else float('inf'),
            
            "learning_avg_score": learning_avg_score,
            "baseline_avg_score": baseline_avg_score,
            "score_change": learning_avg_score - baseline_avg_score,
            "score_percent": (learning_avg_score / baseline_avg_score * 100 - 100) if baseline_avg_score > 0 else float('inf'),
            
            "learning_avg_steps": learning_avg_steps,
            "baseline_avg_steps": baseline_avg_steps,
            "steps_change": baseline_avg_steps - learning_avg_steps,  # 步数减少是改进
            "steps_percent": (1 - learning_avg_steps / baseline_avg_steps) * 100 if baseline_avg_steps > 0 else 0,
            
            "learning_avg_exploration": learning_avg_exploration,
            "baseline_avg_exploration": baseline_avg_exploration,
            "exploration_change": learning_avg_exploration - baseline_avg_exploration,
            "exploration_percent": (learning_avg_exploration / baseline_avg_exploration * 100 - 100) if baseline_avg_exploration > 0 else float('inf'),
            
            "improvements": improvements
        }
        
        # 打印比较结果
        print(f"\n{mode} 代理比较:")
        print(f"- 比较地图数: {len(common_indices)}")
        print(f"- 通关率: 学习代理 {learning_success_rate:.2%} vs 基线代理 {baseline_success_rate:.2%} (变化: {learning_success_rate - baseline_success_rate:.2%})")
        print(f"- 平均得分: 学习代理 {learning_avg_score:.1f} vs 基线代理 {baseline_avg_score:.1f} (变化: {learning_avg_score - baseline_avg_score:.1f})")
        print(f"- 平均步数: 学习代理 {learning_avg_steps:.1f} vs 基线代理 {baseline_avg_steps:.1f} (变化: {baseline_avg_steps - learning_avg_steps:.1f})")
        print(f"- 平均探索率: 学习代理 {learning_avg_exploration:.2%} vs 基线代理 {baseline_avg_exploration:.2%} (变化: {learning_avg_exploration - baseline_avg_exploration:.2%})")
    
    # 保存比较结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(comparison_dir, f"comparison_{timestamp}.json")
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n比较结果已保存到: {comparison_file}")
    
    # 生成比较图表
    generate_comparison_charts(comparison, comparison_dir)

def generate_comparison_charts(comparison: Dict[str, Dict[str, Any]], output_dir: str):
    """生成比较图表"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for mode, data in comparison.items():
        # 创建得分比较柱状图
        plt.figure(figsize=(10, 6))
        labels = ['通关率', '平均得分', '平均步数', '平均探索率']
        learning_values = [data['learning_success_rate'], data['learning_avg_score']/100, data['learning_avg_steps']/20, data['learning_avg_exploration']]
        baseline_values = [data['baseline_success_rate'], data['baseline_avg_score']/100, data['baseline_avg_steps']/20, data['baseline_avg_exploration']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - width/2, learning_values, width, label='学习代理')
        rects2 = ax.bar(x + width/2, baseline_values, width, label='基线代理')
        
        ax.set_title(f'{mode} 代理性能比较')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # 添加标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{mode.replace(' ', '_')}_comparison_{timestamp}.png"))
        
        # 地图比较散点图
        if 'improvements' in data and data['improvements']:
            improvements = data['improvements']
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            map_indices = [imp['map_index'] for imp in improvements]
            score_diffs = [imp['score_diff'] for imp in improvements]
            success_improved = [imp['success_improved'] for imp in improvements]
            
            # 使用不同颜色表示成功提升
            colors = ['green' if imp else 'blue' for imp in success_improved]
            
            ax.scatter(map_indices, score_diffs, c=colors, alpha=0.7, s=100)
            
            # 添加零线
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            ax.set_title(f'{mode} 地图得分差异 (学习代理 vs 基线代理)')
            ax.set_xlabel('地图索引')
            ax.set_ylabel('得分差异')
            
            plt.savefig(os.path.join(output_dir, f"{mode.replace(' ', '_')}_score_diff_{timestamp}.png"))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估学习型智能体')
    
    # 添加数据相关参数
    parser.add_argument('--maps_dir', type=str, default=None,
                        help='评估地图目录路径 (默认: data/eval_maps)')
    parser.add_argument('--memory_dir', type=str, default=None,
                        help='记忆保存目录路径 (默认: outputs/memory/agent_memory)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果保存目录路径 (默认: outputs/results/evaluation)')
    
    # 添加评估控制参数
    parser.add_argument('--max_steps', type=int, default=100,
                        help='每个地图的最大步数限制')
    parser.add_argument('--mode', type=str, choices=['level1', 'level2', 'level3', 'all'], default='all',
                        help='评估的游戏模式')
    parser.add_argument('--num_maps', type=int, default=5,
                        help='每个难度评估的地图数量')
    parser.add_argument('--compare', action='store_true',
                        help='是否比较使用记忆和不使用记忆的效果')
    
    # 添加API相关参数
    parser.add_argument('--api_type', type=str, choices=['deepseek', 'openai'], default='deepseek',
                        help='使用的API类型')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API密钥，如果不提供则使用环境变量')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API基础URL，如果不提供则使用默认值')
    parser.add_argument('--model', type=str, default=None,
                        help='使用的模型名称，如果不提供则使用默认值')
    
    args = parser.parse_args()
    
    # 设置默认路径
    maps_dir = args.maps_dir or str(MAZE_EVAL_MAPS_DIR)
    memory_dir = args.memory_dir or str(AGENT_MEMORY_DIR)
    output_dir = args.output_dir or os.path.join(str(RESULTS_DIR), 'evaluation')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置API密钥（优先使用命令行参数，其次使用环境变量）
    api_key = args.api_key
    if api_key is None:
        if args.api_type == 'deepseek':
            api_key = os.environ.get('DEEPSEEK_API_KEY')
        else:
            api_key = os.environ.get('OPENAI_API_KEY')
    
    if api_key is None:
        print(f"错误: 未提供API密钥且未在环境变量中找到对应的API密钥")
        sys.exit(1)
    
    # 创建学习代理
    agent = LearningAgent(
        api_key=api_key,
        model=args.model,
        api_type=args.api_type,
        base_url=args.base_url,
        memory_dir=memory_dir
    )
    
    # 根据指定的模式进行评估
    all_results = []
    
    if args.mode == 'level1' or args.mode == 'all':
        level_results = evaluate_mode(
            agent=agent,
            mode=MODE_LEVEL1,
            maps_dir=maps_dir,
            results_dir=output_dir,
            num_maps=args.num_maps,
            max_steps=args.max_steps,
            use_memory=True
        )
        all_results.extend(level_results)
    
    if args.mode == 'level2' or args.mode == 'all':
        level_results = evaluate_mode(
            agent=agent,
            mode=MODE_LEVEL2,
            maps_dir=maps_dir,
            results_dir=output_dir,
            num_maps=args.num_maps,
            max_steps=args.max_steps,
            use_memory=True
        )
        all_results.extend(level_results)
    
    if args.mode == 'level3' or args.mode == 'all':
        level_results = evaluate_mode(
            agent=agent,
            mode=MODE_LEVEL3,
            maps_dir=maps_dir,
            results_dir=output_dir,
            num_maps=args.num_maps,
            max_steps=args.max_steps,
            use_memory=True
        )
        all_results.extend(level_results)
    
    # 保存结果
    save_results(all_results, output_dir)
    
    print(f"评估完成！结果已保存到: {output_dir}")

if __name__ == "__main__":
    main() 
