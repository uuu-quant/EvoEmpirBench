import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.game.environment import PathFindingEnv
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from src.agent.learning_agent import LearningAgent
from src.game.map_generator import MapGenerator
from src.config.paths import AGENT_MEMORY_DIR, MAZE_EVAL_MAPS_DIR, MAZE_TRAIN_MAPS_DIR

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练学习型代理，使用记忆和真理模块改进性能")
    parser.add_argument("--api_key", type=str, default=None,
                      help="API密钥")
    parser.add_argument("--model", type=str, default=None,
                      help="使用的模型名称")
    parser.add_argument("--api_type", type=str, default="openai", choices=["deepseek", "openai"],
                      help="API类型，可选deepseek或openai")
    parser.add_argument("--base_url", type=str, default=None,
                      help="API基础URL，仅当api_type=openai时使用")
    parser.add_argument("--maps_dir", type=str, default=None,
                      help="地图目录，默认为data/levels/maze_train")
    parser.add_argument("--memory_dir", type=str, default=None,
                      help="记忆保存目录，默认为outputs/memory/agent_memory")
    parser.add_argument("--num_maps", type=int, default=5,
                      help="每个难度训练的地图数量，默认为5")
    parser.add_argument("--max_attempts", type=int, default=3,
                      help="每个地图最大尝试次数，默认为3")
    parser.add_argument("--max_steps", type=int, default=200,
                      help="每次尝试的最大步数，默认为200")
    parser.add_argument("--mode", type=str, default=None,
                      choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3],
                      help="只训练指定模式，默认训练所有模式")
    parser.add_argument("--start_map", type=int, default=0,
                      help="从指定索引的地图开始训练，默认为0")
    parser.add_argument("--clear_memory", action="store_true",
                      help="清除所有已有的记忆和真理知识")
    parser.add_argument("--skip_validation", action="store_true",
                      help="跳过验证步骤，直接将主观记忆提升为真理")
    return parser.parse_args()

def load_maps(mode: str, maps_dir: str) -> List[Dict[str, Any]]:
    """
    加载指定模式的地图
    
    Args:
        mode: 游戏模式
        maps_dir: 地图目录
        
    Returns:
        地图列表
    """
    # 设置地图文件路径
    collection_file = os.path.join(maps_dir, f"{mode.replace(' ', '_')}_collection.json")
    
    # 检查地图文件是否存在
    if not os.path.exists(collection_file):
        print(f"找不到地图文件 {collection_file}，使用默认地图")
        # 使用默认地图目录
        default_maps_dir = str(MAZE_EVAL_MAPS_DIR)
        collection_file = os.path.join(default_maps_dir, f"{mode.replace(' ', '_')}_collection.json")
        
        if not os.path.exists(collection_file):
            print(f"找不到默认地图文件，生成新地图")
            MapGenerator.generate_maps(30, mode, default_maps_dir)
            collection_file = os.path.join(default_maps_dir, f"{mode.replace(' ', '_')}_collection.json")
    
    # 加载地图
    try:
        with open(collection_file, 'r') as f:
            maps = json.load(f)
        print(f"已加载 {len(maps)} 个 {mode} 模式地图")
        return maps
    except Exception as e:
        print(f"加载地图失败: {str(e)}")
        return []

def run_episode(agent: LearningAgent, env: PathFindingEnv, map_data: Dict[str, Any], 
               map_index: int, max_steps: int = 200, use_memory: bool = True) -> Dict[str, Any]:
    """
    运行一个游戏回合
    
    Args:
        agent: 学习代理
        env: 游戏环境
        map_data: 地图数据
        map_index: 地图索引
        max_steps: 最大步数
        use_memory: 是否使用记忆
        
    Returns:
        回合指标
    """
    # 加载地图
    env.load_map(map_data, map_index)
    
    # 开始会话
    agent.start_session(env)
    
    # 游戏循环
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # 获取当前状态
        current_state = env.get_state_dict()
        
        # 获取代理动作
        action, response = agent.get_action(current_state, with_memory=use_memory)
        
        # 执行动作
        _, reward, done, _, _ = env.step(action)
        
        # 记录交互
        agent.log_interaction(current_state, action, response, reward)
        
        step_count += 1
    
    # 结束会话
    success = done and reward > 0  # 如果游戏结束且奖励为正，则视为成功
    metrics = agent.end_session(env, success, env.score)
    
    return metrics

def train_on_map(agent: LearningAgent, env: PathFindingEnv, map_data: Dict[str, Any], 
              max_episode_steps: int = 100, with_memory: bool = True) -> bool:
    """
    在特定地图上训练代理
    
    Args:
        agent: 学习代理
        env: 游戏环境
        map_data: 地图数据
        max_episode_steps: 最大步数
        with_memory: 是否使用记忆增强
        
    Returns:
        是否验证成功
    """
    # 加载地图
    observation, _ = env.load_map_from_dict(map_data)
    
    # 开始第一轮：初始体验
    print(f"\n=== 开始关卡 {agent._get_level_id(env)} 的初始体验 ===")
    agent.start_session(env)
    
    # 记录初始指标
    initial_metrics = {}
    
    # 游戏主循环
    done = False
    step_count = 0
    
    while not done and step_count < max_episode_steps:
        # 获取代理的动作 - 初始体验阶段可以使用真理知识，但不使用当前关卡的主观记忆
        action, response = agent.get_action(
            env.get_state_dict(), 
            with_memory=True  # 允许使用真理知识（全局记忆池）
        )
        
        # 执行动作
        observation, reward, terminated, truncated, _ = env.step(action)
        
        # 记录交互
        agent.log_interaction(env.get_state_dict(), action, response, reward)
        
        done = terminated or truncated
        step_count += 1
    
    # 结束会话并获取结果
    initial_metrics = agent.end_session(env, done and env.agent_pos == GOAL_POS, env.score)
    
    # 反思游戏会话，生成经验总结、优点和缺点
    experience_summary, strengths, weaknesses = agent.reflect_on_session()
    
    # 保存为主观记忆
    agent.record_subjective_memory(experience_summary, strengths, weaknesses)
    
    # 重置环境，准备第二轮：验证体验
    observation, _ = env.load_map_from_dict(map_data)
    print(f"\n=== 开始关卡 {agent._get_level_id(env)} 的验证体验 ===")
    agent.start_session(env)
    
    # 游戏主循环
    done = False
    step_count = 0
    
    while not done and step_count < max_episode_steps:
        # 获取代理的动作，使用真理知识+当前关卡的主观记忆
        action, response = agent.get_action(
            env.get_state_dict(), 
            with_memory=with_memory  # 使用全部记忆（真理知识+当前关卡主观记忆）
        )
        
        # 执行动作
        observation, reward, terminated, truncated, _ = env.step(action)
        
        # 记录交互
        agent.log_interaction(env.get_state_dict(), action, response, reward)
        
        done = terminated or truncated
        step_count += 1
    
    # 结束会话并获取结果
    validation_metrics = agent.end_session(env, done and env.agent_pos == GOAL_POS, env.score)
    
    # 验证主观记忆的有效性
    is_valid = agent.validate_subjective_memory(initial_metrics, validation_metrics)
    
    # 如果验证有效，将主观记忆提升为真理知识
    if is_valid:
        agent.promote_memory_to_truth()
    else:
        print(f"关卡 {agent._get_level_id(env)} 的主观记忆验证失败，不提升为真理知识")
        agent.clear_current_subjective_memory()
    
    return is_valid

def train_mode(agent: LearningAgent, maps_dir: str, mode: str, maps_count: int = 10, 
             with_memory: bool = True, max_episode_steps: int = 100):
    """
    在指定模式上训练代理
    
    Args:
        agent: 学习代理
        maps_dir: 地图目录
        mode: 游戏模式
        maps_count: 地图数量
        with_memory: 是否使用记忆增强
        max_episode_steps: 最大步数
    """
    print(f"=== 开始在 {mode} 模式上训练 ===")
    
    # 创建环境
    env = PathFindingEnv(mode=mode)
    
    # 修改地图文件路径，以匹配生成的文件名格式
    mode_file_name = mode.replace(" ", "_")  # 将空格替换为下划线
    collection_file = os.path.join(maps_dir, f"{mode_file_name}_collection.json")
    
    # 尝试加载地图
    try:
        if os.path.exists(collection_file):
            with open(collection_file, 'r') as f:
                all_maps = json.load(f)
            print(f"已加载 {len(all_maps)} 个 {mode} 地图")
        else:
            print(f"找不到地图文件: {collection_file}")
            print(f"将尝试生成新的地图...")
            
            # 如果地图文件不存在，尝试生成新地图
            os.makedirs(maps_dir, exist_ok=True)
            from src.game.map_generator import MapGenerator
            all_maps = MapGenerator.generate_maps(maps_count, mode, maps_dir)
            print(f"已生成 {len(all_maps)} 个 {mode} 地图")
    except Exception as e:
        print(f"加载地图列表失败: {str(e)}")
        return
    
    # 限制地图数量
    all_maps = all_maps[:maps_count]
    
    # 记录成功验证的地图数量
    successful_validations = 0
    
    # 遍历每张地图
    for i, map_data in enumerate(tqdm(all_maps, desc=f"训练 {mode}")):
        # 在每张地图上训练
        is_successful = train_on_map(
            agent=agent,
            env=env,
            map_data=map_data,
            max_episode_steps=max_episode_steps,
            with_memory=with_memory
        )
        
        if is_successful:
            successful_validations += 1
    
    print(f"=== 在 {mode} 模式上完成训练 ===")
    print(f"总地图数: {len(all_maps)}")
    print(f"成功验证: {successful_validations}")
    print(f"验证率: {successful_validations / len(all_maps) * 100:.2f}%")
    
    # 关闭环境
    env.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练学习型智能体')
    
    # 添加数据相关参数
    parser.add_argument('--maps_dir', type=str, default=str(MAZE_TRAIN_MAPS_DIR),
                        help='训练地图目录路径')
    parser.add_argument('--memory_dir', type=str, default=str(AGENT_MEMORY_DIR),
                        help='记忆保存目录路径')
    
    # 添加训练控制参数
    parser.add_argument('--max_steps', type=int, default=100,
                        help='每个地图的最大步数限制')
    parser.add_argument('--mode', type=str, choices=['level1', 'level2', 'level3', 'all'], default='all',
                        help='训练的游戏模式')
    parser.add_argument('--num_maps', type=int, default=10,
                        help='每个难度使用的地图数量')
    
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
        memory_dir=args.memory_dir
    )
    
    # 加载训练地图数据
    maps_directory = args.maps_dir
    if not os.path.exists(maps_directory):
        print(f"错误: 找不到训练地图目录: {maps_directory}")
        sys.exit(1)
    
    # 根据指定的模式进行训练
    if args.mode == 'level1' or args.mode == 'all':
        train_mode(
            agent=agent,
            mode=MODE_LEVEL1,
            maps_dir=maps_directory,
            max_episode_steps=args.max_steps,
            maps_count=args.num_maps,
            with_memory=True
        )
    
    if args.mode == 'level2' or args.mode == 'all':
        train_mode(
            agent=agent,
            mode=MODE_LEVEL2,
            maps_dir=maps_directory,
            max_episode_steps=args.max_steps,
            maps_count=args.num_maps,
            with_memory=True
        )
    
    if args.mode == 'level3' or args.mode == 'all':
        train_mode(
            agent=agent,
            mode=MODE_LEVEL3,
            maps_dir=maps_directory,
            max_episode_steps=args.max_steps,
            maps_count=args.num_maps,
            with_memory=True
        )
    
    print("所有训练完成！")

if __name__ == "__main__":
    main() 
