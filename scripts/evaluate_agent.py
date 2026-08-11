import os
import sys
import json
import time
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
import random
import glob
import re

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.game.environment import PathFindingEnv
from src.game.map_generator import MapGenerator
from src.agent.agent_interface import DeepSeekAgent
from src.agent.gpt_client import GPTClient
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3, DISCOVERED, COINS_COUNT, INITIAL_LIVES
from src.config.api_config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, DEFAULT_API_TYPE
from src.config.paths import MAZE_EVAL_MAPS_DIR, RESULTS_DIR

class AgentEvaluator:
    """评估AI代理在不同难度地图上的表现"""
    
    def __init__(self, api_key: str = None, model: str = None, api_type: str = DEFAULT_API_TYPE,
                 base_url: str = None, maps_dir: str = None, results_dir: str = None):
        """
        初始化评估器
        
        Args:
            api_key: API密钥
            model: 使用的模型名称
            api_type: API类型，"deepseek"或"openai"
            base_url: API基础URL，仅当api_type="openai"时使用
            maps_dir: 地图目录
            results_dir: 结果保存目录
        """
        # 根据API类型设置默认值
        if api_type.lower() == "deepseek":
            self.api_key = api_key if api_key else DEEPSEEK_API_KEY
            self.model = model if model else DEEPSEEK_MODEL
            self.base_url = None
        else:  # openai
            self.api_key = api_key if api_key else OPENAI_API_KEY
            self.model = model if model else OPENAI_MODEL
            self.base_url = base_url if base_url else OPENAI_BASE_URL
        
        self.api_type = api_type.lower()
        
        # 设置地图目录
        if maps_dir is None:
            self.maps_dir = str(MAZE_EVAL_MAPS_DIR)
        else:
            self.maps_dir = maps_dir
            
        # 创建地图目录
        os.makedirs(self.maps_dir, exist_ok=True)
        
        # 设置结果保存目录
        if results_dir is None:
            # 使用标准路径格式：outputs/results/{model}/game1
            self.results_dir = os.path.join(str(RESULTS_DIR), self.model, 'game1')
        else:
            self.results_dir = results_dir
            
        # 创建结果保存目录和日志目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'interaction_logs'), exist_ok=True)
        
        # 检查点文件路径
        self.checkpoint_file = os.path.join(self.results_dir, 'evaluation_checkpoint.json')
        
        # 初始化代理
        self.agent = None
        
        # 游戏环境
        self.env = None
        
        # 评估结果
        self.results = self._load_checkpoint() or {
            MODE_LEVEL1: [],
            MODE_LEVEL2: [],
            MODE_LEVEL3: []
        }
    
    def _load_checkpoint(self) -> Dict:
        """加载检查点文件和评估结果文件"""
        results = None
        
        # 首先尝试加载检查点文件
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                print(f"发现检查点文件，加载已有评估结果...")
                results = checkpoint.get('results', None)
            except Exception as e:
                print(f"加载检查点文件失败: {str(e)}")
        
        # 如果检查点文件没有提供结果，尝试从最新的评估结果文件中获取
        if not results:
            report_pattern = os.path.join(self.results_dir, "evaluation_results_*.json")
            report_files = glob.glob(report_pattern)
            
            if report_files:
                print(f"发现 {len(report_files)} 个评估结果文件")
                # 按文件名排序，取最新的
                latest_report = sorted(report_files)[-1]
                try:
                    print(f"尝试从最新的评估结果文件加载: {os.path.basename(latest_report)}")
                    with open(latest_report, 'r') as f:
                        report_data = json.load(f)
                    if 'results' in report_data:
                        results = report_data['results']
                        print(f"从评估结果文件加载了结果，包含以下模式: {list(results.keys())}")
                except Exception as e:
                    print(f"加载评估结果文件失败: {str(e)}")
        
        return results
    
    def _save_checkpoint(self, mode: str, current_map_index: int):
        """保存检查点，确保所有数据都是JSON可序列化的"""
        def convert_to_serializable(obj):
            """将numpy类型和其他特殊类型转换为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        try:
            checkpoint = {
                'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                'model': self.model,
                'api_type': self.api_type,
                'last_mode': mode,
                'last_map_index': current_map_index,
                'results': convert_to_serializable(self.results)
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"\n检查点已保存: {mode} - 地图 {current_map_index + 1}")
        except Exception as e:
            print(f"保存检查点失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _get_completed_maps(self, mode: str) -> set:
        """获取已完成评估的地图索引"""
        completed = set()
        
        # 从检查点文件获取已完成的地图索引
        if mode in self.results:
            for result in self.results[mode]:
                completed.add(result['map_index'])
        
        # 检查已有的日志文件，查找已完成评估的地图
        log_dir = os.path.join(self.results_dir, 'interaction_logs')
        if os.path.exists(log_dir):
            # 改进：使用精确的模式匹配，处理模式名中的空格
            log_pattern = f"{mode}_map*_log.jsonl"
            log_files = glob.glob(os.path.join(log_dir, log_pattern))
            
            # 如果没找到，尝试替换空格为下划线的模式
            if not log_files:
                log_pattern = f"{mode.replace(' ', '_')}_map*_log.jsonl"
                log_files = glob.glob(os.path.join(log_dir, log_pattern))
            
            # 调试输出
            print(f"在 {log_dir} 中搜索日志文件模式 '{log_pattern}'")
            print(f"找到的日志文件: {log_files}")
            
            for log_file in log_files:
                # 从文件名提取地图索引
                map_index_match = re.search(r'map(\d+)_log', os.path.basename(log_file))
                if map_index_match:
                    try:
                        map_index = int(map_index_match.group(1)) - 1  # 转换为0基索引
                        
                        # 检查文件是否完整（包含成功或失败的结果）
                        file_complete = False
                        try:
                            with open(log_file, 'r') as f:
                                # 读取最后几行判断是否完成
                                lines = f.readlines()
                                if lines:
                                    # 检查最后5行中是否包含完成标志
                                    check_lines = lines[-5:] if len(lines) >= 5 else lines
                                    for line in check_lines:
                                        try:
                                            line_data = json.loads(line)
                                            if 'api_response' in line_data and 'completion' in line_data['api_response']:
                                                # 检查是否有成功/失败信息
                                                text = line_data['api_response']['completion']
                                                if text and ('成功' in text or '失败' in text or '通关' in text or
                                                            'success' in text.lower() or 'fail' in text.lower() or
                                                            'complete' in text.lower() or 'done' in text.lower()):
                                                    file_complete = True
                                                    break
                                        except json.JSONDecodeError:
                                            continue
                        except Exception as e:
                            print(f"读取日志文件失败: {str(e)}")
                        
                        if file_complete:
                            completed.add(map_index)
                            print(f"从日志文件识别到已完成的地图: {mode} - 地图 {map_index + 1}")
                    except ValueError:
                        pass
        
        # 额外检查：查找实际生成的报告文件
        report_pattern = os.path.join(self.results_dir, "evaluation_results_*.json")
        report_files = glob.glob(report_pattern)
        if report_files:
            latest_report = sorted(report_files)[-1]
            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                    if 'results' in report_data and mode in report_data['results']:
                        for result in report_data['results'][mode]:
                            if 'map_index' in result:
                                map_idx = result['map_index']
                                completed.add(map_idx)
                                print(f"从报告文件识别到已完成的地图: {mode} - 地图 {map_idx + 1}")
            except Exception as e:
                print(f"读取报告文件失败: {str(e)}")
        
        print(f"模式 {mode} 已完成评估的地图数量: {len(completed)}")
        return completed
    
    def _initialize_agent(self):
        """初始化代理"""
        if self.agent is None:
            try:
                self.agent = DeepSeekAgent(
                    api_key=self.api_key, 
                    model=self.model,
                    api_type=self.api_type,
                    base_url=self.base_url
                )
                self.agent.set_show_prompt(False)  # 评估时不显示提示
                print(f"已初始化AI代理，使用{self.api_type.upper()} API，模型: {self.model}")
            except Exception as e:
                raise RuntimeError(f"初始化代理失败: {str(e)}")
    
    def _load_maps(self, mode: str) -> List[Dict[str, Any]]:
        """加载地图集合"""
        collection_file = os.path.join(self.maps_dir, f"{mode.replace(' ', '_')}_collection.json")
        
        # 检查地图文件是否存在
        if not os.path.exists(collection_file):
            print(f"找不到地图文件 {collection_file}，正在生成地图...")
            
            # 生成地图
            try:
                MapGenerator.generate_maps(30, mode, self.maps_dir)
                print(f"地图生成成功！")
            except Exception as e:
                raise RuntimeError(f"地图生成失败: {str(e)}")
        
        # 加载地图集合
        maps = MapGenerator.load_map_collection(collection_file)
        
        # 确保有足够的地图
        if len(maps) < 30:
            print(f"地图数量不足30个，正在生成更多地图...")
            try:
                new_maps = MapGenerator.generate_maps(30 - len(maps), mode, self.maps_dir)
                maps.extend(new_maps)
                
                # 保存新的地图集合
                with open(collection_file, 'w') as f:
                    json.dump(maps, f, indent=2)
                
                print(f"已生成新地图，总地图数: {len(maps)}")
            except Exception as e:
                raise RuntimeError(f"生成更多地图失败: {str(e)}")
        
        # 截取前30个地图
        return maps[:30]
    
    def _compute_exploration_rate(self, env: PathFindingEnv) -> float:
        """计算探索率 (已探索区域 / 总区域)"""
        grid_size = env.grid.shape[0]
        total_cells = grid_size * grid_size
        obstacles_count = len(env.obstacles)
        explorable_cells = total_cells - obstacles_count
        
        # 统计已探索的单元格数量
        discovered_count = np.sum(env.vision_map == DISCOVERED)
        
        # 计算探索率
        exploration_rate = discovered_count / explorable_cells if explorable_cells > 0 else 0
        
        return exploration_rate
    
    def _extract_game_state(self, env: PathFindingEnv) -> Dict[str, Any]:
        """从环境中提取游戏状态，确保所有数据都是JSON可序列化的"""
        try:
            # 调试信息
            print(f"\n调试信息:")
            print(f"金币类型: {type(env.coins)}")
            if env.coins:  # 如果集合不为空
                first_coin = next(iter(env.coins))  # 获取集合中的第一个元素
                print(f"第一个金币类型: {type(first_coin)}")
                print(f"第一个金币值: {first_coin}")
            
            print(f"网格类型: {type(env.grid)}")
            print(f"视野地图类型: {type(env.vision_map)}")
            
            # 确保网格和视野地图是numpy数组
            grid = np.array(env.grid)
            vision_map = np.array(env.vision_map)
            
            # 计算最新分数
            discovered_count = int(np.sum(vision_map == DISCOVERED)) - env.initial_discovered
            exploration_score = discovered_count * 10
            coin_score = (COINS_COUNT - len(env.coins)) * 500
            step_penalty = env.steps_count * 50
            life_bonus = env.lives * 1000
            
            # 计算道具奖励（仅适用于Level 3）
            item_bonus = 0
            if env.mode == MODE_LEVEL3:
                shovel_bonus = 100 if env.has_shovel else 0
                sword_bonus = 200 if env.has_sword else 0
                magnet_bonus = 150 if env.has_magnet else 0
                key_bonus = 300 if env.has_key else 0
                item_bonus = shovel_bonus + sword_bonus + magnet_bonus + key_bonus
            
            # 计算当前分数
            current_score = exploration_score + coin_score + item_bonus - step_penalty + life_bonus
            
            # 创建游戏状态对象，确保所有numpy数组都转换为嵌套列表
            game_state = {
                'grid': [[int(cell) for cell in row] for row in grid],
                'vision_map': [[int(cell) for cell in row] for row in vision_map],
                'agent_pos': [int(env.agent_pos[0]), int(env.agent_pos[1])],
                'coins': [],  # 先创建空列表
                'lives': int(env.lives),
                'score': int(current_score),
                'mode': env.mode,  # 添加模式信息
                'steps': int(env.steps_count)  # 添加步数信息
            }
            
            # 处理金币位置（从集合转换为列表）
            for coin in env.coins:
                try:
                    if isinstance(coin, tuple):
                        game_state['coins'].append([int(coin[0]), int(coin[1])])
                    elif isinstance(coin, (list, np.ndarray)):
                        game_state['coins'].append([int(coin[0]), int(coin[1])])
                    else:
                        print(f"警告：无法处理的金币格式: {type(coin)}, 值: {coin}")
                except Exception as e:
                    print(f"处理金币时出错: {str(e)}, 金币值: {coin}")
                    continue
            
            # 处理怪物位置（如果有）
            if hasattr(env, 'monsters') and env.monsters:
                game_state['monsters'] = []
                for monster in env.monsters:
                    try:
                        if isinstance(monster, tuple):
                            game_state['monsters'].append([int(monster[0]), int(monster[1])])
                        elif isinstance(monster, (list, np.ndarray)):
                            game_state['monsters'].append([int(monster[0]), int(monster[1])])
                    except Exception as e:
                        print(f"处理怪物时出错: {str(e)}, 怪物值: {monster}")
                        continue
            
            # 处理障碍物位置（如果有）
            if hasattr(env, 'obstacles') and env.obstacles:
                game_state['obstacles'] = []
                for obstacle in env.obstacles:
                    try:
                        if isinstance(obstacle, tuple):
                            game_state['obstacles'].append([int(obstacle[0]), int(obstacle[1])])
                        elif isinstance(obstacle, (list, np.ndarray)):
                            game_state['obstacles'].append([int(obstacle[0]), int(obstacle[1])])
                    except Exception as e:
                        print(f"处理障碍物时出错: {str(e)}, 障碍物值: {obstacle}")
                        continue
            
            # 如果是Level 3模式，添加道具信息
            if env.mode == MODE_LEVEL3:
                game_state.update({
                    'has_shovel': bool(env.has_shovel),
                    'shovel_uses': int(env.shovel_uses),
                    'has_sword': bool(env.has_sword),
                    'has_magnet': bool(env.has_magnet),
                    'has_key': bool(env.has_key)
                })
            
            # 打印调试信息
            print(f"网格形状: {np.array(game_state['grid']).shape}")
            print(f"视野地图形状: {np.array(game_state['vision_map']).shape}")
            
            return game_state
            
        except Exception as e:
            print(f"提取游戏状态时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _evaluate_map(self, env: PathFindingEnv, map_data: Dict[str, Any], 
                      map_index: int, max_steps: int = 200) -> Dict[str, Any]:
        """评估单个地图上的代理表现"""
        try:
            # 加载地图
            env.load_map(map_data, map_index)
            
            # 记录评估开始时间
            start_time = time.time()
            
            # 创建日志目录
            log_dir = os.path.join(self.results_dir, 'interaction_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # 为当前地图创建日志文件
            log_file = os.path.join(log_dir, f'{env.mode}_map{map_index}_log.jsonl')
            
            # 记录初始状态
            exploration_start = np.sum(env.vision_map == DISCOVERED)
            initial_coins = len(env.coins)
            initial_monsters = len(env.monsters) if hasattr(env, 'monsters') else 0
            initial_obstacles = len(env.obstacles) if hasattr(env, 'obstacles') else 0
            
            # 统计信息
            stats = {
                'steps': 0,
                'total_reward': 0,
                'killed_monsters': 0,
                'destroyed_obstacles': 0,
                'api_calls': 0,
                'success': False,
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0
            }
            
            # 游戏循环
            running = True
            while running and stats['steps'] < max_steps:
                # 获取当前状态
                current_state = self._extract_game_state(env)
                
                try:
                    # 每次调用前重置代理，确保是独立对话
                    self.agent.reset()
                    self.agent.set_mode(env.mode)
                    
                    # 获取代理动作
                    action, api_response = self.agent.get_action(current_state)
                    
                    # 确保动作是整数
                    if isinstance(action, (tuple, list)):
                        action = action[0]  # 如果是元组或列表，取第一个元素
                    action = int(action)  # 转换为整数
                    
                    stats['api_calls'] += 1
                    
                    # 更新token统计
                    if hasattr(api_response, 'usage'):
                        stats['prompt_tokens'] += api_response.usage.prompt_tokens
                        stats['completion_tokens'] += api_response.usage.completion_tokens
                        stats['total_tokens'] += api_response.usage.total_tokens
                    
                    # 输出模型交互日志
                    print("\n" + "="*80)
                    print(f"步骤 {stats['steps'] + 1} - 地图 {map_index + 1}")
                    print("="*80)
                    
                    print("\n系统提示词:")
                    print("-"*80)
                    print(api_response.get("system_prompt", "无法获取系统提示词"))
                    print("-"*80)
                    
                    print("\n发送给模型的提示词:")
                    print("-"*80)
                    print(api_response.get("prompt", "无法获取提示词"))
                    print("-"*80)
                    
                    print("\n模型的完整输出:")
                    print("-"*80)
                    if "choices" in api_response and api_response["choices"]:
                        print(api_response["choices"][0]["message"]["content"])
                    else:
                        print("无法获取模型输出")
                    print("-"*80)
                    
                    print(f"\n选择的动作: {action} ({env.get_action_meaning(action)})")
                    if "usage" in api_response:
                        print(f"Token 使用情况: 提示词 {api_response['usage'].get('prompt_tokens', '?')} + "
                              f"完成 {api_response['usage'].get('completion_tokens', '?')} = "
                              f"总计 {api_response['usage'].get('total_tokens', '?')}")
                    print("="*80)
                    
                    # 记录交互日志
                    interaction_log = {
                        'step': stats['steps'],
                        'state': current_state,
                        'action': action,
                        'api_response': {
                            'system_prompt': api_response.get("system_prompt"),
                            'prompt': api_response.get("prompt"),
                            'completion': api_response["choices"][0]["message"]["content"] if api_response.get("choices") else None,
                            'usage': api_response.get("usage", {})
                        }
                    }
                    
                    # 将交互记录写入日志文件
                    with open(log_file, 'a') as f:
                        f.write(json.dumps(interaction_log) + '\n')
                    
                    # 记录执行前的状态
                    before_monsters = len(env.monsters) if hasattr(env, 'monsters') else 0
                    before_obstacles = len(env.obstacles) if hasattr(env, 'obstacles') else 0
                    
                    # 执行动作
                    _, reward, done, _, _ = env.step(action)
                    
                    # 更新统计信息
                    stats['total_reward'] += reward
                    stats['steps'] += 1
                    
                    # 更新怪物和障碍物统计
                    if before_monsters > len(env.monsters):
                        stats['killed_monsters'] += before_monsters - len(env.monsters)
                    if before_obstacles > len(env.obstacles):
                        stats['destroyed_obstacles'] += before_obstacles - len(env.obstacles)
                    
                    # 检查游戏是否结束
                    if done:
                        stats['success'] = reward > 0
                        running = False
                
                except Exception as e:
                    print(f"执行动作时出错: {str(e)}")
                    print(f"动作类型: {type(action)}, 值: {action}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # 计算最终统计信息
            final_stats = {
                "map_index": int(map_index),
                "steps": int(stats['steps']),
                "total_reward": float(stats['total_reward']),
                "custom_score": self._calculate_score(
                    env, stats, exploration_start, initial_coins
                ),
                "exploration_rate": float(self._compute_exploration_rate(env)),
                "success": bool(stats['success']),
                "lives_remaining": int(env.lives),
                "collected_coins": int(initial_coins - len(env.coins)),
                "killed_monsters": int(stats['killed_monsters']),
                "destroyed_obstacles": int(stats['destroyed_obstacles']),
                "runtime": float(time.time() - start_time),
                "api_calls": int(stats['api_calls']),
                "total_tokens": int(stats['total_tokens']),
                "prompt_tokens": int(stats['prompt_tokens']),
                "completion_tokens": int(stats['completion_tokens']),
                "log_file": log_file
            }
            
            return final_stats
            
        except Exception as e:
            print(f"评估地图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _calculate_score(self, env: PathFindingEnv, stats: Dict, 
                        exploration_start: int, initial_coins: int) -> int:
        """计算最终得分，确保返回Python原生int类型"""
        collected_coins = initial_coins - len(env.coins)
        exploration_bonus = int(np.sum(env.vision_map == DISCOVERED) - exploration_start) * 10  # 每格10分
        
        score = (int(collected_coins) * 500 +                  # 金币奖励
                exploration_bonus +                            # 探索奖励
                stats['killed_monsters'] * 500 +               # 怪物击杀奖励
                stats['destroyed_obstacles'] * 500 +           # 障碍破坏奖励
                (2000 if stats['success'] else 0) -           # 到达终点奖励
                stats['steps'] * 50)                          # 步数惩罚
        
        return int(score)  # 确保返回Python原生int类型
    
    def evaluate_mode(self, mode: str, num_maps: int = 30, max_steps: int = 200, start_map_index: int = 0) -> List[Dict[str, Any]]:
        """
        评估指定模式下的多个地图
        
        Args:
            mode: 评估模式
            num_maps: 要评估的地图数量
            max_steps: 每张地图的最大步数
            start_map_index: 从哪个地图索引开始评估 (0-29)
        
        Returns:
            评估结果列表
        """
        print(f"\n开始评估 {mode} 模式...")
        
        # 验证start_map_index的合法性
        if start_map_index < 0 or start_map_index >= num_maps:
            print(f"警告: 提供的开始地图索引({start_map_index})超出范围(0-{num_maps-1})，将使用默认值0")
            start_map_index = 0
            
        if start_map_index > 0:
            print(f"根据指定参数，将从第 {start_map_index + 1} 张地图开始评估")
        
        # 确保代理已初始化
        self._initialize_agent()
        
        # 加载地图
        maps = self._load_maps(mode)
        maps = maps[:num_maps]  # 只取前num_maps个地图
        
        # 获取已完成的地图索引
        completed_maps = self._get_completed_maps(mode)
        
        # 打印已完成地图的详细信息
        if completed_maps:
            print(f"发现已完成评估的地图: {len(completed_maps)}/{num_maps}")
            print(f"已完成的地图索引: {sorted(completed_maps)}")
            
            # 检查是否所有地图都已完成
            remaining_maps_count = sum(1 for i in range(start_map_index, num_maps) if i not in completed_maps)
            if remaining_maps_count == 0:
                print(f"{mode} 从索引 {start_map_index} 开始的所有地图已评估完成！跳过评估...")
                
                # 加载已有的评估结果，确保返回正确数据
                if mode in self.results and self.results[mode]:
                    return self.results[mode]
                else:
                    # 尝试从评估结果文件加载
                    report_pattern = os.path.join(self.results_dir, "evaluation_results_*.json")
                    report_files = glob.glob(report_pattern)
                    if report_files:
                        latest_report = sorted(report_files)[-1]
                        try:
                            with open(latest_report, 'r') as f:
                                report_data = json.load(f)
                                if 'results' in report_data and mode in report_data['results']:
                                    self.results[mode] = report_data['results'][mode]
                                    print(f"从报告文件加载了 {len(self.results[mode])} 个评估结果")
                                    return self.results[mode]
                        except Exception as e:
                            print(f"加载报告文件失败: {str(e)}")
                    
                    # 如果没有找到结果，返回空列表
                    print("未找到已有评估结果，但所有地图已完成，返回空结果")
                    return []
        
        # 创建环境，完全禁用pygame
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 使用虚拟显示
        self.env = PathFindingEnv(mode=mode)
        
        # 评估结果
        if mode not in self.results:
            self.results[mode] = []
        
        # 筛选未完成的地图，同时考虑start_map_index
        remaining_maps = [(i, map_data) for i, map_data in enumerate(maps) 
                         if i not in completed_maps and i >= start_map_index]
        
        # 根据已完成的评估，确保结果数组的完整性
        for i in range(num_maps):
            if i in completed_maps and not any(r.get('map_index') == i for r in self.results[mode]):
                # 尝试从评估结果文件加载该地图数据
                found = False
                report_pattern = os.path.join(self.results_dir, "evaluation_results_*.json")
                report_files = glob.glob(report_pattern)
                for report_file in sorted(report_files, reverse=True):
                    try:
                        with open(report_file, 'r') as f:
                            report_data = json.load(f)
                            if 'results' in report_data and mode in report_data['results']:
                                for result in report_data['results'][mode]:
                                    if result.get('map_index') == i:
                                        self.results[mode].append(result)
                                        found = True
                                        print(f"从报告文件加载了地图 {i+1} 的评估结果")
                                        break
                            if found:
                                break
                    except Exception as e:
                        print(f"尝试从报告文件加载地图 {i+1} 数据失败: {str(e)}")
        
        if not remaining_maps:
            print(f"{mode} 所有需要评估的地图已完成！")
            return self.results[mode]
        
        print(f"需要评估的地图数量: {len(remaining_maps)}")
        print(f"将评估以下地图索引: {[i+1 for i, _ in remaining_maps]}")
        
        # 使用tqdm显示进度条，只显示未完成的地图
        for i, map_data in tqdm(remaining_maps, desc=f"评估 {mode}"):
            print(f"\n开始评估地图 {i+1}...")
            result = self._evaluate_map(self.env, map_data, i, max_steps)
            
            # 检查是否已有该地图的结果，避免重复添加
            if not any(r.get('map_index') == i for r in self.results[mode]):
                self.results[mode].append(result)
            
            # 每完成一张地图就保存检查点
            self._save_checkpoint(mode, i)
            
            # 每5个地图显示一次统计信息
            if (len([r for r in self.results[mode] if r.get('map_index') >= start_map_index])) % 5 == 0 or i == remaining_maps[-1][0]:
                all_results = [r for r in self.results[mode] if r.get('map_index') >= start_map_index]
                if all_results:
                    success_rate = sum(r["success"] for r in all_results) / len(all_results)
                    avg_score = sum(r["custom_score"] for r in all_results) / len(all_results)
                    print(f"\n当前进度 (从索引 {start_map_index+1} 开始评估的地图中完成了 {len(all_results)} 个):")
                    print(f"  通关率: {success_rate:.2%}")
                    print(f"  平均得分: {avg_score:.1f}")
        
        # 按地图索引排序结果
        self.results[mode].sort(key=lambda x: x['map_index'])
        
        return self.results[mode]
    
    def _print_evaluation_progress(self):
        """打印当前的评估进度"""
        print("\n===== 当前评估进度 =====")
        all_completed = True
        
        for mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
            completed = set()
            # 从结果中获取已完成的地图
            if mode in self.results:
                for result in self.results[mode]:
                    if 'map_index' in result:
                        completed.add(result['map_index'])
            
            # 补充检查日志文件
            log_dir = os.path.join(self.results_dir, 'interaction_logs')
            if os.path.exists(log_dir):
                log_pattern = f"{mode}_map*_log.jsonl"
                log_files = glob.glob(os.path.join(log_dir, log_pattern))
                
                # 如果没找到，尝试替换空格为下划线的模式
                if not log_files:
                    log_pattern = f"{mode.replace(' ', '_')}_map*_log.jsonl"
                    log_files = glob.glob(os.path.join(log_dir, log_pattern))
                
                for log_file in log_files:
                    map_index_match = re.search(r'map(\d+)_log', os.path.basename(log_file))
                    if map_index_match:
                        try:
                            map_index = int(map_index_match.group(1)) - 1  # 转换为0基索引
                            completed.add(map_index)
                        except ValueError:
                            pass
            
            # 计算进度
            total_maps = 30  # 默认值，应与主要参数保持一致
            remaining = total_maps - len(completed)
            progress_percent = len(completed) / total_maps * 100
            
            print(f"{mode}: {len(completed)}/{total_maps} 完成 ({progress_percent:.1f}%) - 剩余: {remaining}")
            
            # 如果任何一个模式未完成，则标记为未全部完成
            if len(completed) < total_maps:
                all_completed = False
        
        print("所有评估" + ("已完成！" if all_completed else "尚未完成。"))
        print("========================\n")
        
        return all_completed

    def evaluate_all_modes(self, num_maps: int = 30, max_steps: int = 200, start_map_index: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        """评估所有模式
        
        Args:
            num_maps: 每个模式评估的地图数量
            max_steps: 每张地图的最大步数
            start_map_index: 从哪个地图索引开始评估
        
        Returns:
            评估结果字典
        """
        modes = [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]
        
        # 先打印当前的评估进度
        all_completed = self._print_evaluation_progress()
        
        # 如果所有评估已完成，直接返回结果
        if all_completed:
            print("所有模式的所有地图已评估完成，无需再次评估。")
            return self.results
        
        for mode in modes:
            # 打印当前开始评估的模式信息
            print(f"\n========== 开始评估模式: {mode} ==========")
            
            # 对于每个模式使用相同的start_map_index
            self.evaluate_mode(mode, num_maps, max_steps, start_map_index)
            
            # 每个模式评估完成后保存一次完整结果
            self.save_results()
            
            # 再次打印评估进度
            self._print_evaluation_progress()
        
        # 评估全部完成后，删除检查点文件
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                print("评估完成，已删除检查点文件")
            except Exception as e:
                print(f"删除检查点文件失败: {str(e)}")
        
        # 生成最终报告
        self.generate_report()
        
        return self.results
    
    def calculate_statistics(self, mode: str = None) -> Dict[str, Any]:
        """计算统计信息"""
        if mode is None:
            # 计算所有模式的统计信息
            return {
                mode: self.calculate_statistics(mode)
                for mode in self.results.keys()
                if self.results[mode]  # 只处理有数据的模式
            }
        
        results = self.results[mode]
        if not results:
            return {}
        
        # 计算关键指标
        success_rate = sum(r["success"] for r in results) / len(results)
        avg_score = sum(r["custom_score"] for r in results) / len(results)
        avg_exploration = sum(r["exploration_rate"] for r in results) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)
        avg_coins = sum(r["collected_coins"] for r in results) / len(results)
        coin_collection_rate = avg_coins / COINS_COUNT
        
        # 计算其他可能有用的统计数据
        avg_runtime = sum(r["runtime"] for r in results) / len(results)
        avg_api_calls = sum(r["api_calls"] for r in results) / len(results)
        
        # 怪物和障碍物（可能在Level 1中没有）
        avg_killed_monsters = sum(r["killed_monsters"] for r in results) / len(results)
        avg_destroyed_obstacles = sum(r["destroyed_obstacles"] for r in results) / len(results)
        
        # 时间和API调用效率
        steps_per_second = avg_steps / avg_runtime if avg_runtime > 0 else 0
        reward_per_call = avg_score / avg_api_calls if avg_api_calls > 0 else 0
        
        return {
            "sample_size": len(results),
            "success_rate": success_rate,
            "avg_score": avg_score,
            "avg_exploration_rate": avg_exploration,
            "avg_steps": avg_steps,
            "avg_collected_coins": avg_coins,
            "coin_collection_rate": coin_collection_rate,
            "avg_killed_monsters": avg_killed_monsters,
            "avg_destroyed_obstacles": avg_destroyed_obstacles,
            "avg_runtime": avg_runtime,
            "avg_api_calls": avg_api_calls,
            "steps_per_second": steps_per_second,
            "reward_per_call": reward_per_call
        }
    
    def save_results(self) -> str:
        """保存评估结果到文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # 将结果转换为JSON兼容格式
        json_results = {
            "timestamp": timestamp,
            "model": self.model,
            "api_type": self.api_type,
            "results": self.results
        }
        
        # 保存到文件
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"评估结果已保存到: {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """生成评估报告"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"evaluation_report_{timestamp}.html"
        report_path = os.path.join(self.results_dir, report_file)
        
        # 计算所有模式的统计信息
        stats = self.calculate_statistics()
        
        # 创建统计数据的DataFrame
        stats_data = []
        for mode, mode_stats in stats.items():
            row = {"模式": mode}
            row.update({
                "样本数量": mode_stats["sample_size"],
                "通关率": f"{mode_stats['success_rate']:.2%}",
                "平均得分": f"{mode_stats['avg_score']:.1f}",
                "平均探索率": f"{mode_stats['avg_exploration_rate']:.2%}",
                "平均步数": f"{mode_stats['avg_steps']:.1f}",
                "金币收集率": f"{mode_stats['coin_collection_rate']:.2%}",
                "怪物击杀数": f"{mode_stats['avg_killed_monsters']:.2f}",
                "障碍破坏数": f"{mode_stats['avg_destroyed_obstacles']:.2f}",
                "平均运行时间": f"{mode_stats['avg_runtime']:.2f}秒",
                "平均API调用": f"{mode_stats['avg_api_calls']:.1f}次"
            })
            stats_data.append(row)
        
        stats_df = pd.DataFrame(stats_data)
        
        # 为每个模式创建详细的数据表
        mode_dfs = {}
        for mode, results in self.results.items():
            if not results:
                continue
            
            # 创建DataFrame
            df = pd.DataFrame(results)
            # 添加更多处理和格式化...
            mode_dfs[mode] = df
        
        # 生成HTML报告
        with open(report_path, 'w') as f:
            f.write("<html><head>")
            f.write("<title>AI代理评估报告</title>")
            f.write("<style>")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
            f.write("h1, h2, h3 { color: #2c3e50; }")
            f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            f.write("th { background-color: #f2f2f2; }")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }")
            f.write("</style>")
            f.write("</head><body>")
            
            # 报告标题
            f.write(f"<h1>AI代理评估报告</h1>")
            f.write(f"<p>API类型: {self.api_type.upper()}</p>")
            f.write(f"<p>模型: {self.model}</p>")
            f.write(f"<p>评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # 统计摘要
            f.write("<h2>评估摘要</h2>")
            f.write(stats_df.to_html(index=False))
            
            # 为每个模式添加详细结果
            for mode, df in mode_dfs.items():
                f.write(f"<h2>{mode} 模式详细结果</h2>")
                
                # 增加一些数据可视化
                try:
                    # 创建通关率饼图
                    plt.figure(figsize=(8, 6))
                    success_count = df['success'].sum()
                    failure_count = len(df) - success_count
                    plt.pie(
                        [success_count, failure_count], 
                        labels=['成功', '失败'], 
                        autopct='%1.1f%%',
                        colors=['#2ecc71', '#e74c3c']
                    )
                    plt.title(f'{mode} 通关率')
                    pie_file = f"{mode.replace(' ', '_')}_success_rate_pie_{timestamp}.png"
                    plt.savefig(os.path.join(self.results_dir, pie_file))
                    plt.close()
                    f.write(f"<img src='{pie_file}' alt='通关率饼图' style='width:400px;'>")
                    
                    # 创建探索率直方图
                    plt.figure(figsize=(10, 6))
                    plt.hist(df['exploration_rate'], bins=10, alpha=0.7, color='#3498db')
                    plt.title(f'{mode} 探索率分布')
                    plt.xlabel('探索率')
                    plt.ylabel('地图数量')
                    hist_file = f"{mode.replace(' ', '_')}_exploration_hist_{timestamp}.png"
                    plt.savefig(os.path.join(self.results_dir, hist_file))
                    plt.close()
                    f.write(f"<img src='{hist_file}' alt='探索率直方图' style='width:400px;'>")
                    
                    # 添加得分与步数的散点图
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df['steps'], df['custom_score'], alpha=0.7, c=df['success'].map({True: '#2ecc71', False: '#e74c3c'}))
                    plt.title(f'{mode} 得分与步数关系')
                    plt.xlabel('步数')
                    plt.ylabel('得分')
                    plt.grid(True, alpha=0.3)
                    scatter_file = f"{mode.replace(' ', '_')}_score_steps_scatter_{timestamp}.png"
                    plt.savefig(os.path.join(self.results_dir, scatter_file))
                    plt.close()
                    f.write(f"<img src='{scatter_file}' alt='得分与步数关系' style='width:400px;'>")
                except Exception as e:
                    f.write(f"<p>生成图表时出错: {str(e)}</p>")
                
                # 添加详细数据表
                f.write("<h3>详细数据</h3>")
                f.write(df.to_html(index=False))
            
            f.write("</body></html>")
        
        print(f"评估报告已生成: {report_path}")
        return report_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AI代理评估脚本")
    parser.add_argument("--api_key", type=str, default=None,
                      help="API密钥（根据api_type参数确定类型）")
    parser.add_argument("--model", type=str, default=None,
                      help="模型名称（根据api_type参数确定）")
    parser.add_argument("--api_type", type=str, default=DEFAULT_API_TYPE, choices=["deepseek", "openai"],
                      help="API类型，可选deepseek或openai")
    parser.add_argument("--base_url", type=str, default=None,
                      help="API基础URL，仅当api_type=openai时使用")
    parser.add_argument("--maps_dir", type=str, default=None,
                      help="地图目录")
    parser.add_argument("--results_dir", type=str, default=None,
                      help="结果保存目录")
    parser.add_argument("--num_maps", type=int, default=30,
                      help="每个难度评估的地图数量")
    parser.add_argument("--max_steps", type=int, default=200,
                      help="每张地图的最大步数")
    parser.add_argument("--mode", type=str, default=None,
                      choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3, None],
                      help=f"仅评估指定模式，不指定则评估全部")
    parser.add_argument("--resume", action="store_true", default=True,
                      help="是否启用续传模式，跳过已评估的地图")
    parser.add_argument("--start_map_index", type=int, default=0,
                      help="从指定索引的地图开始评估（0-29，默认为0)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建评估器
    evaluator = AgentEvaluator(
        api_key=args.api_key,
        model=args.model,
        api_type=args.api_type,
        base_url=args.base_url,
        maps_dir=args.maps_dir,
        results_dir=args.results_dir
    )
    
    # 输出续传模式状态
    if args.resume:
        print("已启用续传模式，将跳过已评估的地图")
    else:
        print("已禁用续传模式，将重新评估所有地图")
        # 如果禁用续传模式，清空结果
        evaluator.results = {
            MODE_LEVEL1: [],
            MODE_LEVEL2: [],
            MODE_LEVEL3: []
        }
    
    # 显示起始地图信息
    if args.start_map_index > 0:
        print(f"将从第 {args.start_map_index + 1} 张地图开始评估（跳过前 {args.start_map_index} 张地图）")
    
    # 如果指定了模式，则只评估该模式
    if args.mode:
        print(f"开始评估模式: {args.mode}")
        evaluator.evaluate_mode(args.mode, args.num_maps, args.max_steps, args.start_map_index)
        evaluator.save_results()
        evaluator.generate_report()
    else:
        # 评估所有模式
        print(f"开始评估所有模式")
        evaluator.evaluate_all_modes(args.num_maps, args.max_steps, args.start_map_index)
    
    # 计算统计信息
    stats = evaluator.calculate_statistics()
    
    # 显示汇总结果
    print("\n评估结果汇总:")
    for mode, mode_stats in stats.items():
        print(f"\n{mode} 模式:")
        print(f"  样本数量: {mode_stats['sample_size']}")
        print(f"  通关率: {mode_stats['success_rate']:.2%}")
        print(f"  平均得分: {mode_stats['avg_score']:.1f}")
        print(f"  平均探索率: {mode_stats['avg_exploration_rate']:.2%}")
        print(f"  金币收集率: {mode_stats['coin_collection_rate']:.2%}")
        if mode != MODE_LEVEL1:  # Level 1没有怪物
            print(f"  平均击杀怪物: {mode_stats['avg_killed_monsters']:.2f}")
        if mode == MODE_LEVEL3:  # 只有Level 3有道具
            print(f"  平均破坏障碍: {mode_stats['avg_destroyed_obstacles']:.2f}")
    
    print("\n详细报告已生成，查看结果目录获取完整评估报告。")

if __name__ == "__main__":
    main() 
