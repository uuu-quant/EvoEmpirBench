import os
import json
import time
import numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.game.environment import PathFindingEnv
from src.game.map_generator import MapGenerator
from src.agent.agent_interface import DeepSeekAgent
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3, DISCOVERED, COINS_COUNT, INITIAL_LIVES
from src.config.api_config import DEFAULT_API_TYPE
from src.config.paths import MAZE_EVAL_MAPS_DIR, RESULTS_DIR

class QuickEvaluator:
    """AI代理快速评估工具"""
    
    def __init__(self, api_key: str = None, model: str = None,
                 api_type: str = DEFAULT_API_TYPE):
        """初始化评估器"""
        self.api_key = api_key
        self.model = model
        self.api_type = api_type
        
        self.maps_dir = str(MAZE_EVAL_MAPS_DIR)
        
        # 确保地图目录存在
        os.makedirs(self.maps_dir, exist_ok=True)
        
        # 初始化代理
        self.agent = DeepSeekAgent(api_key=api_key, model=model, api_type=api_type)
        self.model = self.agent.model
        self.api_type = self.agent.api_type
        self.agent.set_show_prompt(False)  # 评估时不显示提示
        
        # 初始化环境
        self.env = None
        
        # 评估结果
        self.results = {}
    
    def _load_maps(self, mode: str, num_maps: int = 30) -> List[Dict]:
        """加载地图，如果不存在则生成"""
        collection_file = os.path.join(self.maps_dir, f"{mode.replace(' ', '_')}_collection.json")
        
        # 检查地图文件是否存在
        if not os.path.exists(collection_file):
            print(f"地图文件不存在，正在生成 {mode} 地图...")
            MapGenerator.generate_maps(num_maps, mode, self.maps_dir)
        
        # 加载地图
        maps = MapGenerator.load_map_collection(collection_file)
        
        # 确保有足够的地图
        if len(maps) < num_maps:
            print(f"地图数量不足，正在生成更多 {mode} 地图...")
            additional_maps = MapGenerator.generate_maps(num_maps - len(maps), mode, self.maps_dir)
            maps.extend(additional_maps)
            # 保存更新后的地图集合
            with open(collection_file, 'w') as f:
                json.dump(maps, f, indent=2)
        
        return maps[:num_maps]
    
    def _compute_exploration_rate(self, env: PathFindingEnv) -> float:
        """计算探索率"""
        grid_size = env.grid.shape[0]
        total_cells = grid_size * grid_size
        obstacles_count = len(env.obstacles)
        explorable_cells = total_cells - obstacles_count
        
        # 统计已探索的单元格数量
        discovered_count = np.sum(env.vision_map == DISCOVERED)
        
        return discovered_count / explorable_cells if explorable_cells > 0 else 0
    
    def _extract_game_state(self, env: PathFindingEnv) -> Dict[str, Any]:
        """提取游戏状态"""
        # 计算分数各组成部分
        discovered_count = np.sum(env.vision_map == DISCOVERED) - env.initial_discovered
        exploration_score = discovered_count * 10
        coin_score = (COINS_COUNT - len(env.coins)) * 500
        step_penalty = env.steps_count * 50
        life_bonus = env.lives * 1000
        
        # 道具奖励（Level 3）
        item_bonus = 0
        if env.mode == MODE_LEVEL3:
            shovel_bonus = 100 if env.has_shovel else 0
            sword_bonus = 200 if env.has_sword else 0
            magnet_bonus = 150 if env.has_magnet else 0
            key_bonus = 300 if env.has_key else 0
            item_bonus = shovel_bonus + sword_bonus + magnet_bonus + key_bonus
        
        # 计算总分
        current_score = exploration_score + coin_score + item_bonus - step_penalty + life_bonus
        
        # 更新环境对象中的分数
        env.score = current_score
        
        # 创建游戏状态
        game_state = {
            'grid': env.grid,
            'vision_map': env.vision_map,
            'agent_pos': env.agent_pos,
            'coins': env.coins,
            'lives': env.lives,
            'score': current_score
        }
        
        # Level 3特有道具信息
        if env.mode == MODE_LEVEL3:
            game_state.update({
                'has_shovel': env.has_shovel,
                'shovel_uses': env.shovel_uses,
                'has_sword': env.has_sword,
                'has_magnet': env.has_magnet,
                'has_key': env.has_key
            })
        
        return game_state
    
    def evaluate_map(self, env: PathFindingEnv, map_data: Dict, max_steps: int = 200) -> Dict:
        """评估单个地图上的代理表现"""
        # 加载地图 - 确保不渲染
        try:
            env.load_map(map_data, 0, render=False)
        except TypeError:
            # 如果不接受render参数，则直接调用
            env.load_map(map_data, 0)
        
        # 重置代理
        self.agent.reset()
        self.agent.set_mode(env.mode)
        
        # 记录初始状态
        exploration_start = np.sum(env.vision_map == DISCOVERED)
        initial_coins = len(env.coins)
        
        # 统计信息
        steps = 0
        total_reward = 0
        killed_monsters = 0
        destroyed_obstacles = 0
        success = False
        
        # 游戏循环
        running = True
        while running and steps < max_steps:
            # 获取游戏状态
            game_state = self._extract_game_state(env)
            
            # 获取代理动作
            try:
                # 记录执行前怪物和障碍物数量
                before_monsters = len(env.monsters)
                before_obstacles = len(env.obstacles)
                
                # 获取动作
                action, _ = self.agent.get_action(game_state)
                
                # 执行动作 - 检查是否支持render参数
                try:
                    _, reward, done, _, _ = env.step(action, render=False)
                except TypeError:
                    # 如果不支持render参数，则直接调用
                    _, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                steps += 1
                
                # 检查怪物击杀和障碍物破坏
                if before_monsters > len(env.monsters):
                    killed_monsters += before_monsters - len(env.monsters)
                
                if before_obstacles > len(env.obstacles):
                    destroyed_obstacles += before_obstacles - len(env.obstacles)
                
                # 检查游戏是否结束
                if done:
                    success = reward > 0
                    running = False
            
            except Exception as e:
                print(f"执行动作时出错: {str(e)}")
                break
        
        # 计算最终探索率
        exploration_rate = self._compute_exploration_rate(env)
        
        # 计算收集的金币数
        collected_coins = initial_coins - len(env.coins)
        
        # 自定义评分计算 (按照要求的评分标准)
        custom_score = (collected_coins * 500 -                  # 金币奖励
                       (INITIAL_LIVES - env.lives) * 1000 -      # 生命损失惩罚
                       steps * 50 +                              # 步数惩罚
                       (np.sum(env.vision_map == DISCOVERED) - exploration_start) * 100 +  # 探索奖励
                       killed_monsters * 500 +                   # 怪物击杀奖励
                       destroyed_obstacles * 500 +               # 障碍破坏奖励
                       (2000 if success else 0))                 # 到达终点奖励
        
        # 返回评估结果
        return {
            "steps": steps,
            "total_reward": total_reward,
            "custom_score": custom_score,
            "exploration_rate": exploration_rate,
            "success": success,
            "lives_remaining": env.lives,
            "collected_coins": collected_coins,
            "killed_monsters": killed_monsters,
            "destroyed_obstacles": destroyed_obstacles,
            "max_steps_reached": steps >= max_steps
        }
    
    def evaluate_mode(self, mode: str, num_maps: int = 30, max_steps: int = 200) -> Dict:
        """评估某个模式下的性能"""
        print(f"\n===== 开始评估 {mode} 模式 =====")
        
        # 加载地图
        maps = self._load_maps(mode, num_maps)
        
        # 创建环境 - 尝试使用无渲染模式
        try:
            # 首先尝试使用无渲染模式参数初始化
            self.env = PathFindingEnv(mode=mode, no_render=True)
        except TypeError:
            # 如果不支持no_render参数，则使用默认初始化
            self.env = PathFindingEnv(mode=mode)
            # 尝试禁用渲染
            if hasattr(self.env, 'render_mode'):
                self.env.render_mode = None
        
        # 结果列表
        results = []
        
        # 使用tqdm显示进度
        for i, map_data in enumerate(tqdm(maps, desc=f"评估 {mode}")):
            result = self.evaluate_map(self.env, map_data, max_steps)
            results.append(result)
            
            # 每5个地图显示一次中间结果
            if (i + 1) % 5 == 0:
                success_count = sum(1 for r in results if r["success"])
                avg_score = sum(r["custom_score"] for r in results) / len(results)
                print(f"  已完成: {i+1}/{num_maps} 地图, 通关: {success_count}/{i+1} ({success_count/(i+1):.1%}), 平均分: {avg_score:.1f}")
        
        # 计算统计数据
        success_count = sum(1 for r in results if r["success"])
        success_rate = success_count / len(results)
        avg_score = sum(r["custom_score"] for r in results) / len(results)
        avg_exploration = sum(r["exploration_rate"] for r in results) / len(results)
        avg_collected_coins = sum(r["collected_coins"] for r in results) / len(results)
        coin_collection_rate = avg_collected_coins / COINS_COUNT
        
        # 保存结果
        self.results[mode] = {
            "raw_results": results,
            "stats": {
                "num_maps": len(results),
                "success_count": success_count,
                "success_rate": success_rate,
                "avg_score": avg_score,
                "avg_exploration": avg_exploration,
                "avg_collected_coins": avg_collected_coins,
                "coin_collection_rate": coin_collection_rate,
                "avg_steps": sum(r["steps"] for r in results) / len(results),
                "avg_killed_monsters": sum(r["killed_monsters"] for r in results) / len(results),
                "avg_destroyed_obstacles": sum(r["destroyed_obstacles"] for r in results) / len(results)
            }
        }
        
        # 显示结果摘要
        print(f"\n{mode} 评估结果:")
        print(f"  样本数量: {len(results)}")
        print(f"  通关数量: {success_count}")
        print(f"  通关率: {success_rate:.2%}")
        print(f"  平均得分: {avg_score:.1f}")
        print(f"  平均探索率: {avg_exploration:.2%}")
        print(f"  金币收集率: {coin_collection_rate:.2%}")
        if mode != MODE_LEVEL1:  # Level 1没有怪物
            print(f"  平均击杀怪物: {self.results[mode]['stats']['avg_killed_monsters']:.2f}")
        if mode == MODE_LEVEL3:  # 只有Level 3有道具
            print(f"  平均破坏障碍: {self.results[mode]['stats']['avg_destroyed_obstacles']:.2f}")
        
        return self.results[mode]
    
    def evaluate_all(self, num_maps: int = 30, max_steps: int = 200):
        """评估所有难度级别"""
        # 三个难度依次评估
        for mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
            self.evaluate_mode(mode, num_maps, max_steps)
        
        # 保存结果到文件
        self.save_results()
        
        # 打印总结
        print("\n===== 评估总结 =====")
        for mode, result in self.results.items():
            stats = result["stats"]
            print(f"\n{mode}:")
            print(f"  通关率: {stats['success_rate']:.2%}")
            print(f"  平均得分: {stats['avg_score']:.1f}")
            print(f"  平均探索率: {stats['avg_exploration']:.2%}")
        
        return self.results
    
    def save_results(self):
        """保存评估结果到文件"""
        results_dir = os.path.join(str(RESULTS_DIR), 'quick_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        # 创建结果文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"quick_evaluation_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # 保存结果
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "model": self.model,
                "results": self.results
            }, f, indent=2)
        
        print(f"\n评估结果已保存到: {filepath}")
        return filepath

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AI代理快速评估工具")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API密钥；默认从环境变量读取")
    parser.add_argument("--model", type=str, default=None,
                       help="模型名称；默认根据api_type读取环境变量")
    parser.add_argument("--api_type", type=str, default=DEFAULT_API_TYPE, choices=["deepseek", "openai"],
                       help="API类型")
    parser.add_argument("--num_maps", type=int, default=30,
                       help="每个难度评估的地图数量")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="每张地图的最大步数")
    parser.add_argument("--mode", type=str, choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3],
                       help="仅评估指定模式（不指定则评估全部）")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("===== AI代理性能评估工具 =====")
    print(f"API类型: {args.api_type}")
    print(f"模型: {args.model}")
    print(f"每个难度评估地图数量: {args.num_maps}")
    print(f"每张地图最大步数: {args.max_steps}")
    
    # 创建评估器
    evaluator = QuickEvaluator(api_key=args.api_key, model=args.model, api_type=args.api_type)
    
    # 开始评估
    if args.mode:
        # 只评估指定模式
        evaluator.evaluate_mode(args.mode, args.num_maps, args.max_steps)
    else:
        # 评估所有模式
        evaluator.evaluate_all(args.num_maps, args.max_steps)

if __name__ == "__main__":
    main() 
