import pygame
import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import glob
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.game.environment import PathFindingEnv
from src.game.map_generator import MapGenerator
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3, DISCOVERED, COINS_COUNT, INITIAL_LIVES, GOAL_POS
from src.config.paths import MAZE_EVAL_MAPS_DIR, RESULTS_DIR

def _convert_to_serializable(obj):
    """将NumPy类型转换为Python原生类型，确保可JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    return obj

def _extract_game_stats(env: PathFindingEnv, start_time: float, total_steps: int) -> Dict[str, Any]:
    """提取游戏统计数据，与AI评估保持一致"""
    # 计算最终统计信息
    end_time = time.time()
    runtime = end_time - start_time
    
    # 计算探索率
    grid_size = env.grid.shape[0]
    total_cells = grid_size * grid_size
    obstacles_count = len(env.obstacles)
    explorable_cells = total_cells - obstacles_count
    discovered_count = np.sum(env.vision_map == DISCOVERED)
    exploration_rate = discovered_count / explorable_cells if explorable_cells > 0 else 0
    
    # 创建与AI评估结果格式相同的统计数据
    stats = {
        "steps": int(total_steps),  # 确保是Python原生int类型
        "total_reward": float(env.score),  # 确保是Python原生float类型
        "custom_score": float(env.score),  # 确保是Python原生float类型
        "exploration_rate": float(exploration_rate),
        "success": bool(env.agent_pos == GOAL_POS),  # 确保是Python原生bool类型
        "lives_remaining": int(env.lives),  # 确保是Python原生int类型
        "collected_coins": int(COINS_COUNT - len(env.coins)),  # 确保是Python原生int类型
        "killed_monsters": 0,
        "destroyed_obstacles": 0,
        "runtime": float(runtime),  # 确保是Python原生float类型
        "api_calls": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }
    
    return stats

def _load_map_collection(mode: str) -> List:
    """加载地图集合"""
    # 加载地图集合
    maps_dir = str(MAZE_EVAL_MAPS_DIR)
    collection_file = os.path.join(maps_dir, f"{mode.replace(' ', '_')}_collection.json")
    
    # 检查地图文件是否存在
    if not os.path.exists(collection_file):
        print(f"错误：找不到地图文件 {collection_file}")
        print("请先运行 python scripts/generate_maps.py 生成地图")
        return []
    
    return MapGenerator.load_map_collection(collection_file)

def _get_results_dir() -> str:
    """获取结果保存目录"""
    # 设置结果保存目录
    results_dir = os.path.join(str(RESULTS_DIR), 'human')
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

def _get_completed_levels() -> Dict[str, List[int]]:
    """获取已完成的关卡信息"""
    results_dir = _get_results_dir()
    completed_levels = {
        MODE_LEVEL1: [],
        MODE_LEVEL2: [],
        MODE_LEVEL3: []
    }
    
    # 加载现有评估结果文件
    report_pattern = os.path.join(results_dir, "evaluation_results_*.json")
    report_files = glob.glob(report_pattern)
    
    if report_files:
        # 按文件名排序，取最新的
        latest_report = sorted(report_files)[-1]
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            if 'results' in report_data:
                for mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
                    if mode in report_data['results']:
                        for result in report_data['results'][mode]:
                            if 'map_index' in result:
                                completed_levels[mode].append(result['map_index'])
                        
                        # 确保列表是有序的
                        completed_levels[mode].sort()
        except Exception as e:
            print(f"加载评估结果文件失败: {str(e)}")
    
    return completed_levels

def _save_game_result(mode: str, map_index: int, stats: Dict[str, Any]):
    """保存游戏结果"""
    results_dir = _get_results_dir()
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 尝试加载现有结果文件
    results = {
        MODE_LEVEL1: [],
        MODE_LEVEL2: [],
        MODE_LEVEL3: []
    }
    
    report_pattern = os.path.join(results_dir, "evaluation_results_*.json")
    report_files = glob.glob(report_pattern)
    
    if report_files:
        # 按文件名排序，取最新的
        latest_report = sorted(report_files)[-1]
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            if 'results' in report_data:
                results.update(report_data['results'])
        except Exception as e:
            print(f"加载评估结果文件失败: {str(e)}")
    
    # 添加地图索引到统计数据
    stats["map_index"] = int(map_index)
    
    # 检查是否已有该地图的结果，如果有则替换
    updated = False
    if mode in results and results[mode]:
        for i, result in enumerate(results[mode]):
            if result.get('map_index') == map_index:
                results[mode][i] = stats
                updated = True
                break
    
    # 如果没有找到该地图结果，添加新结果
    if not updated:
        if mode not in results:
            results[mode] = []
        results[mode].append(stats)
    
    # 按地图索引排序结果
    results[mode].sort(key=lambda x: x.get('map_index', 0))
    
    # 保存结果到新文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # 将结果转换为JSON兼容格式
    json_results = {
        "timestamp": timestamp,
        "model": "human",
        "api_type": "human",
        "results": _convert_to_serializable(results)  # 确保所有数据都可序列化
    }
    
    # 保存到文件
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"游戏结果已保存: {mode} - 地图 {map_index + 1}")

def play_human_evaluation():
    """人类玩家评估模式"""
    print("欢迎来到游戏评估模式！")
    print("在此模式下，您将完成90个关卡（每个难度30个），并记录您的表现数据。")
    print("数据将用于与AI代理的表现进行比较。\n")
    
    # 获取已完成的关卡
    completed_levels = _get_completed_levels()
    total_completed = sum(len(levels) for levels in completed_levels.values())
    
    print(f"您已完成 {total_completed}/90 个关卡。")
    print(f"Level 1: {len(completed_levels[MODE_LEVEL1])}/30")
    print(f"Level 2: {len(completed_levels[MODE_LEVEL2])}/30")
    print(f"Level 3: {len(completed_levels[MODE_LEVEL3])}/30\n")
    
    # 选择要开始的模式和关卡
    mode = None
    map_index = None
    
    # 按顺序查找第一个未完成的关卡
    for current_mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
        if len(completed_levels[current_mode]) < 30:
            # 找到第一个未完成的地图索引
            completed = set(completed_levels[current_mode])
            for i in range(30):
                if i not in completed:
                    mode = current_mode
                    map_index = i
                    break
            if mode is not None:
                break
    
    # 如果所有关卡都已完成
    if mode is None:
        print("恭喜！您已完成所有90个关卡！")
        return
    
    print(f"即将开始 {mode} - 地图 {map_index + 1}")
    input("按回车键继续...")
    
    # 加载地图
    maps = _load_map_collection(mode)
    
    if not maps or map_index >= len(maps):
        print(f"错误：找不到 {mode} 的地图 {map_index + 1}")
        return
    
    # 游戏循环
    keep_playing = True
    current_mode = mode
    current_map_index = map_index
    
    while keep_playing:
        # 如果所有难度都完成了30张地图，游戏结束
        all_completed = True
        for m in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
            if len(completed_levels[m]) < 30:
                all_completed = False
                break
        
        if all_completed:
            print("恭喜！您已完成所有90个关卡！")
            break
        
        # 加载当前难度的地图
        current_maps = _load_map_collection(current_mode)
        
        if not current_maps or current_map_index >= len(current_maps):
            print(f"错误：找不到 {current_mode} 的地图 {current_map_index + 1}")
            # 尝试下一个难度
            if current_mode == MODE_LEVEL1:
                current_mode = MODE_LEVEL2
            elif current_mode == MODE_LEVEL2:
                current_mode = MODE_LEVEL3
            else:
                current_mode = MODE_LEVEL1
            
            # 找到该难度下第一个未完成的地图
            completed = set(completed_levels[current_mode])
            current_map_index = None
            for i in range(30):
                if i not in completed:
                    current_map_index = i
                    break
            
            if current_map_index is None:
                continue  # 该难度全部完成，尝试下一个难度
            
            continue
        
        print(f"\n开始游戏：{current_mode} - 地图 {current_map_index + 1}")
        
        try:
            # 创建环境并加载地图
            env = PathFindingEnv(mode=current_mode)
            env.load_map(current_maps[current_map_index], current_map_index)
            
            # 初始化字体和状态跟踪
            pygame.font.init()
            font = pygame.font.Font(None, 20)
            stats_font = pygame.font.Font(None, 24)
            
            # 记录游戏开始时间和步数
            start_time = time.time()
            total_steps = 0
            killed_monsters = 0
            destroyed_obstacles = 0
            
            # 游戏主循环
            running = True
            game_result = None
            
            while running:
                # 处理输入
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        keep_playing = False
                    elif event.type == pygame.KEYDOWN:
                        action = None
                        
                        # 移动控制
                        if event.key == pygame.K_UP:
                            action = 0  # 向上移动1步
                        elif event.key == pygame.K_DOWN:
                            action = 3  # 向下移动1步
                        elif event.key == pygame.K_LEFT:
                            action = 6  # 向左移动1步
                        elif event.key == pygame.K_RIGHT:
                            action = 9  # 向右移动1步
                        elif event.key == pygame.K_ESCAPE:  # 退出游戏
                            running = False
                            keep_playing = False
                        
                        # 执行动作
                        if action is not None:
                            # 记录怪物和障碍物数量（用于计算击杀和破坏数量）
                            monsters_before = len(env.monsters)
                            obstacles_before = len(env.obstacles)
                            
                            # 执行动作并更新步数
                            obs, reward, done, _, _ = env.step(action)
                            total_steps += 1
                            
                            # 检查怪物击杀和障碍物破坏情况
                            if monsters_before > len(env.monsters):
                                killed_monsters += monsters_before - len(env.monsters)
                            if obstacles_before > len(env.obstacles):
                                destroyed_obstacles += obstacles_before - len(env.obstacles)
                            
                            if done:
                                # 提取并保存游戏统计数据
                                stats = _extract_game_stats(env, start_time, total_steps)
                                stats["killed_monsters"] = int(killed_monsters)  # 确保是Python原生int类型
                                stats["destroyed_obstacles"] = int(destroyed_obstacles)  # 确保是Python原生int类型
                                
                                _save_game_result(current_mode, current_map_index, stats)
                                
                                # 记录该关卡已完成
                                if current_map_index not in completed_levels[current_mode]:
                                    completed_levels[current_mode].append(current_map_index)
                                
                                # 显示游戏结果
                                if reward > 0:
                                    game_result = f"成功！完成地图 {current_map_index + 1}！得分：{env.score}"
                                else:
                                    game_result = f"失败！游戏结束！得分：{env.score}"
                                
                                # 等待2秒后进入下一关
                                result_time = time.time()
                                display_result = True
                                
                                while display_result and time.time() - result_time < 2.0:
                                    env.render()
                                    
                                    # 显示游戏结果
                                    result_text = stats_font.render(game_result, True, (0, 0, 0))
                                    text_rect = result_text.get_rect(center=(env.screen.get_width()//2, env.screen.get_height()//2))
                                    pygame.draw.rect(env.screen, (255, 255, 255, 200), text_rect.inflate(20, 20))
                                    env.screen.blit(result_text, text_rect)
                                    
                                    pygame.display.flip()
                                    
                                    for event in pygame.event.get():
                                        if event.type == pygame.QUIT:
                                            display_result = False
                                            running = False
                                            keep_playing = False
                                        elif event.type == pygame.KEYDOWN:
                                            display_result = False
                                
                                running = False  # 退出当前地图循环
                
                # 更新显示
                env.render()
                
                # 显示控制说明
                control_text = "方向键:移动 | ESC:退出"
                text_surface = font.render(control_text, True, (0, 0, 0))
                env.screen.blit(text_surface, (10, env.screen.get_height() - 20))
                
                # 显示实时评估指标
                current_stats = _extract_game_stats(env, start_time, total_steps)
                current_stats["killed_monsters"] = int(killed_monsters)  # 确保是Python原生int类型
                current_stats["destroyed_obstacles"] = int(destroyed_obstacles)  # 确保是Python原生int类型
                
                stats_text = [
                    f"当前得分: {int(current_stats['custom_score'])}",  # 确保是Python原生int类型
                    f"步数: {total_steps}",
                    f"探索率: {current_stats['exploration_rate']:.1%}",
                    f"金币: {current_stats['collected_coins']}/{COINS_COUNT}",
                    f"关卡进度: {current_mode} - 地图 {current_map_index + 1}/30",
                    f"总进度: {sum(len(levels) for levels in completed_levels.values())}/90"
                ]
                
                for i, line in enumerate(stats_text):
                    stats_surface = stats_font.render(line, True, (0, 0, 0))
                    env.screen.blit(stats_surface, (10, 10 + i * 25))
                
                pygame.display.flip()
            
            # 关闭环境
            env.close()
            
            # 如果是因为完成当前地图而退出循环，找到下一个未完成的地图
            if keep_playing:
                # 寻找下一个地图
                next_found = False
                
                # 首先在当前难度中查找
                for i in range(current_map_index + 1, 30):
                    if i not in completed_levels[current_mode]:
                        current_map_index = i
                        next_found = True
                        break
                
                # 如果当前难度没有更多未完成的地图，尝试下一个难度
                if not next_found:
                    if current_mode == MODE_LEVEL1:
                        next_mode = MODE_LEVEL2
                    elif current_mode == MODE_LEVEL2:
                        next_mode = MODE_LEVEL3
                    else:
                        next_mode = MODE_LEVEL1
                    
                    # 查找下一个难度中的第一个未完成地图
                    for i in range(30):
                        if i not in completed_levels[next_mode]:
                            current_mode = next_mode
                            current_map_index = i
                            next_found = True
                            break
                
                # 如果没有找到任何未完成的地图，游戏结束
                if not next_found:
                    print("恭喜！您已完成所有90个关卡！")
                    keep_playing = False
        
        except Exception as e:
            print(f"游戏运行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            keep_playing = False

def play_game(mode: str, map_index: int = 0):
    """使用预制地图运行游戏"""
    # 加载地图集合
    maps_dir = str(MAZE_EVAL_MAPS_DIR)
    collection_file = os.path.join(maps_dir, f"{mode.replace(' ', '_')}_collection.json")
    
    # 检查地图文件是否存在
    if not os.path.exists(collection_file):
        print(f"错误：找不到地图文件 {collection_file}")
        print("请先运行 python scripts/generate_maps.py 生成地图")
        return
    
    maps = MapGenerator.load_map_collection(collection_file)
    
    if map_index >= len(maps):
        print(f"错误：地图索引 {map_index} 超出范围（总共 {len(maps)} 个地图）")
        return
    
    try:
        # 创建环境并加载地图
        env = PathFindingEnv(mode=mode)
        env.load_map(maps[map_index], map_index)
        
        # 初始化字体
        pygame.font.init()
        font = pygame.font.Font(None, 20)
        
        # 游戏主循环
        running = True
        while running:
            # 处理输入
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    action = None
                    
                    # 移动控制
                    if event.key == pygame.K_UP:
                        action = 0  # 向上移动1步
                    elif event.key == pygame.K_DOWN:
                        action = 3  # 向下移动1步
                    elif event.key == pygame.K_LEFT:
                        action = 6  # 向左移动1步
                    elif event.key == pygame.K_RIGHT:
                        action = 9  # 向右移动1步
                    elif event.key == pygame.K_n:  # 切换到下一个地图
                        if map_index < len(maps) - 1:
                            map_index += 1
                            env.load_map(maps[map_index], map_index)
                    elif event.key == pygame.K_p:  # 切换到上一个地图
                        if map_index > 0:
                            map_index -= 1
                            env.load_map(maps[map_index], map_index)
                    elif event.key == pygame.K_r:  # 重置当前地图
                        env.load_map(maps[map_index], map_index)
                    elif event.key == pygame.K_ESCAPE:  # 退出游戏
                        running = False
                    
                    # 执行动作
                    if action is not None:
                        obs, reward, done, _, _ = env.step(action)
                        if done:
                            if reward > 0:
                                print(f"恭喜！完成地图 {map_index + 1}！得分：{env.score}")
                            else:
                                print(f"游戏结束！得分：{env.score}")
                            # 等待1秒后重置地图
                            pygame.time.wait(1000)
                            env.load_map(maps[map_index], map_index)  # 重置当前地图
            
            # 更新显示
            env.render()
            
            # 显示控制说明
            control_text = "方向键:移动 | N:下一图 | P:上一图 | R:重置 | ESC:退出"
            text_surface = font.render(control_text, True, (0, 0, 0))
            env.screen.blit(text_surface, (10, env.screen.get_height() - 20))
            
            pygame.display.flip()
    
    finally:
        # 确保在任何情况下都能正确关闭游戏
        env.close()

def main():
    print("欢迎来到寻路游戏！")
    print("\n选择游戏模式：")
    print("1. Level 1 (7x7地图，无怪物)")
    print("2. Level 2 (9x9地图，有怪物)")
    print("3. Level 3 (11x11地图，有怪物和道具)")
    print("4. 人类评估模式 (90关卡，记录得分)")
    
    while True:
        choice = input("\n请输入选择（1, 2, 3 或 4）：")
        if choice in ['1', '2', '3', '4']:
            if choice == '4':
                play_human_evaluation()
                return
            
            modes = {
                '1': MODE_LEVEL1,
                '2': MODE_LEVEL2,
                '3': MODE_LEVEL3
            }
            mode = modes[choice]
            break
        print("无效的选择，请重试。")
    
    map_index = 0
    while True:
        try:
            max_maps = 10  # 每个难度10张地图
            map_index = int(input(f"\n请输入要开始的地图编号（1-{max_maps}）：")) - 1
            if 0 <= map_index < max_maps:
                break
            print(f"地图编号必须在1-{max_maps}之间。")
        except ValueError:
            print("请输入有效的数字。")
    
    print("\n游戏控制：")
    print("- 方向键：移动")
    print("- N：下一个地图")
    print("- P：上一个地图")
    print("- R：重置当前地图")
    print("- ESC：退出游戏")
    
    if mode == MODE_LEVEL3:
        print("\n道具说明：")
        print("- 棕色方块 (T): 铲子，可破坏障碍3次")
        print("- 银色方块 (W): 剑，可击杀怪物")
        print("- 青色方块 (N): 磁铁，可吸取周围金币")
        print("- 金色方块 (K): 钥匙，必须拾取才能通关")
    
    input("\n按回车键开始游戏...")
    play_game(mode, map_index)

if __name__ == "__main__":
    main() 
