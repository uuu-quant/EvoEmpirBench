import pygame
import sys
import os
import time
import argparse
from typing import Dict, Any, Optional
import re
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.game.environment import PathFindingEnv
from src.game.map_generator import MapGenerator
from src.agent.agent_interface import DeepSeekAgent
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3, DISCOVERED, COINS_COUNT
from src.config.api_config import DEFAULT_API_TYPE
from src.config.paths import MAZE_EVAL_MAPS_DIR
from src.data_collector.game_data_collector import GameDataCollector  # 导入数据收集器

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="用LLM代理玩迷宫游戏")
    # 这里可以通过--mode参数指定游戏级别，默认为MODE_LEVEL1
    # 示例用法: python examples/play_with_agent.py --mode "level 1"
    # 或者:     python examples/play_with_agent.py --mode "level 2"
    parser.add_argument("--mode", type=str, default=MODE_LEVEL1,
                        help=f"游戏模式，选项: {MODE_LEVEL1}, {MODE_LEVEL2} 或 {MODE_LEVEL3}")
    parser.add_argument("--map_index", type=int, default=0,
                        help="起始地图索引 (0-9)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥；默认从环境变量读取")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称；默认根据api_type读取环境变量")
    parser.add_argument("--api_type", type=str, default=DEFAULT_API_TYPE, choices=["deepseek", "openai"],
                        help="API类型")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="每步之间的延迟（秒）")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="每张地图的最大步数")
    parser.add_argument("--show_prompt", action="store_true",
                        help="显示发送给模型的提示文本")
    parser.add_argument("--hide_prompt", action="store_true",
                        help="隐藏发送给模型的提示文本")
    # 修改数据收集相关参数，默认为开启
    parser.add_argument("--no_collect_data", action="store_true",
                        help="禁用数据收集功能")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据保存目录，默认为'outputs/collected_data'")
    parser.add_argument("--max_episodes", type=int, default=10,
                        help="最大收集的回合数，默认为10个回合")
    return parser.parse_args()

def extract_game_state(env: PathFindingEnv) -> Dict[str, Any]:
    """从环境中提取游戏状态"""
    # 首先计算最新分数，确保传递给代理的是最新分数
    discovered_count = np.sum(env.vision_map == DISCOVERED) - env.initial_discovered
    exploration_score = discovered_count * 10
    coin_score = (COINS_COUNT - len(env.coins)) * 500
    step_penalty = env.steps_count * 50  # 确保使用环境中的步数计数
    life_bonus = env.lives * 1000
    
    # 计算当前分数
    current_score = exploration_score + coin_score - step_penalty + life_bonus
    
    # 更新环境对象中的分数
    env.score = current_score
    
    # 创建游戏状态对象
    game_state = {
        'grid': env.grid,
        'vision_map': env.vision_map,
        'agent_pos': env.agent_pos,
        'coins': env.coins,
        'lives': env.lives,
        'score': current_score  # 使用刚刚计算的最新分数
    }
    
    # 如果是Level 3模式，添加道具信息
    if env.mode == MODE_LEVEL3:
        game_state.update({
            'has_shovel': env.has_shovel,
            'shovel_uses': env.shovel_uses,
            'has_sword': env.has_sword,
            'has_magnet': env.has_magnet,
            'has_key': env.has_key
        })
    
    return game_state

def play_with_agent(mode: str, map_index: int = 0, api_key: str = None,
                    model: str = None, delay: float = 1.0, max_steps: int = 100,
                    show_prompt: bool = True, collect_data: bool = False, 
                    data_dir: str = None, max_episodes: int = 10, api_type: str = DEFAULT_API_TYPE):
    """使用LLM代理玩游戏"""
    # 确保maps目录存在
    maps_dir = str(MAZE_EVAL_MAPS_DIR)
    os.makedirs(maps_dir, exist_ok=True)
    
    collection_file = os.path.join(maps_dir, f"{mode.replace(' ', '_')}_collection.json")
    
    # 检查地图文件是否存在
    if not os.path.exists(collection_file):
        print(f"错误：找不到地图文件 {collection_file}")
        print("正在生成地图...")
        
        # 调用地图生成脚本
        try:
            MapGenerator.generate_maps(10, mode, maps_dir)
            print(f"地图生成成功！")
        except Exception as e:
            print(f"地图生成失败: {str(e)}")
            print("请手动运行 python scripts/generate_maps.py 生成地图")
            return
    
    maps = MapGenerator.load_map_collection(collection_file)
    
    if map_index >= len(maps):
        print(f"错误：地图索引 {map_index} 超出范围（总共 {len(maps)} 个地图）")
        return
    
    # 初始化代理
    try:
        agent = DeepSeekAgent(api_key=api_key, model=model, api_type=api_type)
        agent.set_show_prompt(show_prompt)  # 设置是否显示提示
        agent.set_mode(mode)  # 设置游戏模式
        print(f"已初始化LLM代理，API类型: {agent.api_type}, 模型: {agent.model}")
        if show_prompt:
            print("已启用提示显示模式，将显示发送给模型的完整文本")
    except Exception as e:
        print(f"初始化代理失败: {str(e)}")
        print("请检查API密钥和网络连接。")
        return
    
    # 初始化数据收集器
    data_collector = None
    if collect_data:
        try:
            data_collector = GameDataCollector(data_dir)
            print(f"已启用数据收集功能，数据将保存到: {data_collector.data_dir}")
        except Exception as e:
            print(f"初始化数据收集器失败: {str(e)}")
            print("数据收集功能将被禁用")
            collect_data = False
    
    episodes_completed = 0
    
    try:
        # 创建环境并加载地图
        env = PathFindingEnv(mode=mode)
        env.load_map(maps[map_index], map_index)
        
        print("\n游戏开始！代理将自动控制角色探索迷宫。")
        print("按ESC键可随时退出游戏。")
        print(f"开始游玩地图 {map_index+1}/{len(maps)}\n")
        
        # 如果启用了数据收集，开始记录新回合
        if collect_data:
            data_collector.start_new_episode(map_index, maps[map_index])
        
        # 游戏主循环
        running = True
        steps = 0
        total_reward = 0
        
        # 创建状态文本字体
        pygame.font.init()
        font = pygame.font.Font(None, 24)
        
        while running and steps < max_steps:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 获取游戏状态
            game_state = extract_game_state(env)
            
            # 渲染当前状态
            env.render()
            
            # 显示代理状态
            agent_status = f"Thinking... (Steps: {steps}/{max_steps})"
            text_surface = font.render(agent_status, True, (0, 0, 0))
            env.screen.blit(text_surface, (10, 10))
            pygame.display.flip()
            
            # 获取代理的动作
            try:
                action, reasoning = agent.get_action(game_state)
                
                # 显示代理的推理过程
                print(f"\n===== Step {steps + 1} =====")
                action_meaning = env.get_action_meaning(action)
                print(f"Agent selected action: {action} ({action_meaning})")
                
                # 记录原始推理过程，同时突出显示动作部分
                highlighted_reasoning = reasoning
                # 尝试找到并高亮"Action:"部分
                action_choice_match = re.search(r"(Action\s*[:：]\s*\d+)", reasoning)
                if action_choice_match:
                    action_text = action_choice_match.group(1)
                    highlighted_reasoning = reasoning.replace(
                        action_text, 
                        f"\n>>> {action_text} <<<\n"
                    )
                
                print(f"Reasoning process:\n{highlighted_reasoning}\n")
            except Exception as e:
                print(f"Failed to get agent action: {str(e)}")
                print("Using default action (0)...")
                action = 0
                reasoning = "获取代理动作失败，使用默认动作（向上移动1步）。"
            
            # 保存执行动作前的状态用于数据收集
            if collect_data:
                state_before = extract_game_state(env)
            
            # 执行动作
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # 如果启用了数据收集，记录这一步的数据
            if collect_data:
                state_after = extract_game_state(env)
                data_collector.record_step(
                    state_before=state_before,
                    action=action,
                    reasoning=reasoning,
                    reward=reward,
                    state_after=state_after,
                    done=done
                )
            
            # 计算当前分数详情
            try:
                discovered_count = np.sum(env.vision_map == DISCOVERED) - env.initial_discovered
                exploration_score = discovered_count * 10
                coin_score = (COINS_COUNT - len(env.coins)) * 500
                step_penalty = env.steps_count * 50  # 使用环境中的steps_count而不是局部变量steps
                life_bonus = env.lives * 1000
                
                # 计算道具奖励（如果有）
                item_bonus = 0
                if env.mode == MODE_LEVEL3:
                    shovel_bonus = 100 if env.has_shovel else 0
                    sword_bonus = 200 if env.has_sword else 0
                    magnet_bonus = 150 if env.has_magnet else 0
                    key_bonus = 300 if env.has_key else 0
                    item_bonus = shovel_bonus + sword_bonus + magnet_bonus + key_bonus
                
                # 更新当前分数计算，与结尾计算公式保持一致
                current_score = exploration_score + coin_score + item_bonus - step_penalty + life_bonus
                
                # 关键修改：将计算的当前分数更新到环境对象中
                env.score = current_score
                
                # 显示动作结果和分数信息，包含完整的分数明细
                action_text = f"Executed action: {action} ({env.get_action_meaning(action)})"
                score_text = f"Current score: {current_score} (Exploration: +{exploration_score}, Coins: +{coin_score}, Lives: +{life_bonus}, Steps: -{step_penalty})"
                
                text_surface1 = font.render(action_text, True, (0, 0, 0))
                text_surface2 = font.render(score_text, True, (0, 0, 0))
                
                env.screen.blit(text_surface1, (10, 40))
                env.screen.blit(text_surface2, (10, 70))
                
                # 额外添加分数细节显示
                exploration_text = f"Exploration: +{exploration_score} ({discovered_count} cells × 10 pts)"
                coins_text = f"Coins: +{coin_score} ({COINS_COUNT - len(env.coins)}/{COINS_COUNT} coins × 500 pts)"
                lives_text = f"Lives bonus: +{life_bonus} ({env.lives} lives × 1000 pts)"
                steps_text = f"Step penalty: -{step_penalty} ({env.steps_count} steps × 50 pts)"
                
                text_surface3 = font.render(exploration_text, True, (0, 0, 0))
                text_surface4 = font.render(coins_text, True, (0, 0, 0))
                text_surface5 = font.render(lives_text, True, (0, 0, 0))
                text_surface6 = font.render(steps_text, True, (0, 0, 0))
                
                env.screen.blit(text_surface3, (10, 100))
                env.screen.blit(text_surface4, (10, 130))
                env.screen.blit(text_surface5, (10, 160))
                env.screen.blit(text_surface6, (10, 190))
                
                # 如果是Level 3，显示道具状态
                if env.mode == MODE_LEVEL3:
                    y_pos = 220
                    
                    # 钥匙状态
                    key_status = "Key: Found ✓" if env.has_key else "Key: Not found ✗ (Required!)"
                    key_surface = font.render(key_status, True, (0, 0, 0))
                    env.screen.blit(key_surface, (10, y_pos))
                    y_pos += 30
                    
                    # 铲子状态
                    if env.has_shovel:
                        shovel_status = f"Shovel: Active ({env.shovel_uses} uses left)"
                    else:
                        shovel_status = "Shovel: Not found"
                    shovel_surface = font.render(shovel_status, True, (0, 0, 0))
                    env.screen.blit(shovel_surface, (10, y_pos))
                    y_pos += 30
                    
                    # 剑状态
                    sword_status = "Sword: Equipped ✓" if env.has_sword else "Sword: Not found"
                    sword_surface = font.render(sword_status, True, (0, 0, 0))
                    env.screen.blit(sword_surface, (10, y_pos))
                    y_pos += 30
                    
                    # 磁铁状态
                    magnet_status = "Magnet: Active ✓" if env.has_magnet else "Magnet: Not found"
                    magnet_surface = font.render(magnet_status, True, (0, 0, 0))
                    env.screen.blit(magnet_surface, (10, y_pos))
            except Exception as e:
                print(f"Error calculating score: {str(e)}")
                # 显示简化版本的分数信息
                action_text = f"Executed action: {action} ({env.get_action_meaning(action)})"
                text_surface = font.render(action_text, True, (0, 0, 0))
                env.screen.blit(text_surface, (10, 40))
            
            pygame.display.flip()
            
            # 延迟，以便能看清代理的动作
            time.sleep(delay)
            
            # 检查是否完成或失败
            if done:
                try:
                    # 计算详细分数，与游戏过程中的计算保持一致
                    discovered_count = np.sum(env.vision_map == DISCOVERED) - env.initial_discovered
                    exploration_score = discovered_count * 10
                    coin_score = (COINS_COUNT - len(env.coins)) * 500
                    step_penalty = env.steps_count * 50  # 使用环境中的steps_count
                    life_bonus = env.lives * 1000  # 无论是否成功都计算生命奖励
                    
                    # 道具奖励（Level 3）
                    item_bonus = 0
                    if env.mode == MODE_LEVEL3:
                        shovel_bonus = 100 if env.has_shovel else 0
                        sword_bonus = 200 if env.has_sword else 0
                        magnet_bonus = 150 if env.has_magnet else 0
                        key_bonus = 300 if env.has_key else 0
                        item_bonus = shovel_bonus + sword_bonus + magnet_bonus + key_bonus
                    
                    # 计算总分，确保与游戏过程中的计算一致
                    final_score = exploration_score + coin_score + item_bonus - step_penalty + life_bonus
                    
                    score_breakdown = {
                        "Exploration reward": f"+{exploration_score} pts ({discovered_count} new areas × 10 pts)",
                        "Coin reward": f"+{coin_score} pts ({COINS_COUNT - len(env.coins)} coins × 500 pts)",
                        "Step penalty": f"-{step_penalty} pts ({env.steps_count} steps × 50 pts)",
                        "Life bonus": f"+{life_bonus} pts ({env.lives} lives × 1000 pts)"
                    }
                    
                    # 如果是Level 3，添加道具奖励
                    if env.mode == MODE_LEVEL3:
                        score_breakdown["Item bonus"] = f"+{item_bonus} pts (Shovel: {shovel_bonus}, Sword: {sword_bonus}, Magnet: {magnet_bonus}, Key: {key_bonus})"
                    
                    score_breakdown["Total score"] = f"{final_score} pts"
                    
                    if reward > 0:
                        print(f"\nCongratulations! Map {map_index + 1} completed!")
                    else:
                        print(f"\nMap {map_index + 1} failed.")
                    
                    print("\nScore details:")
                    for category, score in score_breakdown.items():
                        print(f"  {category}: {score}")
                except Exception as e:
                    print(f"Error calculating score details: {str(e)}")
                    if reward > 0:
                        print(f"\nCongratulations! Map {map_index + 1} completed! Total score: {total_reward}")
                    else:
                        print(f"\nMap {map_index + 1} failed. Total score: {total_reward}")
                
                # 结束当前回合数据收集
                if collect_data:
                    data_collector.end_episode()
                    episodes_completed += 1
                
                # 等待一段时间后继续下一张地图或结束
                completion_text = "Success!" if reward > 0 else "Failed!"
                text_surface = font.render(completion_text, True, (255, 0, 0))
                env.screen.blit(text_surface, (env.screen.get_width()//2 - 50, env.screen.get_height()//2))
                pygame.display.flip()
                time.sleep(2)
                
                # 检查是否达到最大回合数
                if collect_data and episodes_completed >= max_episodes:
                    print(f"已达到最大收集回合数 ({max_episodes}), 停止游戏并保存数据")
                    break
                
                # 如果还有下一张地图，加载它
                if map_index < len(maps) - 1:
                    map_index += 1
                    env.load_map(maps[map_index], map_index)
                    steps = 0
                    total_reward = 0
                    agent.reset()  # 重置代理的对话历史
                    print(f"\nStarting map {map_index+1}/{len(maps)}\n")
                    
                    # 开始新回合数据收集
                    if collect_data:
                        data_collector.start_new_episode(map_index, maps[map_index])
                else:
                    print("All maps completed!")
                    running = False
            
            # 如果达到最大步数限制
            if steps >= max_steps and running:
                print(f"Maximum steps limit reached ({max_steps}), switching to next map.")
                
                # 结束当前回合数据收集
                if collect_data:
                    data_collector.end_episode()
                    episodes_completed += 1
                
                # 检查是否达到最大回合数
                if collect_data and episodes_completed >= max_episodes:
                    print(f"已达到最大收集回合数 ({max_episodes}), 停止游戏并保存数据")
                    break
                
                # 如果还有下一张地图，加载它
                if map_index < len(maps) - 1:
                    map_index += 1
                    env.load_map(maps[map_index], map_index)
                    steps = 0
                    total_reward = 0
                    agent.reset()  # 重置代理的对话历史
                    print(f"\nStarting map {map_index+1}/{len(maps)}\n")
                    
                    # 开始新回合数据收集
                    if collect_data:
                        data_collector.start_new_episode(map_index, maps[map_index])
                else:
                    print("All maps attempted!")
                    running = False
    
    finally:
        # 保存收集的数据
        if collect_data and data_collector is not None:
            try:
                saved_file = data_collector.save_data()
                print(f"已收集 {episodes_completed} 个回合的数据，并保存到: {saved_file}")
                
                # 显示统计信息
                stats = data_collector.get_summary_statistics()
                print("\n数据统计信息:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"保存数据时出错: {str(e)}")
        
        # 确保在任何情况下都能正确关闭游戏
        env.close()

def main():
    args = parse_args()
    show_prompt = True  # 默认显示提示
    
    # 处理提示显示选项
    if args.hide_prompt:
        show_prompt = False
    elif args.show_prompt:
        show_prompt = True
    
    # 处理数据收集选项
    collect_data = not args.no_collect_data  # 默认开启数据收集
    
    print(f"启动LLM代理模式...")
    print(f"API类型: {args.api_type}")
    print(f"游戏模式: {args.mode}")
    print(f"起始地图索引: {args.map_index}")
    print(f"每步延迟: {args.delay}秒")
    print(f"每张地图最大步数: {args.max_steps}")
    print(f"提示显示: {'开启' if show_prompt else '关闭'}")
    
    # 显示数据收集相关信息
    if collect_data:
        print(f"数据收集: 开启")
        print(f"最大收集回合数: {args.max_episodes}")
        if args.data_dir:
            print(f"数据保存目录: {args.data_dir}")
    else:
        print(f"数据收集: 关闭")
    
    play_with_agent(
        mode=args.mode,
        map_index=args.map_index,
        api_key=args.api_key,
        model=args.model,
        api_type=args.api_type,
        delay=args.delay,
        max_steps=args.max_steps,
        show_prompt=show_prompt,
        collect_data=collect_data,
        data_dir=args.data_dir,
        max_episodes=args.max_episodes
    )

if __name__ == "__main__":
    main() 
