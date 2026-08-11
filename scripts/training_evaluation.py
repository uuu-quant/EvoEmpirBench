#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional
import time
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from src.config.paths import MAZE_TRAIN_MAPS_DIR, RESULTS_DIR
from src.agent.agent_interface import DeepSeekAgent
from src.game.environment import PathFindingEnv

class TrainingMapEvaluator:
    """训练地图评估器，用于评估AI代理在训练地图上的表现"""
    
    def __init__(self, 
                 map_dir: str = None, 
                 results_dir: str = None,
                 api_type: str = "deepseek",
                 model: str = "deepseek-reasoner",
                 api_key: str = None,
                 base_url: str = None,
                 maps_per_level: int = 50,
                 max_steps: int = 200,
                 mode: str = None,
                 resume: bool = True,
                 start_map_index: int = 0):
        """
        初始化评估器
        
        Args:
            map_dir: 训练地图目录，默认为项目根目录下的 data/levels/maze_train
            results_dir: 结果保存目录，默认为项目根目录下的 outputs/results/{model}/training
            api_type: API类型，deepseek或openai
            model: 模型名称
            api_key: API密钥
            base_url: API的基础URL
            maps_per_level: 每个难度级别评估的地图数量
            max_steps: 每张地图的最大步数
            mode: 要评估的游戏模式，None表示评估所有模式
            resume: 是否继续上次的评估，跳过已评估的地图
            start_map_index: 起始地图索引
        """
        # 设置训练地图目录
        self.map_dir = map_dir or str(MAZE_TRAIN_MAPS_DIR)
        
        # 设置结果保存目录
        self.model = model
        self.results_dir = results_dir or os.path.join(str(RESULTS_DIR), model, 'training')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # API设置
        self.api_type = api_type
        self.api_key = api_key
        self.base_url = base_url
        
        # 评估设置
        self.maps_per_level = maps_per_level
        self.max_steps = max_steps
        self.mode = mode
        self.resume = resume
        self.start_map_index = start_map_index
        
        # 初始化代理和环境
        self.agent = None
        self.env = None
        
        # 初始化结果存储
        self.results = {
            MODE_LEVEL1: [],
            MODE_LEVEL2: [],
            MODE_LEVEL3: []
        }
        
        # 加载已有结果（如果启用续传）
        if self.resume:
            self._load_existing_results()
        
        # 初始化统计数据
        self.stats = {mode: defaultdict(list) for mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]}
    
    def _load_existing_results(self):
        """加载已有评估结果（用于续传）"""
        result_files = glob.glob(os.path.join(self.results_dir, "evaluation_results_*.json"))
        if result_files:
            latest_result_file = sorted(result_files)[-1]
            print(f"找到已有评估结果: {latest_result_file}，启用续传模式")
            try:
                with open(latest_result_file, 'r') as f:
                    results_data = json.load(f)
                    if "results" in results_data:
                        for mode, mode_results in results_data["results"].items():
                            self.results[mode] = mode_results
                        print(f"已加载已有结果:")
                        for mode, mode_results in self.results.items():
                            print(f"- {mode}: {len(mode_results)}张地图")
            except Exception as e:
                print(f"加载已有结果失败: {str(e)}")
    
    def _init_agent(self):
        """初始化AI代理"""
        if self.agent is None:
            print(f"正在初始化代理... (类型: {self.api_type}, 模型: {self.model})")
            self.agent = DeepSeekAgent(
                api_key=self.api_key,
                model=self.model,
                api_type=self.api_type,
                base_url=self.base_url
            )
    
    def _init_environment(self, mode: str):
        """初始化游戏环境"""
        if self.env is None or self.env.mode != mode:
            print(f"正在初始化环境... (模式: {mode})")
            # 关闭现有环境
            if self.env is not None:
                self.env.close()
            
            # 创建新环境
            self.env = PathFindingEnv(mode=mode)
            
            # 设置代理模式
            if self.agent is not None:
                self.agent.set_mode(mode)
    
    def _load_map_file(self, mode: str, map_index: int):
        """加载特定的地图文件"""
        map_filename = f"{mode.replace(' ', '_')}_training_{map_index+1}.json"
        map_filepath = os.path.join(self.map_dir, map_filename)
        
        if not os.path.exists(map_filepath):
            # 尝试查找集合文件
            collection_filename = f"{mode.replace(' ', '_')}_collection.json"
            collection_filepath = os.path.join(self.map_dir, collection_filename)
            
            if os.path.exists(collection_filepath):
                print(f"正在从集合文件加载地图: {collection_filepath}, 索引: {map_index}")
                with open(collection_filepath, 'r') as f:
                    maps_collection = json.load(f)
                
                if map_index < len(maps_collection):
                    return maps_collection[map_index]
                else:
                    print(f"警告: 地图索引 {map_index} 超出集合范围 {len(maps_collection)}")
                    return None
            else:
                print(f"错误: 未找到地图 {map_filepath} 或集合 {collection_filepath}")
                return None
        
        # 加载单个地图文件
        print(f"正在加载地图: {map_filepath}")
        with open(map_filepath, 'r') as f:
            return json.load(f)
    
    def _save_results(self):
        """保存评估结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_data = {
            "model": self.model,
            "api_type": self.api_type,
            "maps_per_level": self.maps_per_level,
            "max_steps": self.max_steps,
            "results": self.results,
            "timestamp": timestamp
        }
        
        result_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"结果已保存至: {result_file}")
    
    def _evaluate_map(self, mode: str, map_index: int):
        """评估单张地图"""
        # 检查是否已评估（续传模式）
        if self.resume and map_index < len(self.results[mode]):
            print(f"跳过已评估的地图: {mode}, 索引 {map_index}")
            return self.results[mode][map_index]
        
        # 加载地图
        map_data = self._load_map_file(mode, map_index)
        if map_data is None:
            print(f"无法加载地图: {mode}, 索引 {map_index}")
            return None
        
        # 初始化环境和代理
        self._init_environment(mode)
        self._init_agent()
        
        # 加载地图到环境
        self.env.load_map(map_data, map_index)
        
        # 评估结果
        result = {
            "map_index": map_index,
            "success": False,
            "score": 0,
            "steps": 0,
            "api_calls": 0,
            "exploration_rate": 0,
            "collected_coins": 0,
            "lives_remaining": 0,
            "custom_score": 0  # 自定义得分
        }
        
        # 记录初始状态
        state = self.env.get_state_dict()
        total_cells = self.env.grid.shape[0] * self.env.grid.shape[1]
        initial_discovered = np.sum(state["vision_map"] == 1)
        total_coins = len(state["coins"])
        
        # 游戏循环
        done = False
        for step in range(self.max_steps):
            # 获取当前状态
            state = self.env.get_state_dict()
            
            # 获取代理动作
            action, response = self.agent.get_action(state)
            
            # 执行动作
            _, reward, done, truncated, info = self.env.step(action)
            
            # 可视化（可选）
            # self.env.render()
            
            # 更新统计信息
            result["steps"] = step + 1
            result["api_calls"] += 1
            
            # 完成游戏或达到最大步数时退出
            if done or step + 1 >= self.max_steps:
                break
        
        # 记录最终状态
        final_state = self.env.get_state_dict()
        result["score"] = final_state["score"]
        result["success"] = done and final_state["score"] > 0  # 成功完成地图
        result["lives_remaining"] = final_state["lives"]
        
        # 计算探索率
        final_discovered = np.sum(final_state["vision_map"] == 1)
        result["exploration_rate"] = (final_discovered - initial_discovered) / (total_cells - initial_discovered)
        
        # 计算收集的金币数量
        result["collected_coins"] = total_coins - len(final_state["coins"])
        
        # 计算自定义得分（综合考虑多个指标）
        # 使用加权分数：50%成功率 + 20%探索率 + 20%金币收集率 + 10%剩余生命
        success_weight = 0.5
        exploration_weight = 0.2
        coins_weight = 0.2
        lives_weight = 0.1
        
        success_score = 1.0 if result["success"] else 0.0
        exploration_score = result["exploration_rate"]
        coins_score = result["collected_coins"] / max(1, total_coins)
        lives_score = final_state["lives"] / 3.0  # 假设最大生命值为3
        
        result["custom_score"] = (
            success_weight * success_score +
            exploration_weight * exploration_score +
            coins_weight * coins_score +
            lives_weight * lives_score
        )
        
        print(f"地图 {mode}, 索引 {map_index} 评估完成:")
        print(f"  成功: {result['success']}, 得分: {result['score']}, 步数: {result['steps']}")
        print(f"  探索率: {result['exploration_rate']:.2%}, 收集金币: {result['collected_coins']}/{total_coins}")
        print(f"  剩余生命: {result['lives_remaining']}, 自定义得分: {result['custom_score']:.2f}")
        
        return result
    
    def evaluate(self):
        """评估所有地图"""
        modes = [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]
        
        # 如果指定了特定模式，只评估该模式
        if self.mode:
            if self.mode in modes:
                modes = [self.mode]
            else:
                print(f"无效的模式: {self.mode}")
                return
        
        # 对每个模式进行评估
        for mode in modes:
            print(f"\n===== 开始评估 {mode} =====")
            
            for map_index in range(self.start_map_index, self.maps_per_level):
                print(f"\n评估地图 {map_index+1}/{self.maps_per_level} ({mode})")
                
                # 评估单张地图
                result = self._evaluate_map(mode, map_index)
                
                # 如果是新结果，添加到结果列表
                if result is not None and (not self.resume or map_index >= len(self.results[mode])):
                    # 确保结果列表长度等于当前索引（处理跳过的索引）
                    while len(self.results[mode]) <= map_index:
                        self.results[mode].append(None)
                    
                    # 更新结果
                    self.results[mode][map_index] = result
                    
                    # 定期保存结果（每5张地图）
                    if (map_index + 1) % 5 == 0:
                        self._save_results()
            
            print(f"\n{mode} 评估完成，共 {len([r for r in self.results[mode] if r is not None])} 张地图")
        
        # 保存最终结果
        self._save_results()
        
        # 分析结果
        self.analyze_results()
    
    def analyze_results(self):
        """分析评估结果并生成统计信息"""
        print("\n===== 结果分析 =====")
        
        # 准备数据
        stats_data = []
        
        # 处理每个模式的结果
        for mode, mode_results in self.results.items():
            # 跳过空结果
            valid_results = [r for r in mode_results if r is not None]
            if not valid_results:
                continue
            
            # 计算统计指标
            total_maps = len(valid_results)
            success_count = sum(1 for r in valid_results if r.get("success", False))
            success_rate = success_count / total_maps if total_maps > 0 else 0
            
            avg_score = np.mean([r.get("score", 0) for r in valid_results])
            avg_custom_score = np.mean([r.get("custom_score", 0) for r in valid_results])
            avg_steps = np.mean([r.get("steps", 0) for r in valid_results])
            avg_api_calls = np.mean([r.get("api_calls", 0) for r in valid_results])
            avg_exploration = np.mean([r.get("exploration_rate", 0) for r in valid_results])
            
            # 计算金币收集率
            avg_coins_collected = np.mean([r.get("collected_coins", 0) for r in valid_results])
            
            # 计算平均剩余生命
            avg_lives = np.mean([r.get("lives_remaining", 0) for r in valid_results])
            
            # 记录统计结果
            stats_data.append({
                "模型": self.model,
                "难度级别": mode,
                "样本数": total_maps,
                "成功率": success_rate,
                "平均得分": avg_score,
                "自定义得分": avg_custom_score,
                "平均步数": avg_steps,
                "平均API调用数": avg_api_calls,
                "平均探索率": avg_exploration,
                "平均金币收集数": avg_coins_collected,
                "平均剩余生命": avg_lives
            })
        
        # 如果有数据，计算所有级别的综合统计
        if stats_data:
            all_modes_stats = {
                "模型": self.model,
                "难度级别": "所有级别",
                "样本数": sum(s["样本数"] for s in stats_data),
                "成功率": np.mean([s["成功率"] for s in stats_data]),
                "平均得分": np.mean([s["平均得分"] for s in stats_data]),
                "自定义得分": np.mean([s["自定义得分"] for s in stats_data]),
                "平均步数": np.mean([s["平均步数"] for s in stats_data]),
                "平均API调用数": np.mean([s["平均API调用数"] for s in stats_data]),
                "平均探索率": np.mean([s["平均探索率"] for s in stats_data]),
                "平均金币收集数": np.mean([s["平均金币收集数"] for s in stats_data]),
                "平均剩余生命": np.mean([s["平均剩余生命"] for s in stats_data])
            }
            stats_data.append(all_modes_stats)
        
        # 转换为DataFrame
        stats_df = pd.DataFrame(stats_data)
        
        # 格式化百分比
        format_cols = ["成功率", "平均探索率"]
        for col in format_cols:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].apply(lambda x: f"{x:.2%}")
        
        # 保存为CSV和Excel
        csv_path = os.path.join(self.results_dir, "training_stats_summary.csv")
        xlsx_path = os.path.join(self.results_dir, "training_stats_summary.xlsx")
        
        stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        try:
            stats_df.to_excel(xlsx_path, index=False)
            print(f"统计结果已保存至: {csv_path} 和 {xlsx_path}")
        except Exception as e:
            print(f"保存Excel文件失败: {str(e)}")
            print(f"统计结果已保存至: {csv_path}")
        
        # 打印结果
        print("\n训练地图评估统计:")
        from tabulate import tabulate
        print(tabulate(stats_df, headers='keys', tablefmt='grid', showindex=False))
        
        # 生成可视化
        self._create_visualization(stats_df)
    
    def _create_visualization(self, stats_df):
        """创建结果可视化"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取按难度级别的数据
        levels_df = stats_df[stats_df["难度级别"] != "所有级别"].copy()
        
        # 将百分比字符串转换回浮点数
        for col in ["成功率", "平均探索率"]:
            if col in levels_df.columns:
                levels_df[col] = levels_df[col].str.rstrip('%').astype('float') / 100
        
        # 1. 按难度级别的成功率
        plt.figure(figsize=(10, 6))
        plt.bar(levels_df["难度级别"], levels_df["成功率"])
        plt.title(f"模型 {self.model} 在不同难度级别的成功率")
        plt.xlabel("难度级别")
        plt.ylabel("成功率")
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_success_rate_by_level.png"))
        plt.close()
        
        # 2. 按难度级别的自定义得分
        plt.figure(figsize=(10, 6))
        plt.bar(levels_df["难度级别"], levels_df["自定义得分"])
        plt.title(f"模型 {self.model} 在不同难度级别的自定义得分")
        plt.xlabel("难度级别")
        plt.ylabel("自定义得分")
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_custom_score_by_level.png"))
        plt.close()
        
        # 3. 按难度级别的探索率和平均金币收集
        plt.figure(figsize=(12, 6))
        width = 0.35
        x = np.arange(len(levels_df))
        
        plt.bar(x - width/2, levels_df["平均探索率"], width, label="平均探索率")
        
        # 计算金币收集率（假设每个地图最多5个金币）
        coin_collection_rate = levels_df["平均金币收集数"] / 5
        plt.bar(x + width/2, coin_collection_rate, width, label="金币收集率")
        
        plt.title(f"模型 {self.model} 在不同难度级别的探索率和金币收集率")
        plt.xlabel("难度级别")
        plt.ylabel("比率")
        plt.xticks(x, levels_df["难度级别"])
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_exploration_coins_by_level.png"))
        plt.close()
        
        print(f"可视化结果已保存至: {self.results_dir}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估AI代理在训练地图上的表现")
    parser.add_argument("--api_type", type=str, default="deepseek", choices=["deepseek", "openai"],
                       help="API类型 (deepseek 或 openai，默认: deepseek)")
    parser.add_argument("--model", type=str, default="deepseek-reasoner",
                       help="模型名称 (默认: deepseek-reasoner 或 gpt-4o，取决于API类型)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API密钥 (默认从环境变量获取)")
    parser.add_argument("--base_url", type=str, default=None,
                       help="API基础URL (仅当api_type=openai时使用)")
    parser.add_argument("--map_dir", type=str, default=None,
                       help="训练地图目录 (默认: ./data/levels/maze_train)")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="结果保存目录 (默认: ./outputs/results/{model}/training)")
    parser.add_argument("--num_maps", type=int, default=50,
                       help="每个难度评估的地图数量 (默认: 50)")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="每张地图的最大步数 (默认: 200)")
    parser.add_argument("--mode", type=str, default=None, choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3],
                       help="仅评估指定模式 (不指定则评估所有模式)")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="启用续传模式，跳过已评估的地图")
    parser.add_argument("--no_resume", action="store_false", dest="resume",
                       help="禁用续传模式，重新评估所有地图")
    parser.add_argument("--start_map_index", type=int, default=0,
                       help="从指定索引的地图开始评估 (0-49，默认: 0)")
    
    args = parser.parse_args()
    
    # 根据API类型设置默认模型
    if args.api_type == "openai" and args.model == "deepseek-reasoner":
        args.model = "gpt-4o"
    
    print("==== 训练地图评估 ====")
    print(f"API类型: {args.api_type}")
    print(f"模型: {args.model}")
    print(f"地图数量: {args.num_maps}")
    print(f"最大步数: {args.max_steps}")
    print(f"续传模式: {args.resume}")
    print(f"起始地图索引: {args.start_map_index}")
    if args.mode:
        print(f"评估模式: {args.mode}")
    else:
        print("评估模式: 全部")
    
    # 创建评估器
    evaluator = TrainingMapEvaluator(
        map_dir=args.map_dir,
        results_dir=args.results_dir,
        api_type=args.api_type,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        maps_per_level=args.num_maps,
        max_steps=args.max_steps,
        mode=args.mode,
        resume=args.resume,
        start_map_index=args.start_map_index
    )
    
    # 开始评估
    evaluator.evaluate()

if __name__ == "__main__":
    main() 
