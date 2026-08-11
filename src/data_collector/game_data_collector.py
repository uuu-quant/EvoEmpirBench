import os
import json
import time
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np

from src.config.paths import COLLECTED_DATA_DIR

class GameDataCollector:
    """游戏数据收集器，用于收集游戏中的各种数据，如动作、奖励、状态等"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据收集器
        
        Args:
            data_dir: 数据保存目录，如果为None，则使用默认目录
        """
        # 设置数据保存目录
        if data_dir is None:
            data_dir = str(COLLECTED_DATA_DIR)
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        
        # 初始化数据结构
        self.data = {
            "episodes": [],
            "metadata": {
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        
        # 当前回合数据
        self.current_episode = None
    
    def start_new_episode(self, map_index: int, map_data: Dict[str, Any]):
        """
        开始记录新的回合数据
        
        Args:
            map_index: 地图索引
            map_data: 地图数据
        """
        # 结束上一个回合（如果有）
        if self.current_episode is not None:
            self.end_episode()
        
        # 创建新的回合数据
        self.current_episode = {
            "map_index": map_index,
            "map_data": {
                "obstacles": map_data.get("obstacles", []),
                "coins": map_data.get("coins", []),
                "monsters": map_data.get("monsters", []),
                "mode": map_data.get("mode", "unknown")
            },
            "steps": [],
            "start_time": time.time(),
            "end_time": None,
            "total_steps": 0,
            "total_reward": 0,
            "success": False,
            "lives_remaining": 0
        }
    
    def record_step(self, state_before: Dict[str, Any], action: int, reasoning: str, 
                    reward: float, state_after: Dict[str, Any], done: bool):
        """
        记录一步游戏数据
        
        Args:
            state_before: 动作前的状态
            action: 执行的动作
            reasoning: 动作的推理过程（如果是代理执行的动作）
            reward: 获得的奖励
            state_after: 动作后的状态
            done: 是否结束
        """
        if self.current_episode is None:
            raise ValueError("必须先调用start_new_episode开始新回合")
        
        # 收集步骤数据
        step_data = {
            "action": action,
            "reward": reward,
            "done": done,
            "positions": {
                "before": state_before.get("agent_pos"),
                "after": state_after.get("agent_pos")
            },
            "score": {
                "before": state_before.get("score", 0),
                "after": state_after.get("score", 0)
            },
            "lives": {
                "before": state_before.get("lives", 0),
                "after": state_after.get("lives", 0)
            },
            "timestamp": time.time()
        }
        
        # 如果有推理过程，添加到数据中
        if reasoning:
            step_data["reasoning"] = reasoning
        
        # 添加到当前回合的步骤列表
        self.current_episode["steps"].append(step_data)
        
        # 更新回合总计数据
        self.current_episode["total_steps"] += 1
        self.current_episode["total_reward"] += reward
        
        # 如果游戏结束，记录最终状态
        if done:
            self.current_episode["success"] = reward > 0
            self.current_episode["lives_remaining"] = state_after.get("lives", 0)
            self.end_episode()
    
    def end_episode(self):
        """结束当前回合，将数据添加到总数据中"""
        if self.current_episode is None:
            return
        
        # 记录结束时间
        self.current_episode["end_time"] = time.time()
        self.current_episode["duration"] = self.current_episode["end_time"] - self.current_episode["start_time"]
        
        # 添加到总数据中
        self.data["episodes"].append(self.current_episode)
        
        # 重置当前回合
        self.current_episode = None
    
    def save_data(self) -> str:
        """
        保存收集的数据到文件
        
        Returns:
            保存的文件路径
        """
        # 确保当前回合已结束
        if self.current_episode is not None:
            self.end_episode()
        
        # 更新元数据
        self.data["metadata"]["episodes_count"] = len(self.data["episodes"])
        self.data["metadata"]["end_timestamp"] = time.time()
        self.data["metadata"]["end_datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data["metadata"]["duration"] = self.data["metadata"]["end_timestamp"] - self.data["metadata"]["timestamp"]
        
        # 创建文件名，使用时间戳
        filename = f"game_data_{int(time.time())}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # 保存到文件
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        return filepath
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取收集数据的摘要统计信息
        
        Returns:
            统计信息字典
        """
        if not self.data["episodes"]:
            return {"status": "无数据"}
        
        # 计算统计信息
        episodes_count = len(self.data["episodes"])
        total_steps = sum(ep["total_steps"] for ep in self.data["episodes"])
        total_rewards = sum(ep["total_reward"] for ep in self.data["episodes"])
        success_count = sum(1 for ep in self.data["episodes"] if ep.get("success", False))
        success_rate = success_count / episodes_count if episodes_count > 0 else 0
        
        avg_steps = total_steps / episodes_count if episodes_count > 0 else 0
        avg_reward = total_rewards / episodes_count if episodes_count > 0 else 0
        
        # 收集模式信息
        modes = {}
        for ep in self.data["episodes"]:
            mode = ep["map_data"].get("mode", "unknown")
            modes[mode] = modes.get(mode, 0) + 1
        
        return {
            "总回合数": episodes_count,
            "总步数": total_steps,
            "总分数": total_rewards,
            "成功回合数": success_count,
            "成功率": f"{success_rate:.2%}",
            "平均步数/回合": f"{avg_steps:.2f}",
            "平均分数/回合": f"{avg_reward:.2f}",
            "游戏模式分布": modes
        } 
