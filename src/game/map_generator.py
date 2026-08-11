import numpy as np
import json
import os
import random
from typing import List, Dict, Tuple, Set
from collections import deque
from src.config.game_config import *
from src.game.obstacles import Obstacle

def _apply_grid_config(mode: str):
    """Sync dynamic grid config into this module's imported constants."""
    global GRID_SIZE, START_POS, GOAL_POS, WINDOW_SIZE
    GRID_SIZE, START_POS, GOAL_POS, WINDOW_SIZE = update_grid_config(mode)

class MapGenerator:
    @staticmethod
    def _has_valid_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """使用BFS检查是否存在从起点到终点的有效路径"""
        if grid[start] == OBSTACLE or grid[goal] == OBSTACLE:
            return False
            
        # 初始化访问数组
        visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        visited[start] = True
        
        # 初始化队列
        queue = deque([start])
        
        # BFS搜索
        while queue:
            current = queue.popleft()
            
            # 如果到达终点，返回True
            if current == goal:
                return True
            
            # 检查所有可能的移动方向
            for dx, dy in DIRECTIONS.values():
                for steps in range(1, MAX_STEPS + 1):  # 考虑1-3步的移动
                    new_x = current[0] + dx * steps
                    new_y = current[1] + dy * steps
                    new_pos = (new_x, new_y)
                    
                    # 检查是否在网格内且未访问过
                    if (0 <= new_x < GRID_SIZE and 
                        0 <= new_y < GRID_SIZE and 
                        not visited[new_x, new_y]):
                        
                        # 检查路径上是否有障碍物
                        path_blocked = False
                        for s in range(1, steps + 1):
                            check_x = current[0] + dx * s
                            check_y = current[1] + dy * s
                            if grid[check_x, check_y] == OBSTACLE:
                                path_blocked = True
                                break
                        
                        if not path_blocked:
                            visited[new_x, new_y] = True
                            queue.append(new_pos)
        
        return False

    @staticmethod
    def generate_map(mode: str) -> Dict:
        """生成单个地图，确保存在有效路径"""
        # 更新网格配置
        _apply_grid_config(mode)
        
        while True:
            grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
            
            # 设置起点和终点
            grid[START_POS] = START
            grid[GOAL_POS] = GOAL
            
            # 生成障碍物
            obstacles = Obstacle.get_random_obstacle_layout(GRID_SIZE)
            
            # 验证地图是否有效
            temp_grid = grid.copy()
            for obs_pos in obstacles:
                temp_grid[obs_pos] = OBSTACLE
            
            # 检查是否存在有效路径
            if MapGenerator._has_valid_path(temp_grid, START_POS, GOAL_POS):
                # 如果存在有效路径，使用这个地图
                grid = temp_grid
                break
        
        # 生成可用位置列表（不包括起点、终点和障碍物）
        available_positions = [
            (i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
            if grid[i, j] == EMPTY
        ]
        
        # 生成金币
        coins = set()
        if len(available_positions) >= COINS_COUNT:
            coin_indices = np.random.choice(
                len(available_positions), 
                COINS_COUNT, 
                replace=False
            )
            for idx in coin_indices:
                pos = available_positions[idx]
                coins.add(pos)
                grid[pos] = COIN
            
            # 更新可用位置
            available_positions = [pos for pos in available_positions if pos not in coins]
        
        # 为Level 2和Level 3生成怪物
        monsters = []
        if mode in [MODE_LEVEL2, MODE_LEVEL3]:
            num_monsters = random.randint(MIN_MONSTERS, MAX_MONSTERS)
            if len(available_positions) >= num_monsters:
                monster_indices = np.random.choice(
                    len(available_positions), 
                    num_monsters, 
                    replace=False
                )
                for idx in monster_indices:
                    pos = available_positions[idx]
                    monsters.append(pos)
                    grid[pos] = MONSTER
                
                # 更新可用位置
                available_positions = [pos for pos in available_positions if grid[pos] == EMPTY]
        
        # 为Level 3生成道具
        items = {"shovel": None, "sword": None, "magnet": None, "key": None}
        if mode == MODE_LEVEL3 and len(available_positions) >= 4:
            # 生成钥匙（必须有）
            key_idx = np.random.choice(len(available_positions))
            key_pos = available_positions[key_idx]
            grid[key_pos] = KEY
            items["key"] = key_pos
            available_positions.remove(key_pos)
            
            # 生成铲子
            if available_positions:
                shovel_idx = np.random.choice(len(available_positions))
                shovel_pos = available_positions[shovel_idx]
                grid[shovel_pos] = SHOVEL
                items["shovel"] = shovel_pos
                available_positions.remove(shovel_pos)
            
            # 生成剑
            if available_positions:
                sword_idx = np.random.choice(len(available_positions))
                sword_pos = available_positions[sword_idx]
                grid[sword_pos] = SWORD
                items["sword"] = sword_pos
                available_positions.remove(sword_pos)
            
            # 生成磁铁
            if available_positions:
                magnet_idx = np.random.choice(len(available_positions))
                magnet_pos = available_positions[magnet_idx]
                grid[magnet_pos] = MAGNET
                items["magnet"] = magnet_pos
                available_positions.remove(magnet_pos)
        
        # 转换为可序列化的格式
        return {
            'grid': grid.tolist(),
            'obstacles': list(obstacles),
            'coins': list(coins),
            'monsters': monsters,
            'items': {k: list(v) if v else None for k, v in items.items()},
            'mode': mode,
            'grid_size': GRID_SIZE
        }
    
    @staticmethod
    def generate_maps(count: int, mode: str, save_dir: str):
        """生成多个地图并保存"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成地图
        maps = []
        for i in range(count):
            print(f"正在生成第 {i+1}/{count} 个地图...")
            map_data = MapGenerator.generate_map(mode)
            maps.append(map_data)
            
            # 保存单个地图
            filename = f"{mode.replace(' ', '_')}_{i+1}.json"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(map_data, f, indent=2)
        
        # 保存地图集合
        collection_filename = f"{mode.replace(' ', '_')}_collection.json"
        collection_filepath = os.path.join(save_dir, collection_filename)
        with open(collection_filepath, 'w') as f:
            json.dump(maps, f, indent=2)
        
        return maps
    
    @staticmethod
    def load_map(filepath: str) -> Dict:
        """加载单个地图"""
        with open(filepath, 'r') as f:
            map_data = json.load(f)
        return map_data
    
    @staticmethod
    def load_map_collection(filepath: str) -> List[Dict]:
        """加载地图集合"""
        with open(filepath, 'r') as f:
            maps = json.load(f)
        return maps 
