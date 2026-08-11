import numpy as np
from typing import Dict, Tuple, List, Set

from src.config.game_config import *

class MapProcessor:
    """处理游戏地图，生成适合AI代理使用的表示形式"""
    
    @staticmethod
    def get_map_representation(grid, vision_map, agent_pos):
        """获取地图的字符串表示，使用与游戏系统一致的坐标系统"""
        grid = np.array(grid)  # 确保是numpy数组
        vision_map = np.array(vision_map)  # 确保是numpy数组
        rows, cols = grid.shape
        
        # 创建带坐标的地图表示
        map_repr = []
        
        # 添加坐标系说明
        map_repr.append("Game Coordinate System (row,col):")
        map_repr.append("- Rows: 0 (top) to N (bottom)")
        map_repr.append("- Cols: 0 (left) to N (right)")
        map_repr.append("")
        
        # 添加列坐标标题
        col_header = "   " + " ".join([str(j).rjust(2) for j in range(cols)])  # 使用右对齐确保对齐
        map_repr.append(col_header)
        
        # 添加分隔线
        map_repr.append("  " + "-" * (cols * 3 + 1))
        
        # 从上到下遍历地图（与游戏系统坐标一致）
        for i in range(rows):
            row = [f"{str(i).rjust(2)} |"]  # 添加行坐标，右对齐
            for j in range(cols):
                pos = (i, j)
                # 使用数组索引而不是元组
                if vision_map[i, j] == NOT_DISCOVERED:
                    row.append(' ?')  # 未探索区域
                else:
                    # 如果是代理位置
                    if pos == tuple(agent_pos):
                        row.append(' A')  # 代理
                    else:
                        # 根据格子类型添加对应的符号
                        cell_type = grid[i, j]
                        if cell_type == EMPTY:
                            row.append(' .')  # 空地
                        elif cell_type == OBSTACLE:
                            row.append(' #')  # 障碍物
                        elif cell_type == COIN:
                            row.append(' C')  # 金币
                        elif cell_type == MONSTER:
                            row.append(' M')  # 怪物
                        elif cell_type == GOAL:
                            row.append(' G')  # 目标
                        elif cell_type == SHOVEL:
                            row.append(' S')  # 铲子
                        elif cell_type == SWORD:
                            row.append(' W')  # 剑
                        elif cell_type == MAGNET:
                            row.append(' T')  # 磁铁
                        elif cell_type == KEY:
                            row.append(' K')  # 钥匙
                        else:
                            row.append(' ?')  # 未知类型
            map_repr.append("".join(row))
        
        # 添加当前位置信息
        map_repr.append(f"\nCurrent position (row,col): ({agent_pos[0]},{agent_pos[1]})")
        
        return "\n".join(map_repr)
    
    @staticmethod
    def format_map_for_agent(grid, vision_map, agent_pos, coins=None, lives=None, 
                            score=None, mode=None, monsters=None, obstacles=None, **kwargs):
        """格式化地图信息为代理可理解的格式"""
        # 确保输入数据是numpy数组
        grid = np.array(grid)
        vision_map = np.array(vision_map)
        agent_pos = np.array(agent_pos)
        
        # 获取地图的字符串表示
        map_repr = MapProcessor.get_map_representation(grid, vision_map, agent_pos)
        
        # 构建基本信息
        info = {
            "map": map_repr,
            "agent_position": [int(agent_pos[0]), int(agent_pos[1])],
            "mode": mode if mode else "Unknown",
            "grid_size": grid.shape[0],
            "lives": lives if lives is not None else 0,
            "score": score if score is not None else 0
        }
        
        # 添加可见的游戏对象信息
        visible_objects = {
            "coins": [],
            "monsters": [],
            "obstacles": []
        }
        
        # 处理金币
        if coins is not None and isinstance(coins, (set, list, tuple)):
            for coin in coins:
                x, y = coin if isinstance(coin, (tuple, list)) else (coin[0], coin[1])
                if vision_map[int(x), int(y)] == DISCOVERED:
                    visible_objects["coins"].append([int(x), int(y)])
        
        # 处理怪物
        if monsters is not None and isinstance(monsters, (list, tuple)):
            for monster in monsters:
                x, y = monster if isinstance(monster, (tuple, list)) else (monster[0], monster[1])
                if vision_map[int(x), int(y)] == DISCOVERED:
                    visible_objects["monsters"].append([int(x), int(y)])
        
        # 处理障碍物
        if obstacles is not None and isinstance(obstacles, (set, list, tuple)):
            for obstacle in obstacles:
                x, y = obstacle if isinstance(obstacle, (tuple, list)) else (obstacle[0], obstacle[1])
                if vision_map[int(x), int(y)] == DISCOVERED:
                    visible_objects["obstacles"].append([int(x), int(y)])
        
        # 添加可见对象信息
        info.update(visible_objects)
        
        # 添加其他可能的游戏状态信息
        for key, value in kwargs.items():
            if isinstance(value, (bool, int, float, str, list)):
                info[key] = value
        
        return info
    
    @staticmethod
    def get_actions_description() -> str:
        """获取动作描述信息"""
        actions = [
            "0: Move UP 1 step",
            "1: Move UP 2 steps",
            "2: Move UP 3 steps",
            "3: Move DOWN 1 step",
            "4: Move DOWN 2 steps",
            "5: Move DOWN 3 steps",
            "6: Move LEFT 1 step",
            "7: Move LEFT 2 steps",
            "8: Move LEFT 3 steps",
            "9: Move RIGHT 1 step",
            "10: Move RIGHT 2 steps",
            "11: Move RIGHT 3 steps"
        ]
        
        return "Available actions:\n" + "\n".join(actions) 