"""
Map generator for maze navigation game.
Generates maze layouts with obstacles, items and enemies.
"""

import numpy as np
import json
import os
import random
from typing import List, Dict, Tuple, Set, Any
from collections import deque
import time

# 使用相对导入
from ..game.config import *
from ..game.obstacles import Obstacle

class MapGenerator:
    """Map generator for maze navigation game"""
    
    @staticmethod
    def _has_valid_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        Check if a valid path exists from start to goal using BFS
        
        Args:
            grid: Grid representation
            start: Starting position
            goal: Goal position
            
        Returns:
            True if a valid path exists, False otherwise
        """
        if grid[start] == OBSTACLE or grid[goal] == OBSTACLE:
            return False
            
        # Initialize visited array
        visited = np.zeros((grid.shape[0], grid.shape[1]), dtype=bool)
        visited[start] = True
        
        # Initialize queue
        queue = deque([start])
        
        # BFS search
        while queue:
            current = queue.popleft()
            
            # If goal reached, return True
            if current == goal:
                return True
            
            # Check all possible move directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            for dx, dy in directions:
                for steps in range(1, 4):  # Consider 1-3 step moves
                    new_x = current[0] + dx * steps
                    new_y = current[1] + dy * steps
                    new_pos = (new_x, new_y)
                    
                    # Check if in bounds and not visited
                    if (0 <= new_x < grid.shape[0] and 
                        0 <= new_y < grid.shape[1] and 
                        not visited[new_x, new_y]):
                        
                        # Check if path is blocked by obstacles
                        path_blocked = False
                        for s in range(1, steps + 1):
                            check_x = current[0] + dx * s
                            check_y = current[1] + dy * s
                            if not (0 <= check_x < grid.shape[0] and 0 <= check_y < grid.shape[1]) or \
                               grid[check_x, check_y] == OBSTACLE:
                                path_blocked = True
                                break
                        
                        if not path_blocked:
                            visited[new_x, new_y] = True
                            queue.append(new_pos)
        
        return False

    @staticmethod
    def generate_map(mode: str = MODE_LEVEL2) -> Dict[str, Any]:
        """
        Generate a single map with valid path
        
        Args:
            mode: Game mode/difficulty
            
        Returns:
            Dictionary with map data
        """
        grid_size = GRID_SIZE
        start_pos = START_POS
        goal_pos = GOAL_POS
        
        # Try generating a valid map
        max_attempts = 10
        for attempt in range(max_attempts):
            # Initialize grid
            grid = np.zeros((grid_size, grid_size), dtype=np.int32)
            
            # Set start and goal
            grid[start_pos] = EMPTY  # Start will be replaced with agent
            grid[goal_pos] = GOAL
            
            # Generate obstacles
            obstacles = Obstacle.get_random_obstacle_layout(
                grid_size, 
                [start_pos, goal_pos], 
                MAX_OBSTACLES_PERCENT
            )
            
            # Add obstacles to grid
            for obs_pos in obstacles:
                grid[obs_pos] = OBSTACLE
            
            # Check if a valid path exists
            if MapGenerator._has_valid_path(grid, start_pos, goal_pos):
                # Map is valid, add coins and monsters
                available_positions = [
                    (i, j) for i in range(grid_size) for j in range(grid_size)
                    if grid[i, j] == EMPTY and (i, j) != start_pos
                ]
                
                if len(available_positions) < COINS_COUNT:
                    continue  # Not enough space for coins
                
                # Add coins
                coins = set()
                if available_positions:
                    coin_indices = random.sample(range(len(available_positions)), 
                                               min(COINS_COUNT, len(available_positions)))
                    for idx in coin_indices:
                        pos = available_positions[idx]
                        coins.add(pos)
                        grid[pos] = COIN
                
                # Update available positions
                available_positions = [pos for pos in available_positions if pos not in coins]
                
                # Add monsters for Level 2 and 3
                monsters = []
                if mode in [MODE_LEVEL2, MODE_LEVEL3] and MONSTERS_COUNT[mode] > 0:
                    if len(available_positions) < MONSTERS_COUNT[mode]:
                        continue  # Not enough space for monsters
                    
                    monster_indices = random.sample(range(len(available_positions)), 
                                                  min(MONSTERS_COUNT[mode], len(available_positions)))
                    for idx in monster_indices:
                        pos = available_positions[idx]
                        monsters.append(pos)
                        grid[pos] = MONSTER
                
                # Update available positions
                available_positions = [pos for pos in available_positions 
                                      if grid[pos[0], pos[1]] == EMPTY]
                
                # Add items for Level 3
                items = {"shovel": None, "sword": None, "magnet": None, "key": None}
                if mode == MODE_LEVEL3:
                    # Need positions for at least key and one other item
                    if len(available_positions) < 2:
                        continue
                    
                    # Add key (required)
                    key_idx = random.randrange(len(available_positions))
                    key_pos = available_positions[key_idx]
                    grid[key_pos] = KEY
                    items["key"] = key_pos
                    available_positions.pop(key_idx)
                    
                    # Add other items if space available
                    item_types = [("shovel", SHOVEL), ("sword", SWORD), ("magnet", MAGNET)]
                    random.shuffle(item_types)
                    
                    for item_name, item_code in item_types:
                        if available_positions:
                            item_idx = random.randrange(len(available_positions))
                            item_pos = available_positions[item_idx]
                            grid[item_pos] = item_code
                            items[item_name] = item_pos
                            available_positions.pop(item_idx)
                
                # Map generation successful
                return {
                    'grid': grid.tolist(),
                    'obstacles': list(obstacles),
                    'coins': list(coins),
                    'monsters': monsters,
                    'items': items,
                    'mode': mode,
                    'grid_size': grid_size,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos
                }
        
        # If all attempts failed, return a simple map
        print("Failed to generate a valid map, creating simple map")
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        grid[start_pos] = EMPTY
        grid[goal_pos] = GOAL
        
        return {
            'grid': grid.tolist(),
            'obstacles': [],
            'coins': [],
            'monsters': [],
            'items': {"shovel": None, "sword": None, "magnet": None, "key": None},
            'mode': mode,
            'grid_size': grid_size,
            'start_pos': start_pos,
            'goal_pos': goal_pos
        }
    
    @staticmethod
    def generate_maps(count: int, mode: str, save_dir: str) -> List[Dict[str, Any]]:
        """
        Generate multiple maps and save them
        
        Args:
            count: Number of maps to generate
            mode: Game mode/difficulty
            save_dir: Directory to save maps
            
        Returns:
            List of generated maps
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate maps
        maps = []
        for i in range(count):
            print(f"Generating map {i+1}/{count}...")
            map_data = MapGenerator.generate_map(mode)
            maps.append(map_data)
            
            # Save individual map
            filename = f"{mode.replace(' ', '_')}_{i+1}.json"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(map_data, f, indent=2)
        
        # Save map collection
        collection_filename = f"{mode.replace(' ', '_')}_collection.json"
        collection_filepath = os.path.join(save_dir, collection_filename)
        with open(collection_filepath, 'w') as f:
            json.dump(maps, f, indent=2)
        
        print(f"Generated and saved {count} maps for {mode}")
        return maps
    
    @staticmethod
    def load_map(filepath: str) -> Dict[str, Any]:
        """
        Load a map from file
        
        Args:
            filepath: Path to map file
            
        Returns:
            Map data
        """
        with open(filepath, 'r') as f:
            map_data = json.load(f)
        return map_data
    
    @staticmethod
    def load_map_collection(filepath: str) -> List[Dict[str, Any]]:
        """
        Load a collection of maps from file
        
        Args:
            filepath: Path to map collection file
            
        Returns:
            List of maps
        """
        with open(filepath, 'r') as f:
            maps = json.load(f)
        return maps 