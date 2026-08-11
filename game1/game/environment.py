"""
Maze navigation game environment.
Implements a gym-compatible environment for maze navigation game.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
from typing import Tuple, Optional, Dict, Any, List, Set
import random
import time
import os
import json

from ..game.config import *
from ..game.obstacles import Obstacle

class Monster:
    """
    Monster that moves randomly in the maze
    """
    def __init__(self, pos: Tuple[int, int]):
        """
        Initialize monster at given position
        
        Args:
            pos: Initial position (x, y)
        """
        self.pos = pos
        self.next_pos = pos
        self.is_moving = False
    
    def plan_move(self, grid: np.ndarray) -> bool:
        """
        Plan next move, returns whether a valid move is found
        
        Args:
            grid: Current grid state
            
        Returns:
            True if a valid move is found, False otherwise
        """
        possible_moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for dx, dy in directions:
            new_x = self.pos[0] + dx
            new_y = self.pos[1] + dy
            if (0 <= new_x < grid.shape[0] and 
                0 <= new_y < grid.shape[1] and 
                grid[new_x, new_y] != OBSTACLE and
                grid[new_x, new_y] != MONSTER):
                possible_moves.append((new_x, new_y))
        
        if possible_moves:
            self.next_pos = random.choice(possible_moves)
            self.is_moving = True
            return True
        return False
    
    def complete_move(self):
        """Complete the planned move"""
        self.pos = self.next_pos
        self.is_moving = False


class MazeNavigationEnv(gym.Env):
    """
    Maze navigation environment for reinforcement learning and LLM agents
    """
    def __init__(self, mode=MODE_LEVEL2, full_vision=False):
        """
        Initialize environment
        
        Args:
            mode: Game mode/difficulty
            full_vision: Whether agent has full visibility of the maze
        """
        super().__init__()
        self.mode = mode
        self.map_index = 0
        self.full_vision = full_vision
        
        # Action space: 4 directions x 3 step sizes
        self.action_space = spaces.Discrete(12)
        
        # Observation space: grid representation
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int32
        )
        
        # Initialize game state
        self.grid = None
        self.agent_pos = None
        self.obstacles = None
        self.lives = LIVES_DEFAULT
        self.score = 0
        self.coins = set()
        self.monsters = []
        self.vision_map = None
        self.steps_count = 0
        
        # Action and reward history
        self.action_history = []
        self.reward_history = []
        
        # Exploration tracking
        self.discovered_cells = 0
        self.total_cells = GRID_SIZE * GRID_SIZE
        
        # Item states
        self.has_shovel = False
        self.shovel_uses = 0
        self.has_sword = False
        self.has_magnet = False
        self.has_key = False
        
        # Item positions
        self.items = {
            "shovel": None, 
            "sword": None, 
            "magnet": None, 
            "key": None
        }
        
        # Initialize pygame for visualization
        pygame.init()
        self.screen = pygame.display.set_mode((GRID_SIZE * 50, GRID_SIZE * 50))
        pygame.display.set_caption(f"Maze Navigation - {mode}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.shadow_surface = pygame.Surface((GRID_SIZE * 50, GRID_SIZE * 50), pygame.SRCALPHA)
    
    def _action_to_direction(self, action: int) -> Tuple[Tuple[int, int], int]:
        """
        Convert action index to direction and steps
        
        Args:
            action: Action index (0-11)
            
        Returns:
            (direction, steps): Direction as (dx, dy) and number of steps
        """
        direction_idx = action // 3
        steps = (action % 3) + 1
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        return directions[direction_idx], steps
    
    def _get_new_position(self, action: int) -> Tuple[int, int]:
        """
        Calculate new position based on action
        
        Args:
            action: Action index (0-11)
            
        Returns:
            (x, y): New position
        """
        direction, steps = self._action_to_direction(action)
        
        # Calculate new position
        new_x = self.agent_pos[0] + direction[0] * steps
        new_y = self.agent_pos[1] + direction[1] * steps
        
        # Ensure position is within grid bounds
        new_x = max(0, min(new_x, GRID_SIZE - 1))
        new_y = max(0, min(new_y, GRID_SIZE - 1))
        
        return (new_x, new_y)
    
    def move_agent(self, new_pos: Tuple[int, int]) -> float:
        """
        Move agent to new position and return reward
        
        Args:
            new_pos: New position (x, y)
            
        Returns:
            reward: Reward for the move
        """
        # Save old position
        old_pos = self.agent_pos
        
        # Check if hitting obstacle
        x, y = new_pos
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and self.grid[x, y] == OBSTACLE:
            # If has shovel, can break obstacle
            if self.has_shovel and self.shovel_uses > 0:
                # Use shovel
                self.shovel_uses -= 1
                
                # Remove obstacle
                self.obstacles.remove((x, y))
                self.grid[x, y] = EMPTY
                
                # Update agent position
                self.agent_pos = new_pos
                
                # If shovel is used up, remove it
                if self.shovel_uses <= 0:
                    self.has_shovel = False
                
                # Return reward for using shovel
                return REWARD_STEP  # Step penalty only
            else:
                # Lose a life
                self.lives -= 1
                
                # If no lives left, return large penalty
                if self.lives <= 0:
                    return REWARD_LIFE_LOST
                
                # Return to start
                self.agent_pos = START_POS
                
                # Return penalty
                return REWARD_LIFE_LOST
        
        # Update agent position
        self.agent_pos = new_pos
        
        # Check if hitting monster
        if self.mode != MODE_LEVEL1 and self._check_monster_collision():
            # If has sword, defeat monster
            if self.has_sword:
                # Find monster at agent position
                for monster in self.monsters[:]:
                    if monster.pos == self.agent_pos:
                        self.monsters.remove(monster)
                        # Update grid
                        self.grid[self.agent_pos] = EMPTY
                        return REWARD_STEP  # Step penalty only
            else:
                # Lose a life
                self.lives -= 1
                
                # If no lives left, return large penalty
                if self.lives <= 0:
                    return REWARD_LIFE_LOST
                
                # Return to start
                self.agent_pos = START_POS
                
                # Return penalty
                return REWARD_LIFE_LOST
        
        # Check if collected coin
        reward = REWARD_STEP  # Default step penalty
        if tuple(self.agent_pos) in self.coins:
            self.coins.remove(tuple(self.agent_pos))
            
            # Update grid
            self.grid[self.agent_pos] = EMPTY
            
            # Add coin reward
            reward += REWARD_COIN
        
        # If has magnet, collect nearby coins
        if self.has_magnet:
            magnet_reward = self._collect_nearby_coins()
            reward += magnet_reward
        
        # Check item collection
        if self.mode == MODE_LEVEL3:
            item_reward = self._check_item_collection()
            reward += item_reward
        
        # Check if reached goal
        if self.agent_pos == GOAL_POS:
            # In Level 3, need key to unlock goal
            if self.mode == MODE_LEVEL3 and not self.has_key:
                # Reset position and return penalty
                self.agent_pos = old_pos
                return REWARD_STEP - 200  # Step penalty + additional penalty
            else:
                # Successfully reached goal
                return reward + REWARD_GOAL
        
        # Return total reward
        return reward
    
    def _is_valid_move(self, new_pos: Tuple[int, int]) -> bool:
        """
        Check if move is valid
        
        Args:
            new_pos: New position (x, y)
            
        Returns:
            True if move is valid, False otherwise
        """
        x, y = new_pos
        # Check if in grid
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        return True
    
    def _generate_coins(self):
        """Generate coins at random positions"""
        self.coins = set()
        available_positions = [
            (i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
            if (i, j) not in self.obstacles and 
               (i, j) != START_POS and 
               (i, j) != GOAL_POS and 
               (i, j) != self.agent_pos
        ]
        
        if len(available_positions) >= COINS_COUNT:
            coin_positions = np.random.choice(
                len(available_positions), 
                COINS_COUNT, 
                replace=False
            )
            self.coins = {available_positions[i] for i in coin_positions}
            for coin_pos in self.coins:
                self.grid[coin_pos] = COIN
    
    def _generate_monsters(self):
        """Generate monsters at random positions"""
        if self.mode == MODE_LEVEL1:
            return
        
        self.monsters = []
        available_positions = [
            (i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
            if (i, j) not in self.obstacles and 
               (i, j) != START_POS and 
               (i, j) != GOAL_POS and 
               (i, j) != self.agent_pos and
               (i, j) not in self.coins
        ]
        
        monster_count = MONSTERS_COUNT[self.mode]
        if len(available_positions) >= monster_count:
            monster_indices = np.random.choice(
                len(available_positions),
                monster_count,
                replace=False
            )
            
            for idx in monster_indices:
                pos = available_positions[idx]
                self.monsters.append(Monster(pos))
                self.grid[pos] = MONSTER
    
    def _move_monsters(self):
        """Move monsters randomly with some probability"""
        for monster in self.monsters:
            if not monster.is_moving and random.random() < MONSTER_MOVE_PROB:
                # Plan move
                if monster.plan_move(self.grid):
                    # Update grid
                    self.grid[monster.pos] = EMPTY
                    self.grid[monster.next_pos] = MONSTER
                    monster.complete_move()
    
    def _check_monster_collision(self) -> bool:
        """
        Check if agent collided with monster
        
        Returns:
            True if collision detected, False otherwise
        """
        # Check if agent is on monster position
        for monster in self.monsters:
            if monster.pos == self.agent_pos:
                return True
        return False
    
    def _collect_nearby_coins(self) -> float:
        """
        Collect coins in adjacent cells if agent has magnet
        
        Returns:
            Total reward from collecting coins
        """
        if not self.has_magnet:
            return 0
        
        total_reward = 0
        x, y = self.agent_pos
        
        # Check adjacent cells (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                (nx, ny) in self.coins):
                # Collect coin
                self.coins.remove((nx, ny))
                self.grid[nx, ny] = EMPTY
                total_reward += REWARD_COIN
        
        return total_reward
    
    def _update_vision(self):
        """Update agent's vision map based on current position"""
        if self.full_vision:
            # Full visibility
            self.vision_map = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int32)
            self.discovered_cells = self.total_cells
            return
        
        # Create vision map if not exists
        if self.vision_map is None:
            self.vision_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        
        # Discover cells in visibility range
        x, y = self.agent_pos
        vision_range = 2  # Cells visible in each direction
        
        newly_discovered = 0
        for i in range(max(0, x - vision_range), min(GRID_SIZE, x + vision_range + 1)):
            for j in range(max(0, y - vision_range), min(GRID_SIZE, y + vision_range + 1)):
                if self.vision_map[i, j] == NOT_DISCOVERED:
                    self.vision_map[i, j] = DISCOVERED
                    newly_discovered += 1
        
        # Update discovery count
        self.discovered_cells += newly_discovered
    
    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            (observation, info): Initial observation and info
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize grid
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        
        # Set start and goal positions
        self.agent_pos = START_POS
        self.grid[GOAL_POS] = GOAL
        
        # Generate obstacles
        self.obstacles = Obstacle.get_random_obstacle_layout(GRID_SIZE)
        for pos in self.obstacles:
            self.grid[pos] = OBSTACLE
        
        # Generate coins
        self._generate_coins()
        
        # Generate monsters
        self._generate_monsters()
        
        # Generate items for Level 3
        if self.mode == MODE_LEVEL3:
            self._generate_items()
        
        # Reset vision map
        self.vision_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.discovered_cells = 0
        
        # Initialize visibility
        self._update_vision()
        
        # Reset game state
        self.lives = LIVES_DEFAULT
        self.score = 0
        self.steps_count = 0
        self.action_history = []
        self.reward_history = []
        
        # Reset items
        self.has_shovel = False
        self.shovel_uses = 0
        self.has_sword = False
        self.has_magnet = False
        self.has_key = False
        
        # Return observation
        observation = self._get_state()
        info = self.get_state_dict()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Action to take (0-11)
            
        Returns:
            (observation, reward, terminated, truncated, info): Step results
        """
        # Calculate new position
        new_pos = self._get_new_position(action)
        
        # Get direction and steps for the action
        direction, steps = self._action_to_direction(action)
        
        # Log action
        action_data = {
            'action': action,
            'direction': direction,
            'steps': steps,
            'from': self.agent_pos,
            'to': new_pos
        }
        self.action_history.append(action_data)
        
        # Move agent and get reward
        reward = self.move_agent(new_pos)
        
        # Log reward
        self.reward_history.append(reward)
        self.score += reward
        
        # Move monsters
        self._move_monsters()
        
        # Update vision
        self._update_vision()
        
        # Increment step counter
        self.steps_count += 1
        
        # Check for termination conditions
        terminated = False
        
        # Check for goal reached
        if self.agent_pos == GOAL_POS:
            terminated = True
        
        # Check for life loss
        if self.lives <= 0:
            terminated = True
        
        # No truncation in this environment
        truncated = False
        
        # Get observation and info
        observation = self._get_state()
        info = self.get_state_dict()
        
        return observation, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation
        
        Returns:
            Grid representation of state
        """
        state = self.grid.copy()
        
        # Add agent position
        state[self.agent_pos] = 5  # Agent representation
        
        # Apply fog of war if not full vision
        if not self.full_vision and self.vision_map is not None:
            # Mask unexplored areas
            masked_state = state.copy()
            masked_state[self.vision_map == NOT_DISCOVERED] = -1
            return masked_state
        
        return state

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary
        
        Returns:
            Dictionary containing state information
        """
        return {
            'mode': self.mode,
            'map_index': self.map_index,
            'full_vision': self.full_vision,
            'grid': self.grid.tolist(),
            'agent_pos': self.agent_pos,
            'obstacles': self.obstacles,
            'lives': self.lives,
            'score': self.score,
            'coins': list(self.coins),
            'monsters': [monster.pos for monster in self.monsters],
            'vision_map': self.vision_map.tolist(),
            'steps_count': self.steps_count,
            'action_history': self.action_history,
            'reward_history': self.reward_history,
            'discovered_cells': self.discovered_cells,
            'total_cells': self.total_cells,
            'has_shovel': self.has_shovel,
            'shovel_uses': self.shovel_uses,
            'has_sword': self.has_sword,
            'has_magnet': self.has_magnet,
            'has_key': self.has_key,
            'items': self.items,
            'mode_name': MODE_NAMES[self.mode]
        }
    
    def render(self):
        """Render the environment using pygame"""
        # Constants for rendering
        CELL_SIZE = 50
        GRID_OFFSET = 0
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        YELLOW = (255, 255, 0)
        PURPLE = (128, 0, 128)
        ORANGE = (255, 165, 0)
        PINK = (255, 192, 203)
        BROWN = (165, 42, 42)
        
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw grid cells
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(
                    j * CELL_SIZE + GRID_OFFSET,
                    i * CELL_SIZE + GRID_OFFSET,
                    CELL_SIZE,
                    CELL_SIZE
                )
                
                # Default cell color
                cell_color = WHITE
                
                # Apply fog of war
                if not self.full_vision and self.vision_map is not None and self.vision_map[i, j] == NOT_DISCOVERED:
                    # Draw black cell for unexplored
                    pygame.draw.rect(self.screen, BLACK, rect)
                    # Draw grid lines
                    pygame.draw.rect(self.screen, GRAY, rect, 1)
                    continue
                
                # Color based on cell content
                if (i, j) == self.agent_pos:
                    cell_color = BLUE
                elif self.grid[i, j] == OBSTACLE:
                    cell_color = GRAY
                elif self.grid[i, j] == COIN:
                    cell_color = YELLOW
                elif self.grid[i, j] == MONSTER:
                    cell_color = RED
                elif (i, j) == GOAL_POS:
                    cell_color = GREEN
                elif self.grid[i, j] == SHOVEL:
                    cell_color = BROWN
                elif self.grid[i, j] == SWORD:
                    cell_color = ORANGE
                elif self.grid[i, j] == MAGNET:
                    cell_color = PURPLE
                elif self.grid[i, j] == KEY:
                    cell_color = PINK
                
                # Draw cell
                pygame.draw.rect(self.screen, cell_color, rect)
                # Draw grid lines
                pygame.draw.rect(self.screen, BLACK, rect, 1)
        
        # Draw status text
        status_text = f"Score: {self.score} | Lives: {self.lives} | Steps: {self.steps_count}"
        text_surface = self.font.render(status_text, True, BLACK)
        self.screen.blit(text_surface, (10, GRID_SIZE * CELL_SIZE + 10))
        
        # Draw item status
        item_text = "Items: "
        if self.has_shovel:
            item_text += f"Shovel({self.shovel_uses}) "
        if self.has_sword:
            item_text += "Sword "
        if self.has_magnet:
            item_text += "Magnet "
        if self.has_key:
            item_text += "Key "
        
        item_surface = self.font.render(item_text, True, BLACK)
        self.screen.blit(item_surface, (10, GRID_SIZE * CELL_SIZE + 40))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
    
    def close(self):
        """Close environment and pygame"""
        pygame.quit()
    
    def load_map(self, map_data: Dict[str, Any], map_index: int = 0):
        """
        Load map from data
        
        Args:
            map_data: Map data dictionary
            map_index: Index for identifying map
        """
        self.map_index = map_index
        
        # Load grid
        self.grid = np.array(map_data['grid'], dtype=np.int32)
        
        # Load obstacles
        self.obstacles = set([tuple(pos) for pos in map_data['obstacles']])
        
        # Load coins
        self.coins = set([tuple(pos) for pos in map_data['coins']])
        
        # Load monsters
        self.monsters = [Monster(tuple(pos)) for pos in map_data['monsters']]
        
        # Load items
        self.items = {
            k: tuple(v) if v else None for k, v in map_data['items'].items()
        }
        
        # Reset agent position
        self.agent_pos = START_POS
        
        # Reset vision map
        self.vision_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.discovered_cells = 0
        
        # Update visibility
        self._update_vision()
        
        # Reset items
        self.has_shovel = False
        self.shovel_uses = 0
        self.has_sword = False
        self.has_magnet = False
        self.has_key = False
        
        # Reset game state
        self.lives = LIVES_DEFAULT
        self.score = 0
        self.steps_count = 0
        self.action_history = []
        self.reward_history = []
    
    def get_action_meaning(self, action: int) -> str:
        """
        Get human-readable meaning of action
        
        Args:
            action: Action index
            
        Returns:
            Human-readable action description
        """
        direction, steps = self._action_to_direction(action)
        direction_name = {
            (-1, 0): "UP",
            (1, 0): "DOWN",
            (0, -1): "LEFT",
            (0, 1): "RIGHT"
        }[direction]
        
        return f"{direction_name} {steps} step(s)"
    
    def end_session(self, done: bool = False, success: bool = False) -> Dict[str, Any]:
        """
        End current session and return statistics
        
        Args:
            done: Whether session is completed
            success: Whether session was successful
            
        Returns:
            Dictionary with session statistics
        """
        # Calculate exploration rate
        exploration_rate = self.discovered_cells / self.total_cells
        
        # Calculate steps to completion
        steps_to_completion = self.steps_count
        
        # Calculate average reward
        avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0
        
        # Calculate coins collected
        initial_coins = COINS_COUNT
        coins_collected = initial_coins - len(self.coins)
        coin_collection_rate = coins_collected / initial_coins if initial_coins > 0 else 0
        
        # Calculate items collected
        items_available = sum(1 for v in self.items.values() if v is not None)
        items_collected = sum([self.has_shovel, self.has_sword, self.has_magnet, self.has_key])
        item_collection_rate = items_collected / items_available if items_available > 0 else 0
        
        # Session statistics
        stats = {
            'mode': self.mode,
            'map_index': self.map_index,
            'done': done,
            'success': success,
            'score': self.score,
            'steps': self.steps_count,
            'lives_remaining': self.lives,
            'exploration_rate': exploration_rate,
            'coins_collected': coins_collected,
            'coin_collection_rate': coin_collection_rate,
            'items_collected': items_collected,
            'item_collection_rate': item_collection_rate,
            'avg_reward': avg_reward
        }
        
        return stats
    
    def _check_item_collection(self) -> float:
        """
        Check if agent collected an item
        
        Returns:
            Reward for collecting item
        """
        if tuple(self.agent_pos) == self.items.get('shovel'):
            self.has_shovel = True
            self.shovel_uses = 3  # Can use shovel 3 times
            self.grid[self.agent_pos] = EMPTY
            self.items['shovel'] = None
            return 100  # Reward for collecting shovel
            
        elif tuple(self.agent_pos) == self.items.get('sword'):
            self.has_sword = True
            self.grid[self.agent_pos] = EMPTY
            self.items['sword'] = None
            return 100  # Reward for collecting sword
            
        elif tuple(self.agent_pos) == self.items.get('magnet'):
            self.has_magnet = True
            self.grid[self.agent_pos] = EMPTY
            self.items['magnet'] = None
            return 100  # Reward for collecting magnet
            
        elif tuple(self.agent_pos) == self.items.get('key'):
            self.has_key = True
            self.grid[self.agent_pos] = EMPTY
            self.items['key'] = None
            return 100  # Reward for collecting key
            
        return 0
    
    def _generate_items(self):
        """Generate items for Level 3"""
        if self.mode != MODE_LEVEL3:
            return
            
        self.items = {
            "shovel": None, 
            "sword": None, 
            "magnet": None, 
            "key": None
        }
        
        available_positions = [
            (i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
            if (i, j) not in self.obstacles and 
               (i, j) != START_POS and 
               (i, j) != GOAL_POS and 
               (i, j) != self.agent_pos and
               (i, j) not in self.coins and
               not any(monster.pos == (i, j) for monster in self.monsters)
        ]
        
        if len(available_positions) < 4:  # Need at least 4 positions for all items
            return
            
        # Add key (required)
        key_idx = random.randrange(len(available_positions))
        key_pos = available_positions[key_idx]
        self.grid[key_pos] = KEY
        self.items['key'] = key_pos
        available_positions.pop(key_idx)
        
        # Add shovel
        if available_positions:
            shovel_idx = random.randrange(len(available_positions))
            shovel_pos = available_positions[shovel_idx]
            self.grid[shovel_pos] = SHOVEL
            self.items['shovel'] = shovel_pos
            available_positions.pop(shovel_idx)
        
        # Add sword
        if available_positions:
            sword_idx = random.randrange(len(available_positions))
            sword_pos = available_positions[sword_idx]
            self.grid[sword_pos] = SWORD
            self.items['sword'] = sword_pos
            available_positions.pop(sword_idx)
        
        # Add magnet
        if available_positions:
            magnet_idx = random.randrange(len(available_positions))
            magnet_pos = available_positions[magnet_idx]
            self.grid[magnet_pos] = MAGNET
            self.items['magnet'] = magnet_pos
            available_positions.pop(magnet_idx) 