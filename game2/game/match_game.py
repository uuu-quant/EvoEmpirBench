"""
Match-3 puzzle game environment.
Implements a color-matching puzzle game where the agent must clear colored tiles.
"""

import random
import json
import time
import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from nips2025.game2.game.config import *
from nips2025.common.utils import ensure_dir, save_json, get_timestamp

class MatchGame:
    """
    Match-3 puzzle game environment.
    
    Features:
    - 8x8 grid with colored tiles (A, B, C, D)
    - Score system based on number of tiles cleared
    - Special powers (row clear, column clear, bomb, hammer)
    - Different difficulty levels
    """
    
    def __init__(self, results_dir=None, levels_dir=None):
        """
        Initialize match game.
        
        Args:
            results_dir: Directory to store results
            levels_dir: Directory containing level files
        """
        # Set directories
        self.results_dir = results_dir
        self.levels_dir = levels_dir or LEVELS_DIR
        
        # Create game state
        self.board = []
        self.score = 0
        self.steps_remaining = 0
        self.max_steps = 0
        self.steps_used = 0
        self.current_level = 0
        self.current_difficulty = 'easy'
        self.game_over = False
        self.inventory = {'row': 1, 'col': 1, 'bomb': 1, 'hammer': 1}
        self.color_targets = {}
        self.color_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.total_cleared = 0
        self.clear_counts = []
        self.session_dir = ""
        self.session_timestamp = ""
        
        # Initialize levels
        self.levels = {'easy': [], 'medium': [], 'hard': []}
        self.load_or_generate_levels()
    
    def load_or_generate_levels(self):
        """Load existing levels or generate new ones."""
        os.makedirs(self.levels_dir, exist_ok=True)
        
        for difficulty in DIFFICULTY_LEVELS:
            difficulty_dir = os.path.join(self.levels_dir, difficulty)
            os.makedirs(difficulty_dir, exist_ok=True)
            levels = []
            
            for i in range(1, LEVELS_PER_DIFFICULTY + 1):
                level_path = os.path.join(difficulty_dir, f"level{i:02d}.json")
                
                if os.path.exists(level_path):
                    # Load existing level
                    with open(level_path, 'r', encoding='utf-8') as f:
                        level = json.load(f)
                    levels.append(level)
                    print(f"Loaded {difficulty} level {i}")
                else:
                    # Generate levels if none exist
                    print(f"No existing levels found for {difficulty}, generating new levels")
                    self.generate_levels()
                    return
            
            self.levels[difficulty] = levels
    
    def generate_levels(self):
        """Generate game levels for all difficulties."""
        for difficulty, params in DIFFICULTY_LEVELS.items():
            levels = []
            difficulty_dir = os.path.join(self.levels_dir, difficulty)
            os.makedirs(difficulty_dir, exist_ok=True)
            
            for i in range(1, LEVELS_PER_DIFFICULTY + 1):
                # Calculate parameters based on difficulty
                base_steps = params['steps_base'] + random.randint(*params['steps_range'])
                target_range = params['target_range']
                
                # Generate color targets
                color_targets = {
                    'A': random.randint(*target_range),
                    'B': random.randint(*target_range),
                    'C': random.randint(*target_range),
                    'D': random.randint(*target_range)
                }
                
                # Calculate steps based on targets
                total_targets = sum(color_targets.values())
                steps = max(base_steps, int(total_targets / AVG_CLEAR_PER_STEP + 0.5))
                
                # Generate initial board
                board = [[random.choice(COLOR_KEYS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                
                # Ensure there are valid moves on the board
                has_valid_moves = False
                while not has_valid_moves:
                    for k in range(GRID_SIZE):
                        for l in range(GRID_SIZE):
                            region = self.find_connected_region(k, l, board[k][l], 
                                                              [[False]*GRID_SIZE for _ in range(GRID_SIZE)], 
                                                              board)
                            if len(region) >= 2:
                                has_valid_moves = True
                                break
                        if has_valid_moves:
                            break
                    
                    if not has_valid_moves:
                        board = [[random.choice(COLOR_KEYS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                
                # Create level data
                level = {
                    'board': board,
                    'max_steps': steps,
                    'inventory': {'row': 1, 'col': 1, 'bomb': 1, 'hammer': 1},
                    'color_targets': color_targets
                }
                
                levels.append(level)
                
                # Save level to file
                level_path = os.path.join(difficulty_dir, f"level{i:02d}.json")
                with open(level_path, 'w', encoding='utf-8') as f:
                    json.dump(level, f, indent=2)
                
                print(f"Generated {difficulty} level {i}: max_steps={steps}, color_targets={color_targets}")
            
            self.levels[difficulty] = levels
    
    def init_game(self, difficulty, level):
        """
        Initialize game with specified difficulty and level.
        
        Args:
            difficulty: Game difficulty
            level: Level number
        """
        self.current_difficulty = difficulty
        self.current_level = level
        
        # Load level data
        self.board = [row[:] for row in self.levels[difficulty][level-1]['board']]
        self.max_steps = self.levels[difficulty][level-1]['max_steps']
        self.inventory = dict(self.levels[difficulty][level-1]['inventory'])
        self.color_targets = dict(self.levels[difficulty][level-1]['color_targets'])
        
        # Initialize game state
        self.color_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.score = 0
        self.steps_remaining = self.max_steps
        self.steps_used = 0
        self.game_over = False
        self.total_cleared = 0
        self.clear_counts = []
        
        # Set up session
        self.session_timestamp = get_timestamp()
        
        # Create results directory
        if self.results_dir:
            self.session_dir = os.path.join(
                self.results_dir,
                "game2",
                difficulty,
                f"level{level:02d}",
                "agent"
            )
            os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"Initialized {difficulty} level {level}: max_steps={self.max_steps}, color_targets={self.color_targets}")
        
        # Save initial state
        self.save_state()
        
        return self.get_state_dict()
    
    def get_state_dict(self):
        """
        Get current game state as dictionary.
        
        Returns:
            Game state dictionary
        """
        return {
            'board': self.board,
            'score': self.score,
            'steps_remaining': self.steps_remaining,
            'max_steps': self.max_steps,
            'steps_used': self.steps_used,
            'inventory': self.inventory,
            'color_targets': self.color_targets,
            'color_counts': self.color_counts,
            'total_cleared': self.total_cleared,
            'difficulty': self.current_difficulty,
            'level': self.current_level
        }
    
    def find_connected_region(self, i, j, color, visited, board_ref):
        """
        Find connected region of same-colored tiles.
        
        Args:
            i, j: Starting position
            color: Color to match
            visited: Array tracking visited positions
            board_ref: Board reference
            
        Returns:
            List of positions in connected region
        """
        # Boundary check
        if i < 0 or i >= GRID_SIZE or j < 0 or j >= GRID_SIZE:
            return []
        
        # Already visited or different color
        if visited[i][j] or board_ref[i][j] != color:
            return []
        
        # Mark as visited
        visited[i][j] = True
        region = [[i, j]]
        
        # Check in four directions
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for di, dj in directions:
            region.extend(self.find_connected_region(i + di, j + dj, color, visited, board_ref))
        
        return region
    
    def shift_columns(self):
        """Shift tiles down to fill empty spaces and add new tiles."""
        for j in range(GRID_SIZE):
            # Find bottom-most empty space
            empty = -1
            for i in range(GRID_SIZE - 1, -1, -1):
                if self.board[i][j] is None:
                    empty = i
                    break
            
            if empty >= 0:
                # Move existing tiles down
                for i in range(empty, -1, -1):
                    if self.board[i][j] is not None:
                        self.board[empty][j] = self.board[i][j]
                        self.board[i][j] = None
                        empty -= 1
                
                # Fill remaining empty spaces with new tiles
                for i in range(empty, -1, -1):
                    self.board[i][j] = random.choice(COLOR_KEYS)
    
    def clear_region(self, region):
        """
        Clear a connected region of tiles.
        
        Args:
            region: List of positions to clear
            
        Returns:
            Whether the operation was successful
        """
        # Check minimum region size
        if len(region) < 2:
            print(f"Region size {len(region)} less than 2, cannot clear")
            return False
        
        # Clear tiles and update color counts
        for i, j in region:
            color = self.board[i][j]
            if color:
                self.color_counts[color] += 1
            self.board[i][j] = None
        
        # Calculate score
        grid_count = len(region)
        self.score += grid_count * BASE_SCORE_MULTIPLIER + BONUS_MULTIPLIER * max(0, grid_count - MIN_COUNT)
        
        # Update steps
        self.steps_remaining = max(0, self.steps_remaining - 1)
        self.steps_used += 1
        
        # Update clear stats
        self.total_cleared += grid_count
        self.clear_counts.append(grid_count)
        
        # Shift tiles down
        self.shift_columns()
        
        # Check game status
        self.check_game_over()
        
        # Save state
        self.save_state()
        
        return True
    
    def use_prop(self, prop, target):
        """
        Use a special power.
        
        Args:
            prop: Power type ('row', 'col', 'bomb', 'hammer')
            target: Target position or index
            
        Returns:
            Whether the operation was successful
        """
        # Check if prop can be used
        if self.score < COSTS[prop] or self.inventory[prop] == 0:
            print(f"Cannot use prop {prop}: insufficient score or no props")
            return False
        
        # Deduct cost and inventory
        self.inventory[prop] -= 1
        self.score -= COSTS[prop]
        
        # Track cleared tiles
        cleared_count = 0
        need_shift = False
        
        # Apply prop effect
        if prop == 'row':
            # Clear entire row
            for j in range(GRID_SIZE):
                color = self.board[target][j]
                if color:
                    self.color_counts[color] += 1
                    cleared_count += 1
                self.board[target][j] = None
            need_shift = True
            
        elif prop == 'col':
            # Clear entire column
            for i in range(GRID_SIZE):
                color = self.board[i][target]
                if color:
                    self.color_counts[color] += 1
                    cleared_count += 1
                self.board[i][target] = None
            need_shift = True
            
        elif prop == 'bomb':
            # Clear 3x3 area
            i, j = target
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                        color = self.board[ni][nj]
                        if color:
                            self.color_counts[color] += 1
                            cleared_count += 1
                        self.board[ni][nj] = None
            need_shift = True
            
        elif prop == 'hammer':
            # Clear single tile
            i, j = target
            if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE and self.board[i][j] is not None:
                color = self.board[i][j]
                self.color_counts[color] += 1
                cleared_count += 1
                self.board[i][j] = None
                need_shift = True
            else:
                # Invalid target
                self.inventory[prop] += 1
                self.score += COSTS[prop]
                print(f"Cannot use hammer at {target}: invalid position")
                return False
        
        # Update steps and shift tiles if needed
        if need_shift:
            self.steps_remaining = max(0, self.steps_remaining - 1)
            self.steps_used += 1
            self.total_cleared += cleared_count
            self.clear_counts.append(cleared_count)
            self.shift_columns()
        
        # Check game status
        self.check_game_over()
        
        # Save state
        self.save_state()
        
        return True
    
    def step(self, action):
        """
        Take a game step with the given action.
        
        Args:
            action: Action dictionary with type and target
            
        Returns:
            (state, reward, done, info): Step results
        """
        reward = 0
        cleared_count = 0
        success = False
        
        if self.game_over:
            return self.get_state_dict(), reward, self.game_over, {"message": "Game already over"}
        
        # Get previous state for comparison
        prev_score = self.score
        prev_steps = self.steps_remaining
        
        # Apply action
        if action and 'action' in action and action['action']:
            action_data = action['action']
            action_type = action_data.get('type')
            pos = action_data.get('pos')
            index = action_data.get('index')
            
            if action_type == 'eliminate' and pos:
                # Clear connected region
                if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE and self.board[pos[0]][pos[1]]:
                    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
                    region = self.find_connected_region(pos[0], pos[1], self.board[pos[0]][pos[1]], 
                                                      visited, self.board)
                    if len(region) >= 2:
                        success = self.clear_region(region)
                        cleared_count = len(region) if success else 0
                    else:
                        print(f"Region size {len(region)} less than 2, cannot clear")
                else:
                    print(f"Invalid position: {pos}")
                    
            elif action_type in ['row', 'col', 'bomb', 'hammer']:
                # Use special power
                if action_type in ['row', 'col']:
                    success = self.use_prop(action_type, index)
                    cleared_count = GRID_SIZE if success else 0
                elif action_type == 'bomb':
                    success = self.use_prop(action_type, pos)
                    cleared_count = 9 if success else 0  # 3x3 area
                elif action_type == 'hammer':
                    success = self.use_prop(action_type, pos)
                    cleared_count = 1 if success else 0
            else:
                print(f"Unknown action type: {action_type}")
        
        # Calculate reward
        reward = self.score - prev_score
        
        # Return step results
        return self.get_state_dict(), reward, self.game_over, {
            "cleared_count": cleared_count,
            "success": success
        }
    
    def save_state(self):
        """Save current game state to file."""
        if not self.session_dir:
            return
        
        state = {
            'board': self.board,
            'score': self.score,
            'steps_remaining': self.steps_remaining,
            'max_steps': self.max_steps,
            'steps_used': self.steps_used,
            'current_difficulty': self.current_difficulty,
            'current_level': self.current_level,
            'game_over': self.game_over,
            'inventory': self.inventory,
            'color_targets': self.color_targets,
            'color_counts': self.color_counts,
            'total_cleared': self.total_cleared,
            'clear_counts': self.clear_counts
        }
        
        state_path = os.path.join(self.session_dir, STATE_FILE.format(self.session_timestamp))
        try:
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Error saving game state: {str(e)}")
    
    def check_game_over(self):
        """Check if the game is over."""
        # Check steps exhausted
        if self.steps_remaining <= 0:
            self.game_over = True
        
        # Check targets met
        targets_met = all(self.color_counts[color] >= self.color_targets[color] for color in COLOR_KEYS)
        if targets_met:
            self.game_over = True
        
        # Log game over reason
        if self.game_over:
            if targets_met:
                print("Game over: All color targets achieved, level cleared!")
            else:
                print("Game over: Steps exhausted, targets not met")
            
            # Save end-game stats
            if self.session_dir:
                avg_score_per_step = self.score / self.steps_used if self.steps_used > 0 else 0
                avg_clear_per_step = self.total_cleared / self.steps_used if self.steps_used > 0 else 0
                
                stats = {
                    'total_score': self.score,
                    'cleared': targets_met,
                    'steps_remaining': self.steps_remaining,
                    'avg_score_per_step': avg_score_per_step,
                    'avg_clear_per_step': avg_clear_per_step
                }
                
                stats_path = os.path.join(self.session_dir, STATS_FILE.format(self.session_timestamp))
                try:
                    with open(stats_path, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2)
                except Exception as e:
                    print(f"Error saving stats: {str(e)}")
    
    def has_valid_moves(self):
        """
        Check if the board has any valid moves.
        
        Returns:
            Whether valid moves exist
        """
        # Check for connected regions
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.board[i][j]:
                    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
                    region = self.find_connected_region(i, j, self.board[i][j], visited, self.board)
                    if len(region) >= 2:
                        return True
        
        # Check if props can create valid moves
        return self.can_props_create_moves()
    
    def can_props_create_moves(self):
        """
        Check if using props can create valid moves.
        
        Returns:
            Whether props can create valid moves
        """
        for prop, cost in COSTS.items():
            if self.score >= cost and self.inventory[prop] > 0:
                return True
        
        return False 