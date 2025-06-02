"""
Map processor module for Game 1.
Processes grid data into formats suitable for agent consumption.
"""

import numpy as np
from typing import Dict, Tuple, List, Set

from ..game.config import *

class MapProcessor:
    """Processes game maps for agent consumption."""
    
    @staticmethod
    def format_map_for_agent(
        grid: np.ndarray, 
        vision_map: np.ndarray, 
        agent_pos: Tuple[int, int],
        coins: Set[Tuple[int, int]] = None,
        lives: int = 3,
        score: int = 0,
        mode: str = MODE_LEVEL2,
        monsters: List[Tuple[int, int]] = None,
        obstacles: Set[Tuple[int, int]] = None
    ) -> str:
        """
        Format map as text representation for agent.
        
        Args:
            grid: Game grid array
            vision_map: Map of discovered cells
            agent_pos: Agent position (row, col)
            coins: Set of coin positions
            lives: Remaining lives
            score: Current score
            mode: Game mode
            monsters: List of monster positions
            obstacles: Set of obstacle positions
            
        Returns:
            Text representation of the map
        """
        result = []
        
        # Check if full vision mode is active
        is_full_vision = np.all(vision_map == DISCOVERED)
        
        # Add coordinate reference header
        col_header = "   " + " ".join([str(i) for i in range(GRID_SIZE)])
        result.append(col_header)
        
        # Generate map representation row by row
        for i in range(GRID_SIZE):
            row = [f"{i:2d}"]
            for j in range(GRID_SIZE):
                # Handle undiscovered cells
                if vision_map[i, j] == NOT_DISCOVERED and not is_full_vision:
                    cell = "?"
                else:
                    # Agent position
                    if (i, j) == tuple(agent_pos):
                        cell = "A"
                    # Start position
                    elif (i, j) == START_POS:
                        cell = "S"
                    # Goal position
                    elif (i, j) == GOAL_POS:
                        cell = "G"
                    # Coin
                    elif coins and (i, j) in coins:
                        cell = "C"
                    # Obstacle
                    elif obstacles and (i, j) in obstacles:
                        cell = "#"
                    # Monster (only in levels 2 and 3)
                    elif monsters:
                        monster_at_position = False
                        for monster_pos in monsters:
                            if isinstance(monster_pos, tuple) and monster_pos == (i, j):
                                monster_at_position = True
                                break
                        if monster_at_position:
                            cell = "M"
                        # Special items (only in level 3)
                        elif mode == MODE_LEVEL3 and grid[i, j] > EMPTY:
                            if grid[i, j] == SHOVEL:
                                cell = "T"  # Shovel
                            elif grid[i, j] == SWORD:
                                cell = "W"  # Sword
                            elif grid[i, j] == MAGNET:
                                cell = "N"  # Magnet
                            elif grid[i, j] == KEY:
                                cell = "K"  # Key
                            else:
                                cell = "."
                        # Empty space
                        else:
                            cell = "."
                    # Special items without monsters
                    elif mode == MODE_LEVEL3 and grid[i, j] > EMPTY:
                        if grid[i, j] == SHOVEL:
                            cell = "T"
                        elif grid[i, j] == SWORD:
                            cell = "W"
                        elif grid[i, j] == MAGNET:
                            cell = "N"
                        elif grid[i, j] == KEY:
                            cell = "K"
                        else:
                            cell = "."
                    # Empty space
                    else:
                        cell = "."
                
                row.append(cell)
            
            result.append(" ".join(row))
        
        # Add full vision mode indicator if applicable
        if is_full_vision:
            result.append("\n[Full Vision Mode: You can see the entire map]")
        
        return "\n".join(result)
    
    @staticmethod
    def get_actions_description() -> str:
        """
        Get formatted description of available actions.
        
        Returns:
            Description of the 12 possible actions
        """
        actions = []
        for i in range(12):
            direction = ["UP", "DOWN", "LEFT", "RIGHT"][i // 3]
            steps = (i % 3) + 1
            actions.append(f"- {i}: Move {direction} {steps} step{'s' if steps > 1 else ''}")
        
        return "\n".join(actions) 