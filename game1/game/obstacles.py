"""
Obstacle generation for maze navigation game.
Defines various obstacle shapes and patterns for the maze.
"""

import numpy as np
from typing import List, Tuple, Set
import random

class Obstacle:
    """Utility class for creating and validating obstacle patterns"""
    
    @staticmethod
    def create_cross(center: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Create a cross-shaped obstacle
        
        Args:
            center: Center position of the cross
            
        Returns:
            List of positions forming the cross
        """
        x, y = center
        return [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]

    @staticmethod
    def create_l_shape(corner: Tuple[int, int], orientation: int = 0) -> List[Tuple[int, int]]:
        """
        Create an L-shaped obstacle with specified orientation
        
        Args:
            corner: Corner position of the L-shape
            orientation: Rotation angle (0, 1, 2, or 3 for 0, 90, 180, 270 degrees)
            
        Returns:
            List of positions forming the L-shape
        """
        x, y = corner
        orientations = [
            [(x, y), (x-1, y), (x, y+1)],  # Base orientation
            [(x, y), (x+1, y), (x, y+1)],  # Rotated 90 degrees
            [(x, y), (x+1, y), (x, y-1)],  # Rotated 180 degrees
            [(x, y), (x-1, y), (x, y-1)]   # Rotated 270 degrees
        ]
        return orientations[orientation % 4]

    @staticmethod
    def create_rectangle(top_left: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
        """
        Create a rectangular obstacle
        
        Args:
            top_left: Top-left position of the rectangle
            width: Width of the rectangle
            height: Height of the rectangle
            
        Returns:
            List of positions forming the rectangle
        """
        x, y = top_left
        return [(i, j) for i in range(x, x + height) for j in range(y, y + width)]

    @staticmethod
    def create_line(start: Tuple[int, int], length: int, horizontal: bool = True) -> List[Tuple[int, int]]:
        """
        Create a line obstacle
        
        Args:
            start: Starting position of the line
            length: Length of the line
            horizontal: Whether the line is horizontal (True) or vertical (False)
            
        Returns:
            List of positions forming the line
        """
        x, y = start
        if horizontal:
            return [(x, y + i) for i in range(length)]
        return [(x + i, y) for i in range(length)]

    @staticmethod
    def validate_positions(positions: List[Tuple[int, int]], grid_size: int) -> bool:
        """
        Validate if obstacle positions are within grid boundaries
        
        Args:
            positions: List of positions to validate
            grid_size: Size of the grid
            
        Returns:
            True if all positions are valid, False otherwise
        """
        return all(0 <= x < grid_size and 0 <= y < grid_size for x, y in positions)

    @staticmethod
    def get_random_obstacle_layout(grid_size: int, 
                                  forbidden_positions: List[Tuple[int, int]] = None,
                                  max_obstacles_percent: float = 0.3) -> Set[Tuple[int, int]]:
        """
        Generate a random obstacle layout based on grid size
        
        Args:
            grid_size: Size of the grid
            forbidden_positions: Positions that should not contain obstacles
            max_obstacles_percent: Maximum percentage of grid cells that can be obstacles
            
        Returns:
            Set of obstacle positions
        """
        if forbidden_positions is None:
            forbidden_positions = []
            
        # Ensure start and goal positions are not occupied
        start_pos = (0, 0)  # Top-left corner
        goal_pos = (grid_size-1, grid_size-1)  # Bottom-right corner
        forbidden_positions.extend([start_pos, goal_pos])
        
        obstacles = set()
        max_obstacles = int(grid_size * grid_size * max_obstacles_percent)
        
        # Determine difficulty based on grid size
        if grid_size <= 7:  # Level 1
            num_obstacles = random.randint(2, 4)
            max_obstacle_size = 2  # Smaller obstacle size
        elif grid_size <= 9:  # Level 2
            num_obstacles = random.randint(3, 5)
            max_obstacle_size = 3
        else:  # Level 3
            num_obstacles = random.randint(4, 7)
            max_obstacle_size = 3
        
        # Add random obstacles
        for _ in range(num_obstacles):
            if len(obstacles) >= max_obstacles:
                break
                
            obstacle_type = random.choice(['cross', 'l_shape', 'rectangle', 'line'])
            
            attempts = 0
            while attempts < 10:  # Limit attempts to prevent infinite loops
                attempts += 1
                
                if obstacle_type == 'cross':
                    center = (random.randint(1, grid_size-2), 
                             random.randint(1, grid_size-2))
                    new_obstacles = Obstacle.create_cross(center)
                
                elif obstacle_type == 'l_shape':
                    corner = (random.randint(1, grid_size-2),
                             random.randint(1, grid_size-2))
                    orientation = random.randint(0, 3)
                    new_obstacles = Obstacle.create_l_shape(corner, orientation)
                
                elif obstacle_type == 'rectangle':
                    top_left = (random.randint(1, grid_size-3),
                               random.randint(1, grid_size-3))
                    width = random.randint(1, min(max_obstacle_size, grid_size-top_left[1]-1))
                    height = random.randint(1, min(max_obstacle_size, grid_size-top_left[0]-1))
                    new_obstacles = Obstacle.create_rectangle(top_left, width, height)
                
                else:  # line
                    start = (random.randint(1, grid_size-max_obstacle_size-1),
                            random.randint(1, grid_size-max_obstacle_size-1))
                    length = random.randint(2, min(max_obstacle_size+1, grid_size-max(start)-1))
                    horizontal = random.choice([True, False])
                    new_obstacles = Obstacle.create_line(start, length, horizontal)
                
                # Validate new obstacles
                if (Obstacle.validate_positions(new_obstacles, grid_size) and
                    not any(pos in forbidden_positions for pos in new_obstacles) and
                    not any(pos in obstacles for pos in new_obstacles)):
                    obstacles.update(new_obstacles)
                    break
        
        return obstacles 