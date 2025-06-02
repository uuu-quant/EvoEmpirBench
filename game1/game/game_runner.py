"""
Game runner for maze navigation game.
Provides a convenient way to run and test the maze navigation game.
"""

import os
import argparse
import time
import json
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from .environment import MazeNavigationEnv
from .config import *
from .map_generator import MapGenerator

class GameRunner:
    """
    Runner class for maze navigation game
    """
    def __init__(self, 
                mode: str = MODE_LEVEL2, 
                full_vision: bool = False,
                maps_dir: str = None):
        """
        Initialize game runner
        
        Args:
            mode: Game mode/difficulty
            full_vision: Whether agent has full visibility
            maps_dir: Directory for map files
        """
        self.mode = mode
        self.full_vision = full_vision
        self.maps_dir = maps_dir if maps_dir else "maps"
        self.env = MazeNavigationEnv(mode=mode, full_vision=full_vision)
        self.maps = []
        
        # Create maps directory if needed
        os.makedirs(self.maps_dir, exist_ok=True)
        
    def generate_maps(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate maps for current mode
        
        Args:
            count: Number of maps to generate
            
        Returns:
            List of generated maps
        """
        print(f"Generating {count} maps for mode {self.mode}...")
        self.maps = MapGenerator.generate_maps(count, self.mode, self.maps_dir)
        return self.maps
    
    def load_maps(self) -> List[Dict[str, Any]]:
        """
        Load maps for current mode
        
        Returns:
            List of loaded maps
        """
        collection_filename = f"{self.mode.replace(' ', '_')}_collection.json"
        collection_filepath = os.path.join(self.maps_dir, collection_filename)
        
        if os.path.exists(collection_filepath):
            print(f"Loading maps from {collection_filepath}")
            self.maps = MapGenerator.load_map_collection(collection_filepath)
            return self.maps
        else:
            print(f"No map collection found at {collection_filepath}")
            return []
    
    def run_map(self, map_data: Dict[str, Any], map_index: int = 0):
        """
        Run a single map
        
        Args:
            map_data: Map data dictionary
            map_index: Map index
        """
        # Load map
        self.env.load_map(map_data, map_index)
        
        print(f"Running map {map_index} for mode {self.mode}")
        print("Controls: Arrow keys to move (hold Shift for 2 steps, Ctrl for 3 steps)")
        print("          R to reset, Q to quit")
        
        # Game loop
        running = True
        while running:
            # Render environment
            self.env.render()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Get modifier keys
                    steps = 1
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        steps = 2
                    elif pygame.key.get_mods() & pygame.KMOD_CTRL:
                        steps = 3
                    
                    # Convert key to action
                    action = None
                    if event.key == pygame.K_UP:
                        action = (0 * 3) + (steps - 1)  # Up
                    elif event.key == pygame.K_DOWN:
                        action = (1 * 3) + (steps - 1)  # Down
                    elif event.key == pygame.K_LEFT:
                        action = (2 * 3) + (steps - 1)  # Left
                    elif event.key == pygame.K_RIGHT:
                        action = (3 * 3) + (steps - 1)  # Right
                    elif event.key == pygame.K_r:
                        # Reset environment
                        self.env.load_map(map_data, map_index)
                        continue
                    elif event.key == pygame.K_q:
                        running = False
                        break
                    
                    # Take action if valid
                    if action is not None:
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        print(f"Action: {self.env.get_action_meaning(action)}, Reward: {reward}")
                        
                        if terminated:
                            print("Game over!")
                            stats = self.env.end_session(True, self.env.agent_pos == GOAL_POS)
                            print(f"Stats: {stats}")
                            time.sleep(2)
                            self.env.load_map(map_data, map_index)
            
            # Limit frame rate
            time.sleep(0.05)
        
        # Close environment
        self.env.close()
    
    def run_random_map(self):
        """Run a random map from the loaded maps"""
        if not self.maps:
            if not self.load_maps():
                print("Generating a new map...")
                self.generate_maps(1)
        
        if self.maps:
            map_index = random.randint(0, len(self.maps) - 1)
            self.run_map(self.maps[map_index], map_index)
        else:
            print("No maps available")
    
    def run_all_maps(self):
        """Run all loaded maps in sequence"""
        if not self.maps:
            if not self.load_maps():
                print("No maps available. Generating new maps...")
                self.generate_maps(5)
        
        for i, map_data in enumerate(self.maps):
            print(f"Running map {i+1}/{len(self.maps)}")
            self.run_map(map_data, i)
            
            # Break if user closed the window
            if not pygame.get_init():
                break


def main():
    """Main function for running the game"""
    parser = argparse.ArgumentParser(description="Maze Navigation Game Runner")
    parser.add_argument("--mode", type=str, default=MODE_LEVEL2, 
                      choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3],
                      help="Game mode/difficulty")
    parser.add_argument("--full-vision", action="store_true", 
                      help="Enable full visibility of the maze")
    parser.add_argument("--maps-dir", type=str, default="maps",
                      help="Directory for map files")
    parser.add_argument("--generate-maps", type=int, default=0,
                      help="Generate specified number of maps and exit")
    parser.add_argument("--run-all", action="store_true",
                      help="Run all available maps in sequence")
    
    args = parser.parse_args()
    
    # Create game runner
    runner = GameRunner(
        mode=args.mode,
        full_vision=args.full_vision,
        maps_dir=args.maps_dir
    )
    
    # Generate maps if requested
    if args.generate_maps > 0:
        runner.generate_maps(args.generate_maps)
        return
    
    # Run maps
    if args.run_all:
        runner.run_all_maps()
    else:
        runner.run_random_map()


if __name__ == "__main__":
    import pygame  # Import here to avoid circular import
    main() 