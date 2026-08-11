#!/usr/bin/env python
"""
Training script for the Maze Navigation game agents.
Trains agents with truth knowledge learning capabilities.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path to allow relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from ..game.environment import MazeNavigationEnv
from ..game.config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from ..agent.learning_agent import MazeNavigationAgent
from ..game.map_generator import MapGenerator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train learning agents with memory and truth knowledge")
    parser.add_argument("--api_key", type=str, default=None,
                      help="API key for the model")
    parser.add_argument("--model", type=str, default="gpt-4",
                      help="Model name to use")
    parser.add_argument("--base_url", type=str, default=None,
                      help="Base URL for API")
    parser.add_argument("--maps_dir", type=str, default=None,
                      help="Maps directory")
    parser.add_argument("--memory_dir", type=str, default=None,
                      help="Memory storage directory")
    parser.add_argument("--results_dir", type=str, default=None,
                      help="Results directory")
    parser.add_argument("--maps_count", type=int, default=5,
                      help="Number of maps to train on per difficulty")
    parser.add_argument("--max_attempts", type=int, default=3,
                      help="Maximum attempts per map")
    parser.add_argument("--max_steps", type=int, default=200,
                      help="Maximum steps per attempt")
    parser.add_argument("--mode", type=str, default=None,
                      choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3],
                      help="Train only on specific mode (default: all modes)")
    parser.add_argument("--start_map", type=int, default=0,
                      help="Start training from this map index")
    parser.add_argument("--clear_memory", action="store_true",
                      help="Clear all existing memory and truth knowledge")
    parser.add_argument("--skip_validation", action="store_true",
                      help="Skip validation step, promote memories directly to truth")
    parser.add_argument("--iterations", type=int, default=3,
                      help="Number of training iterations")
                      
    args = parser.parse_args()
    
    # Ensure directories exist
    if args.memory_dir:
        os.makedirs(args.memory_dir, exist_ok=True)
    
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        
    if args.maps_dir:
        os.makedirs(args.maps_dir, exist_ok=True)
    
    return args

def load_maps(mode: str, maps_dir: str) -> List[Dict[str, Any]]:
    """
    Load maps for a specific mode
    
    Args:
        mode: Game mode
        maps_dir: Maps directory
        
    Returns:
        List of maps
    """
    # Set map file path
    collection_file = os.path.join(maps_dir, f"{mode.replace(' ', '_')}.json")
    
    # Check if map file exists
    if not os.path.exists(collection_file):
        print(f"Map file {collection_file} not found, using default maps")
        # Generate maps if needed
        print(f"Generating new maps...")
        maps = MapGenerator.generate_maps(30, mode, maps_dir)
        return maps
    
    # Load maps
    try:
        with open(collection_file, 'r') as f:
            maps = json.load(f)
        print(f"Loaded {len(maps)} maps for {mode} mode")
        return maps
    except Exception as e:
        print(f"Failed to load maps: {str(e)}")
        return []

def run_episode(agent: MazeNavigationAgent, env: MazeNavigationEnv, map_data: Dict[str, Any], 
               map_index: int, max_steps: int = 200, use_memory: bool = True) -> Dict[str, Any]:
    """
    Run a single episode
    
    Args:
        agent: Learning agent
        env: Game environment
        map_data: Map data
        map_index: Map index
        max_steps: Maximum steps
        use_memory: Whether to use memory
        
    Returns:
        Episode metrics
    """
    # Load map
    env.load_map(map_data, map_index)
    
    # Start session
    agent.start_session(env.mode, map_index)
    
    # Game loop
    done = False
    step_count = 0
    state = env.reset()[0]
    info = {}
    
    # Store initial observation
    agent.observe(state, info)
    
    while not done and step_count < max_steps:
        # Get current state
        state = env.get_state()
        info = env.get_info()
        
        # Get agent action
        action, response = agent.get_action(state, info, with_memory=use_memory)
        
        # Execute action
        next_state, reward, terminated, truncated, next_info = env.step(action)
        
        # Log interaction
        agent.log_interaction(state, action, response, reward, next_info)
        
        # Update state
        state = next_state
        info = next_info
        done = terminated or truncated
        
        step_count += 1
    
    # End session
    success = done and env.goal_reached
    exploration_rate = env._compute_exploration_rate() if hasattr(env, '_compute_exploration_rate') else 0.0
    agent.end_session(success, env.score, exploration_rate, step_count)
    
    return {
        "success": success,
        "score": env.score,
        "steps_used": step_count,
        "exploration_rate": exploration_rate
    }

def train_on_map(agent: MazeNavigationAgent, env: MazeNavigationEnv, map_data: Dict[str, Any], 
              max_episode_steps: int = 100, with_memory: bool = True) -> bool:
    """
    Train agent on a specific map
    
    Args:
        agent: Learning agent
        env: Game environment
        map_data: Map data
        max_episode_steps: Maximum steps per episode
        with_memory: Whether to use memory enhancement
        
    Returns:
        Whether validation was successful
    """
    # Load map
    env.load_map(map_data, map_data.get('map_index', 0))
    
    # ========== Round 1: Initial experience ==========
    print(f"\n=== Starting initial experience on map {map_data.get('map_index', 0)} ===")
    
    # Get metrics from first round
    initial_metrics = run_episode(
        agent=agent,
        env=env,
        map_data=map_data,
        map_index=map_data.get('map_index', 0),
        max_steps=max_episode_steps,
        use_memory=True  # Use global truth knowledge but not map-specific memory
    )
    
    # Generate reflection
    experience_summary, strengths, weaknesses = agent.reflect_on_session()
    
    # Record as subjective memory
    agent.record_subjective_memory(experience_summary, strengths, weaknesses)
    
    # ========== Round 2: Validation experience (with subjective memory) ==========
    print(f"\n=== Starting validation experience on map {map_data.get('map_index', 0)} ===")
    
    # Get metrics from second round
    validation_metrics = run_episode(
        agent=agent,
        env=env,
        map_data=map_data,
        map_index=map_data.get('map_index', 0),
        max_steps=max_episode_steps,
        use_memory=with_memory  # Use all memory (truth knowledge + map-specific subjective memory)
    )
    
    # Validate subjective memory
    is_valid = validation_metrics.get("score", 0) > initial_metrics.get("score", 0) or \
              (validation_metrics.get("success", False) and not initial_metrics.get("success", False))
    
    # If valid, promote subjective memory to truth knowledge
    if is_valid:
        agent.promote_memory_to_truth()
        print(f"Map {map_data.get('map_index', 0)} subjective memory validated successfully, promoted to truth knowledge")
    else:
        print(f"Map {map_data.get('map_index', 0)} subjective memory validation failed, not promoted to truth")
    
    return is_valid

def train_mode(agent: MazeNavigationAgent, maps_dir: str, mode: str, maps_count: int = 10, 
             with_memory: bool = True, max_episode_steps: int = 100):
    """
    Train agent on a specific mode
    
    Args:
        agent: Learning agent
        maps_dir: Maps directory
        mode: Game mode
        maps_count: Number of maps
        with_memory: Whether to use memory enhancement
        max_episode_steps: Maximum steps per episode
    """
    print(f"=== Starting training on {mode} mode ===")
    
    # Create environment
    env = MazeNavigationEnv(mode=mode)
    
    # Load maps
    all_maps = load_maps(mode, maps_dir)
    
    # Limit map count
    all_maps = all_maps[:maps_count]
    
    # Track successful validations
    successful_validations = 0
    
    # Process each map once
    for i, map_data in enumerate(tqdm(all_maps, desc=f"Training on {mode}")):
        # Set map index
        map_data['map_index'] = i
        
        # Train on current map (only one training cycle)
        is_successful = train_on_map(
            agent=agent,
            env=env,
            map_data=map_data,
            max_episode_steps=max_episode_steps,
            with_memory=with_memory
        )
        
        if is_successful:
            successful_validations += 1
    
    print(f"=== Completed training on {mode} mode ===")
    print(f"Total maps: {len(all_maps)}")
    print(f"Successful validations: {successful_validations}")
    print(f"Validation rate: {successful_validations / len(all_maps) * 100:.2f}%")
    
    # Close environment
    env.close()

def main():
    """Main function"""
    args = parse_args()
    
    # Use environment variables if API key not provided
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    
    if api_key is None:
        print("Error: No API key provided and OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Create default directories if not specified
    memory_dir = args.memory_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory")
    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    maps_dir = args.maps_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")
    
    # Ensure directories exist
    for dir_path in [memory_dir, results_dir, maps_dir]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensuring directory exists: {dir_path}")
    
    # Initialize GPT client
    from ...common.gpt_client import GPTClient
    api_client = GPTClient(
        key=api_key,
        url=args.base_url,
        model=args.model
    )
    
    # Create learning agent
    agent = MazeNavigationAgent(
        api_client=api_client,
        memory_dir=memory_dir,
        results_dir=results_dir
    )
    
    # Run training for specified iterations
    for iteration in range(args.iterations):
        print(f"Starting training iteration {iteration+1}/{args.iterations}")
        
        # Train on each difficulty level
        if args.mode == MODE_LEVEL1 or args.mode is None:
            train_mode(
                agent=agent,
                mode=MODE_LEVEL1,
                maps_dir=maps_dir,
                max_episode_steps=args.max_steps,
                maps_count=args.maps_count,
                with_memory=True
            )
        
        if args.mode == MODE_LEVEL2 or args.mode is None:
            train_mode(
                agent=agent,
                mode=MODE_LEVEL2,
                maps_dir=maps_dir,
                max_episode_steps=args.max_steps,
                maps_count=args.maps_count,
                with_memory=True
            )
        
        if args.mode == MODE_LEVEL3 or args.mode is None:
            train_mode(
                agent=agent,
                mode=MODE_LEVEL3,
                maps_dir=maps_dir,
                max_episode_steps=args.max_steps,
                maps_count=args.maps_count,
                with_memory=True
            )
    
    print("All training completed!")

if __name__ == "__main__":
    main() 