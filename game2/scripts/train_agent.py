#!/usr/bin/env python
"""
Training script for Match-3 game agents.
Trains agent and generates truth knowledge.
"""

import os
import sys
import json
import time
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple

from nips2025.common.gpt_client import GPTClient
from nips2025.game2.agent.learning_agent import MatchGameLearningAgent
from nips2025.game2.game.match_game import MatchGame

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train agent on Match-3 game")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model to use for agent")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"], 
                        help="Difficulty level to train on")
    parser.add_argument("--levels_count", type=int, default=5, help="Number of levels to train on")
    parser.add_argument("--levels", type=str, help="Specific levels to train on (comma-separated)")
    parser.add_argument("--api_key", type=str, help="API key for GPT")
    parser.add_argument("--base_url", type=str, help="Base URL for API")
    parser.add_argument("--memory_dir", type=str, help="Directory to store memory")
    parser.add_argument("--results_dir", type=str, help="Directory to store results")
    parser.add_argument("--maps_dir", type=str, help="Directory containing game maps")
    parser.add_argument("--validation_runs", type=int, default=2, help="Number of validation runs")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()

def train_level(agent, difficulty, level, args, validation_runs=2):
    """
    Train agent on a specific level
    
    Args:
        agent: Learning agent
        difficulty: Difficulty level
        level: Level number
        args: Command line arguments
        validation_runs: Number of validation runs
        
    Returns:
        bool: Whether training was successful
    """
    print(f"\n{'='*80}")
    print(f"Training on {difficulty} level {level}")
    print(f"{'='*80}")
    
    # Step 1: First run - without memory
    print("\nInitial run (baseline)...")
    
    # Initialize game
    game = MatchGame(
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        results_dir=args.results_dir,
        maps_dir=args.maps_dir
    )
    
    game.init_game(difficulty, level)
    
    # Start agent session
    agent.start_session(difficulty, level)
    
    # Run game until completion
    while not game.game_over:
        # Get current state
        state = game.get_state_dict()
        
        # Get agent action (without memory)
        action, response = agent.get_action(state, with_memory=False)
        
        if not action:
            print("Failed to get valid action, using random action")
            # Use a fallback action if agent fails
            action = {
                "action": {
                    "type": "eliminate",
                    "pos": [random.randint(0, 7), random.randint(0, 7)]
                }
            }
        
        # Execute action
        action_type = action.get("action", {}).get("type")
        pos = action.get("action", {}).get("pos")
        index = action.get("action", {}).get("index")
        
        # Track pre-action state
        pre_score = game.score
        
        # Execute action
        cleared_count = 0
        reward = 0
        
        if action_type == "eliminate" and pos:
            # Find connected region
            i, j = pos
            if 0 <= i < 8 and 0 <= j < 8 and game.board[i][j]:
                visited = [[False] * 8 for _ in range(8)]
                region = game.find_connected_region(i, j, game.board[i][j], visited, game.board)
                if len(region) >= 2:
                    if game.clear_region(region):
                        cleared_count = len(region)
        elif action_type in ["row", "col", "bomb", "hammer"]:
            target = index if action_type in ["row", "col"] else pos
            if game.use_prop(action_type, target):
                cleared_count = 1  # Simplified
        
        # Calculate reward (score change)
        reward = game.score - pre_score
        
        # Log interaction
        agent.log_interaction(state, action, response, reward, cleared_count)
        
        # Sleep to avoid API rate limits
        time.sleep(0.5)
    
    # End agent session and save baseline metrics
    baseline_metrics = agent.end_session(
        score=game.score,
        color_counts=game.color_counts,
        cleared=all(game.color_counts.get(color, 0) >= game.color_targets.get(color, 0) for color in ["A", "B", "C", "D"]),
        steps_remaining=game.steps_remaining
    )
    
    print(f"Baseline run complete: score={game.score}, cleared={baseline_metrics['cleared']}")
    
    # Generate reflection
    experience_summary, strengths, weaknesses = agent.reflect_on_session()
    
    # Record subjective memory
    agent.record_subjective_memory(experience_summary, strengths, weaknesses)
    
    # Step 2: Validation runs - with memory
    is_valid = False
    best_metrics = None
    
    for i in range(validation_runs):
        print(f"\nValidation run {i+1}/{validation_runs}...")
        
        # Reset game
        game = MatchGame(
            model_name=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            results_dir=args.results_dir,
            maps_dir=args.maps_dir
        )
        
        game.init_game(difficulty, level)
        
        # Start agent session
        agent.start_session(difficulty, level)
        
        # Run game until completion
        while not game.game_over:
            # Get current state
            state = game.get_state_dict()
            
            # Get agent action (with memory)
            action, response = agent.get_action(state, with_memory=True)
            
            if not action:
                print("Failed to get valid action, using random action")
                # Use a fallback action if agent fails
                action = {
                    "action": {
                        "type": "eliminate",
                        "pos": [random.randint(0, 7), random.randint(0, 7)]
                    }
                }
            
            # Execute action
            action_type = action.get("action", {}).get("type")
            pos = action.get("action", {}).get("pos")
            index = action.get("action", {}).get("index")
            
            # Track pre-action state
            pre_score = game.score
            
            # Execute action
            cleared_count = 0
            reward = 0
            
            if action_type == "eliminate" and pos:
                # Find connected region
                i, j = pos
                if 0 <= i < 8 and 0 <= j < 8 and game.board[i][j]:
                    visited = [[False] * 8 for _ in range(8)]
                    region = game.find_connected_region(i, j, game.board[i][j], visited, game.board)
                    if len(region) >= 2:
                        if game.clear_region(region):
                            cleared_count = len(region)
            elif action_type in ["row", "col", "bomb", "hammer"]:
                target = index if action_type in ["row", "col"] else pos
                if game.use_prop(action_type, target):
                    cleared_count = 1  # Simplified
            
            # Calculate reward (score change)
            reward = game.score - pre_score
            
            # Log interaction
            agent.log_interaction(state, action, response, reward, cleared_count)
            
            # Sleep to avoid API rate limits
            time.sleep(0.5)
        
        # End agent session and get metrics
        validation_metrics = agent.end_session(
            score=game.score,
            color_counts=game.color_counts,
            cleared=all(game.color_counts.get(color, 0) >= game.color_targets.get(color, 0) for color in ["A", "B", "C", "D"]),
            steps_remaining=game.steps_remaining
        )
        
        print(f"Validation run {i+1} complete: score={game.score}, cleared={validation_metrics['cleared']}")
        
        # Check if validation improved performance
        current_valid = agent.validate_subjective_memory(
            difficulty, level, baseline_metrics, validation_metrics
        )
        
        if current_valid:
            is_valid = True
            best_metrics = validation_metrics
            print("Memory validation successful")
            break
        
        # Generate reflection
        experience_summary, strengths, weaknesses = agent.reflect_on_session()
        
        # Update subjective memory
        agent.record_subjective_memory(experience_summary, strengths, weaknesses)
    
    # Step 3: Promote to truth if valid
    if is_valid:
        print("\nPromoting validated memory to truth knowledge...")
        agent.promote_memory_to_truth()
        print("Memory promoted to truth knowledge")
        
        # Optimize truth knowledge if needed
        agent.optimize_truth_knowledge()
        print("Truth knowledge optimized")
        
        return True
    else:
        print("\nFailed to validate memory, clearing subjective memory")
        agent.clear_current_subjective_memory()
        return False

def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # Initialize GPT client
    api_client = GPTClient(
        key=args.api_key,
        url=args.base_url,
        model=args.model
    )
    
    # Initialize memory directory
    memory_dir = args.memory_dir
    if not memory_dir:
        memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory")
    
    os.makedirs(memory_dir, exist_ok=True)
    
    # Initialize learning agent
    agent = MatchGameLearningAgent(
        api_client=api_client,
        memory_dir=memory_dir,
        results_dir=args.results_dir
    )
    
    # Determine levels to train on
    if args.levels:
        levels = [int(level) for level in args.levels.split(",")]
    else:
        levels = list(range(1, args.levels_count + 1))
    
    # Train on each level
    successful_levels = 0
    for level in levels:
        if train_level(agent, args.difficulty, level, args, args.validation_runs):
            successful_levels += 1
    
    print(f"\nTraining complete: {successful_levels}/{len(levels)} levels successfully validated")

if __name__ == "__main__":
    main() 