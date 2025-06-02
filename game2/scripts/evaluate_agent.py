#!/usr/bin/env python
"""
Evaluation script for Match-3 game agents.
Evaluates agent performance across difficulty levels.
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
    parser = argparse.ArgumentParser(description="Evaluate agent performance on Match-3 game")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model to use for evaluation")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"], 
                        help="Difficulty level to evaluate")
    parser.add_argument("--levels_count", type=int, default=10, help="Number of levels to evaluate")
    parser.add_argument("--levels", type=str, help="Specific levels to evaluate (comma-separated)")
    parser.add_argument("--with_truth", action="store_true", help="Use truth knowledge")
    parser.add_argument("--truth_path", type=str, help="Path to truth knowledge file")
    parser.add_argument("--api_key", type=str, help="API key for GPT")
    parser.add_argument("--base_url", type=str, help="Base URL for API")
    parser.add_argument("--results_dir", type=str, help="Directory to store results")
    parser.add_argument("--maps_dir", type=str, help="Directory containing game maps")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()

def check_if_evaluated(model_name, difficulty, level, truth_used=False, results_dir=None):
    """
    Check if a specific level has already been evaluated
    
    Args:
        model_name: Name of the model
        difficulty: Difficulty level
        level: Level number
        truth_used: Whether truth knowledge was used
        results_dir: Results directory
    
    Returns:
        bool: True if level has been evaluated
    """
    # Determine path structure
    if results_dir:
        model_suffix = f"{model_name}-truth" if truth_used else model_name
        level_dir = os.path.join(
            results_dir,
            model_suffix,
            "game2",
            difficulty,
            f"level{level:02d}",
            "agent"
        )
    else:
        # Use default path structure
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_suffix = f"{model_name}-truth" if truth_used else model_name
        level_dir = os.path.join(
            project_root,
            "results",
            model_suffix,
            "game2",
            difficulty,
            f"level{level:02d}",
            "agent"
        )
    
    # Check for stats files
    if not os.path.exists(level_dir):
        return False
    
    stats_files = [f for f in os.listdir(level_dir) if f.startswith("stats_") and f.endswith(".json")]
    if not stats_files:
        return False
    
    # Check if any stats file has complete data
    for stats_file in stats_files:
        try:
            with open(os.path.join(level_dir, stats_file), 'r') as f:
                stats = json.load(f)
                # Check for key fields that indicate completion
                if all(field in stats for field in ["total_score", "cleared", "steps_remaining"]):
                    return True
        except:
            continue
    
    return False

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # Initialize GPT client
    api_client = GPTClient(
        key=args.api_key,
        url=args.base_url,
        model=args.model
    )
    
    # Load truth knowledge if specified
    truth_knowledge = None
    if args.with_truth and args.truth_path:
        try:
            with open(args.truth_path, 'r') as f:
                truth_knowledge = json.load(f)
            print(f"Loaded {len(truth_knowledge)} truth knowledge items")
        except Exception as e:
            print(f"Failed to load truth knowledge: {str(e)}")
    
    # Initialize learning agent
    agent = MatchGameLearningAgent(
        api_client=api_client,
        memory_dir=os.path.join(args.results_dir, "memory") if args.results_dir else None,
        results_dir=args.results_dir
    )
    
    # Determine levels to evaluate
    if args.levels:
        levels = [int(level) for level in args.levels.split(",")]
    else:
        levels = list(range(1, args.levels_count + 1))
    
    # Filter out levels that have already been evaluated
    if not args.with_truth:  # Only skip for baseline runs
        levels_to_evaluate = []
        for level in levels:
            if not check_if_evaluated(args.model, args.difficulty, level, False, args.results_dir):
                levels_to_evaluate.append(level)
        
        if len(levels_to_evaluate) < len(levels):
            print(f"Skipping {len(levels) - len(levels_to_evaluate)} already evaluated levels")
            levels = levels_to_evaluate
    
    if not levels:
        print(f"All levels already evaluated for {args.model} on {args.difficulty}")
        return
    
    # Initialize game environment
    game = MatchGame(
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        truth_knowledge_path=args.truth_path if args.with_truth else None,
        results_dir=args.results_dir,
        maps_dir=args.maps_dir
    )
    
    # Evaluate each level
    for level in levels:
        print(f"\n{'='*80}")
        print(f"Evaluating {args.difficulty} level {level} with model {args.model}")
        print(f"{'='*80}")
        
        # Initialize the game
        game.init_game(args.difficulty, level)
        
        # Start agent session
        agent.start_session(args.difficulty, level)
        
        # Run game until completion
        while not game.game_over:
            # Get current state
            state = game.get_state_dict()
            
            # Get agent action
            action, response = agent.get_action(state, with_memory=args.with_truth)
            
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
            
            print(f"Action: {action_type} - pos={pos}, index={index}")
            
            # Track pre-action state
            pre_score = game.score
            pre_cleared = {k: v for k, v in game.color_counts.items()}
            
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
        
        # End agent session
        agent.end_session(
            score=game.score,
            color_counts=game.color_counts,
            cleared=all(game.color_counts.get(color, 0) >= game.color_targets.get(color, 0) for color in ["A", "B", "C", "D"]),
            steps_remaining=game.steps_remaining
        )
        
        # Generate reflection if successful
        if args.with_truth and all(game.color_counts.get(color, 0) >= game.color_targets.get(color, 0) for color in ["A", "B", "C", "D"]):
            print("Level completed successfully, generating reflection...")
            try:
                # Generate reflection
                experience_summary, strengths, weaknesses = agent.reflect_on_session()
                
                # Record subjective memory
                agent.record_subjective_memory(experience_summary, strengths, weaknesses)
                
                print("Reflection and subjective memory recorded")
            except Exception as e:
                print(f"Error during reflection: {str(e)}")
        
        print(f"Evaluation complete for {args.difficulty} level {level}")
        print(f"Score: {game.score}, Steps remaining: {game.steps_remaining}")
        print(f"Color counts: {game.color_counts}")
        print(f"Color targets: {game.color_targets}")
        
        # Clear state before next level
        game = MatchGame(
            model_name=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            truth_knowledge_path=args.truth_path if args.with_truth else None,
            results_dir=args.results_dir,
            maps_dir=args.maps_dir
        )
    
    print("\nEvaluation completed for all levels")

if __name__ == "__main__":
    main() 