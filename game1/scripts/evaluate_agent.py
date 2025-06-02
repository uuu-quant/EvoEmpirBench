#!/usr/bin/env python
"""
Evaluation script for Maze Navigation game agents.
Evaluates agent performance across difficulty levels.
"""

import os
import sys
import json
import time
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple

from ...common.gpt_client import GPTClient
from ..game.environment import MazeNavigationEnv
from ..game.config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from ..agent.learning_agent import MazeNavigationAgent

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate agent performance on Maze Navigation game")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model to use for evaluation")
    parser.add_argument("--mode", type=str, default=MODE_LEVEL2, 
                        choices=[MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3], 
                        help="Game mode/difficulty to evaluate")
    parser.add_argument("--maps_count", type=int, default=10, help="Number of maps to evaluate")
    parser.add_argument("--maps", type=str, help="Specific maps to evaluate (comma-separated)")
    parser.add_argument("--with_truth", action="store_true", help="Use truth knowledge")
    parser.add_argument("--truth_path", type=str, help="Path to truth knowledge file")
    parser.add_argument("--api_key", type=str, help="API key for GPT")
    parser.add_argument("--base_url", type=str, help="Base URL for API")
    parser.add_argument("--results_dir", type=str, help="Directory to store results")
    parser.add_argument("--maps_dir", type=str, help="Directory containing game maps")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--full_vision", action="store_true", help="Enable full vision mode")
    
    return parser.parse_args()

def check_if_evaluated(model_name, mode, map_index, truth_used=False, results_dir=None):
    """
    Check if a specific map has already been evaluated
    
    Args:
        model_name: Name of the model
        mode: Game mode/difficulty
        map_index: Map index number
        truth_used: Whether truth knowledge was used
        results_dir: Results directory
    
    Returns:
        bool: True if map has been evaluated
    """
    # Determine path structure
    if results_dir:
        model_suffix = f"{model_name}-truth" if truth_used else model_name
        map_dir = os.path.join(
            results_dir,
            model_suffix,
            "game1",
            mode,
            f"map{map_index:03d}"
        )
    else:
        # Use default path structure
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_suffix = f"{model_name}-truth" if truth_used else model_name
        map_dir = os.path.join(
            project_root,
            "results",
            model_suffix,
            "game1",
            mode,
            f"map{map_index:03d}"
        )
    
    # Check for results file
    results_file = os.path.join(map_dir, "results.json")
    if not os.path.exists(results_file):
        return False
    
    # Check if results file has complete data
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
            # Check for key fields that indicate completion
            required_fields = ["success", "score", "steps_used", "exploration_rate"]
            if all(field in results for field in required_fields):
                return True
    except:
        pass
    
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
    
    # Initialize navigation agent
    agent = MazeNavigationAgent(
        api_client=api_client,
        memory_dir=os.path.join(args.results_dir, "memory") if args.results_dir else None,
        results_dir=args.results_dir,
        truth_knowledge=truth_knowledge
    )
    
    # Determine maps to evaluate
    if args.maps:
        map_indices = [int(map_idx) for map_idx in args.maps.split(",")]
    else:
        map_indices = list(range(1, args.maps_count + 1))
    
    # Filter out maps that have already been evaluated
    if not args.with_truth:  # Only skip for baseline runs
        maps_to_evaluate = []
        for map_idx in map_indices:
            if not check_if_evaluated(args.model, args.mode, map_idx, False, args.results_dir):
                maps_to_evaluate.append(map_idx)
        
        if len(maps_to_evaluate) < len(map_indices):
            print(f"Skipping {len(map_indices) - len(maps_to_evaluate)} already evaluated maps")
            map_indices = maps_to_evaluate
    
    if not map_indices:
        print(f"All maps already evaluated for {args.model} on {args.mode}")
        return
    
    # Load maps
    from ..game.map_generator import MapGenerator
    maps = []
    
    if args.maps_dir:
        maps_path = os.path.join(args.maps_dir, f"{args.mode}.json")
        if os.path.exists(maps_path):
            try:
                maps = MapGenerator.load_map_collection(maps_path)
                print(f"Loaded {len(maps)} maps from {maps_path}")
            except Exception as e:
                print(f"Failed to load maps: {str(e)}")
    
    # Generate maps if needed
    if not maps:
        print(f"Generating {args.maps_count} maps for {args.mode}")
        maps = MapGenerator.generate_maps(args.maps_count, args.mode, "")
    
    # Ensure we have enough maps
    max_map_index = max(map_indices)
    if len(maps) < max_map_index:
        print(f"Generating additional maps to reach index {max_map_index}")
        additional_maps = MapGenerator.generate_maps(max_map_index - len(maps), args.mode, "")
        maps.extend(additional_maps)
    
    # Create results directory structure
    if args.results_dir:
        model_suffix = f"{args.model}-truth" if args.with_truth else args.model
        base_results_dir = os.path.join(
            args.results_dir,
            model_suffix,
            "game1",
            args.mode
        )
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_suffix = f"{args.model}-truth" if args.with_truth else args.model
        base_results_dir = os.path.join(
            project_root,
            "results",
            model_suffix,
            "game1",
            args.mode
        )
    
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Initialize environment
    env = MazeNavigationEnv(mode=args.mode, full_vision=args.full_vision)
    
    # Evaluate each map
    for map_idx in map_indices:
        if map_idx > len(maps):
            print(f"Map index {map_idx} exceeds available maps")
            continue
        
        map_data = maps[map_idx - 1]
        
        print(f"\n{'='*80}")
        print(f"Evaluating {args.mode} map {map_idx} with model {args.model}")
        print(f"{'='*80}")
        
        # Create map-specific results directory
        map_results_dir = os.path.join(base_results_dir, f"map{map_idx:03d}")
        os.makedirs(map_results_dir, exist_ok=True)
        
        # Load map into environment
        env.load_map(map_data, map_idx)
        
        # Start agent session
        agent.start_session(args.mode, map_idx)
        
        # Run episode
        state, info = env.reset()
        total_reward = 0
        done = False
        terminated = False
        steps = 0
        max_steps = 200  # Prevent infinite loops
        
        # Store initial state
        agent.observe(state, info)
        
        while not (done or terminated) and steps < max_steps:
            # Get agent action
            action, response = agent.get_action(state, info, with_memory=args.with_truth)
            
            if action is None:
                print("Failed to get valid action, using random action")
                action = random.randint(0, env.action_space.n - 1)
            
            # Execute action
            next_state, reward, terminated, truncated, next_info = env.step(action)
            
            # Update done flag
            done = terminated or truncated
            
            # Log interaction
            agent.log_interaction(state, action, response, reward, next_info)
            
            # Update state and info
            state = next_state
            info = next_info
            
            # Accumulate reward
            total_reward += reward
            
            # Increment step counter
            steps += 1
            
            # Sleep to avoid API rate limits
            time.sleep(0.5)
        
        # Calculate exploration rate
        exploration_rate = env._compute_exploration_rate() if hasattr(env, '_compute_exploration_rate') else 0.0
        
        # Get final stats
        stats = env.end_session(done, info.get("goal_reached", False))
        stats.update({
            "reward": total_reward,
            "steps_used": steps,
            "exploration_rate": exploration_rate,
            "max_steps_reached": steps >= max_steps
        })
        
        # Save results
        results_file = os.path.join(map_results_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # End agent session
        agent.end_session(
            success=stats.get("success", False),
            score=stats.get("custom_score", 0),
            exploration_rate=exploration_rate,
            steps_used=steps
        )
        
        # Generate reflection if successful
        if args.with_truth and stats.get("success", False):
            print("Map completed successfully, generating reflection...")
            try:
                # Generate reflection
                experience_summary, strengths, weaknesses = agent.reflect_on_session()
                
                # Record subjective memory
                agent.record_subjective_memory(experience_summary, strengths, weaknesses)
                
                # Save reflection
                reflection_file = os.path.join(map_results_dir, "reflection.json")
                with open(reflection_file, 'w') as f:
                    json.dump({
                        "experience_summary": experience_summary,
                        "strengths": strengths,
                        "weaknesses": weaknesses
                    }, f, indent=2)
                
                print("Reflection and subjective memory recorded")
            except Exception as e:
                print(f"Error during reflection: {str(e)}")
        
        print(f"Evaluation complete for {args.mode} map {map_idx}")
        print(f"Score: {stats.get('custom_score', 0)}, Success: {stats.get('success', False)}")
        print(f"Exploration rate: {exploration_rate:.2f}, Steps used: {steps}")
    
    # Generate evaluation summary
    all_results = []
    for map_idx in map_indices:
        map_results_dir = os.path.join(base_results_dir, f"map{map_idx:03d}")
        results_file = os.path.join(map_results_dir, "results.json")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    results["map_index"] = map_idx
                    all_results.append(results)
            except:
                pass
    
    # Calculate overall statistics
    if all_results:
        success_count = sum(1 for r in all_results if r.get("success", False))
        success_rate = success_count / len(all_results)
        avg_score = sum(r.get("custom_score", 0) for r in all_results) / len(all_results)
        avg_exploration = sum(r.get("exploration_rate", 0) for r in all_results) / len(all_results)
        avg_steps = sum(r.get("steps_used", 0) for r in all_results) / len(all_results)
        
        summary = {
            "model": args.model,
            "mode": args.mode,
            "maps_evaluated": len(all_results),
            "success_count": success_count,
            "success_rate": success_rate,
            "average_score": avg_score,
            "average_exploration_rate": avg_exploration,
            "average_steps_used": avg_steps,
            "with_truth": args.with_truth
        }
        
        # Save summary
        summary_file = os.path.join(base_results_dir, "evaluation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nEvaluation summary:")
        print(f"Mode: {args.mode}, Maps evaluated: {len(all_results)}")
        print(f"Success rate: {success_rate:.2f} ({success_count}/{len(all_results)})")
        print(f"Average score: {avg_score:.2f}")
        print(f"Average exploration rate: {avg_exploration:.2f}")
        print(f"Average steps used: {avg_steps:.2f}")

if __name__ == "__main__":
    main() 