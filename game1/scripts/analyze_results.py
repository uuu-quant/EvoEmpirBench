#!/usr/bin/env python
"""
Analysis script for Maze Navigation game results.
Generates statistics and visualizations from evaluation results.
"""

import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict

from ..game.config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze Maze Navigation game results")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name to analyze")
    parser.add_argument("--results_dir", type=str, help="Results directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for analysis")
    parser.add_argument("--compare_models", type=str, help="Compare models (comma-separated)")
    
    return parser.parse_args()

def load_results(model_name, results_dir, with_truth=False):
    """
    Load results for a specific model
    
    Args:
        model_name: Name of the model
        results_dir: Results directory
        with_truth: Whether to load results with truth knowledge
    
    Returns:
        Dict with results
    """
    # Determine results path
    if not results_dir:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(project_root, "results")
    
    model_suffix = f"{model_name}-truth" if with_truth else model_name
    model_dir = os.path.join(results_dir, model_suffix, "game1")
    
    if not os.path.exists(model_dir):
        print(f"No results found for model {model_suffix}")
        return {}
    
    # Structure to hold results
    results = {
        MODE_LEVEL1: [],
        MODE_LEVEL2: [],
        MODE_LEVEL3: []
    }
    
    # Load results for each mode
    for mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
        mode_dir = os.path.join(model_dir, mode)
        if not os.path.exists(mode_dir):
            continue
        
        # Find all map directories
        map_dirs = glob.glob(os.path.join(mode_dir, "map*"))
        
        for map_dir in map_dirs:
            # Extract map number
            map_name = os.path.basename(map_dir)
            map_number = int(map_name.replace("map", ""))
            
            # Load results
            results_file = os.path.join(map_dir, "results.json")
            if not os.path.exists(results_file):
                continue
            
            try:
                with open(results_file, 'r') as f:
                    map_results = json.load(f)
                
                # Add map number
                map_results["map_number"] = map_number
                
                # Add to results list
                results[mode].append(map_results)
            except Exception as e:
                print(f"Error loading results for {mode} map {map_number}: {str(e)}")
    
    return results

def calculate_statistics(results):
    """
    Calculate overall statistics from results
    
    Args:
        results: Results dictionary
    
    Returns:
        Dict with statistics
    """
    stats = {
        MODE_LEVEL1: {},
        MODE_LEVEL2: {},
        MODE_LEVEL3: {},
        "overall": {}
    }
    
    # Calculate statistics for each mode
    total_maps = 0
    total_success = 0
    total_score = 0
    total_exploration = 0
    total_steps = 0
    total_coins = 0
    total_lives = 0
    
    # Level 3 specific stats
    total_killed_monsters = 0
    total_destroyed_obstacles = 0
    
    for mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
        mode_results = results.get(mode, [])
        
        if not mode_results:
            continue
            
        maps_count = len(mode_results)
        total_maps += maps_count
        
        # Calculate success rate
        success_count = sum(1 for r in mode_results if r.get("success", False))
        success_rate = success_count / maps_count if maps_count > 0 else 0
        total_success += success_count
        
        # Calculate average score
        avg_score = sum(r.get("custom_score", 0) for r in mode_results) / maps_count if maps_count > 0 else 0
        total_score += sum(r.get("custom_score", 0) for r in mode_results)
        
        # Calculate average exploration rate
        avg_exploration = sum(r.get("exploration_rate", 0) for r in mode_results) / maps_count if maps_count > 0 else 0
        total_exploration += sum(r.get("exploration_rate", 0) for r in mode_results)
        
        # Calculate average steps
        avg_steps = sum(r.get("steps_used", 0) for r in mode_results) / maps_count if maps_count > 0 else 0
        total_steps += sum(r.get("steps_used", 0) for r in mode_results)
        
        # Calculate average coins collected
        avg_coins = sum(r.get("collected_coins", 0) for r in mode_results) / maps_count if maps_count > 0 else 0
        coin_collection_rate = avg_coins / 5  # Assuming 5 coins per map
        total_coins += sum(r.get("collected_coins", 0) for r in mode_results)
        
        # Calculate average lives remaining
        avg_lives = sum(r.get("lives_remaining", 0) for r in mode_results) / maps_count if maps_count > 0 else 0
        total_lives += sum(r.get("lives_remaining", 0) for r in mode_results)
        
        # Level 3 specific stats
        if mode == MODE_LEVEL3:
            # Calculate average monsters killed
            avg_killed_monsters = sum(r.get("killed_monsters", 0) for r in mode_results) / maps_count if maps_count > 0 else 0
            total_killed_monsters = sum(r.get("killed_monsters", 0) for r in mode_results)
            
            # Calculate average obstacles destroyed
            avg_destroyed_obstacles = sum(r.get("destroyed_obstacles", 0) for r in mode_results) / maps_count if maps_count > 0 else 0
            total_destroyed_obstacles = sum(r.get("destroyed_obstacles", 0) for r in mode_results)
        
        # Store mode statistics
        stats[mode] = {
            "maps_count": maps_count,
            "success_count": success_count,
            "success_rate": success_rate,
            "average_score": avg_score,
            "average_exploration": avg_exploration,
            "average_steps": avg_steps,
            "average_coins": avg_coins,
            "coin_collection_rate": coin_collection_rate,
            "average_lives": avg_lives
        }
        
        # Add Level 3 specific stats
        if mode == MODE_LEVEL3:
            stats[mode]["average_killed_monsters"] = avg_killed_monsters
            stats[mode]["average_destroyed_obstacles"] = avg_destroyed_obstacles
    
    # Calculate overall statistics
    if total_maps > 0:
        stats["overall"] = {
            "maps_count": total_maps,
            "success_count": total_success,
            "success_rate": total_success / total_maps,
            "average_score": total_score / total_maps,
            "average_exploration": total_exploration / total_maps,
            "average_steps": total_steps / total_maps,
            "average_coins": total_coins / total_maps,
            "coin_collection_rate": total_coins / (total_maps * 5),  # Assuming 5 coins per map
            "average_lives": total_lives / total_maps
        }
        
        # Add Level 3 specific stats if available
        if MODE_LEVEL3 in stats and "average_killed_monsters" in stats[MODE_LEVEL3]:
            level3_maps = stats[MODE_LEVEL3]["maps_count"]
            if level3_maps > 0:
                stats["overall"]["average_killed_monsters"] = total_killed_monsters / level3_maps
                stats["overall"]["average_destroyed_obstacles"] = total_destroyed_obstacles / level3_maps
    
    return stats

def generate_plots(stats, model_name, output_dir):
    """
    Generate plots from statistics
    
    Args:
        stats: Statistics dictionary
        model_name: Model name
        output_dir: Output directory
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Success Rate by Mode
    modes = [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]
    mode_labels = ["Level 1", "Level 2", "Level 3"]
    success_rates = []
    
    for mode in modes:
        if mode in stats and "success_rate" in stats[mode]:
            success_rates.append(stats[mode]["success_rate"] * 100)
        else:
            success_rates.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(mode_labels, success_rates, color=['#4CAF50', '#2196F3', '#F44336'])
    plt.title(f'Success Rate by Level - {model_name}')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Game Level')
    plt.ylim(0, 100)
    
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_success_rate.png'))
    plt.close()
    
    # 2. Average Score by Mode
    avg_scores = []
    
    for mode in modes:
        if mode in stats and "average_score" in stats[mode]:
            avg_scores.append(stats[mode]["average_score"])
        else:
            avg_scores.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(mode_labels, avg_scores, color=['#4CAF50', '#2196F3', '#F44336'])
    plt.title(f'Average Score by Level - {model_name}')
    plt.ylabel('Average Score')
    plt.xlabel('Game Level')
    
    for i, score in enumerate(avg_scores):
        plt.text(i, score + 100, f'{score:.0f}', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_avg_score.png'))
    plt.close()
    
    # 3. Exploration Rate by Mode
    avg_exploration = []
    
    for mode in modes:
        if mode in stats and "average_exploration" in stats[mode]:
            avg_exploration.append(stats[mode]["average_exploration"] * 100)
        else:
            avg_exploration.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(mode_labels, avg_exploration, color=['#4CAF50', '#2196F3', '#F44336'])
    plt.title(f'Exploration Rate by Level - {model_name}')
    plt.ylabel('Exploration Rate (%)')
    plt.xlabel('Game Level')
    plt.ylim(0, 100)
    
    for i, rate in enumerate(avg_exploration):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_exploration_rate.png'))
    plt.close()
    
    # 4. Coin Collection Rate by Mode
    coin_rates = []
    
    for mode in modes:
        if mode in stats and "coin_collection_rate" in stats[mode]:
            coin_rates.append(stats[mode]["coin_collection_rate"] * 100)
        else:
            coin_rates.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(mode_labels, coin_rates, color=['#4CAF50', '#2196F3', '#F44336'])
    plt.title(f'Coin Collection Rate by Level - {model_name}')
    plt.ylabel('Coin Collection Rate (%)')
    plt.xlabel('Game Level')
    plt.ylim(0, 100)
    
    for i, rate in enumerate(coin_rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_coin_collection.png'))
    plt.close()
    
    # 5. Special Metrics for Level 3
    if MODE_LEVEL3 in stats and "average_killed_monsters" in stats[MODE_LEVEL3]:
        metrics = ["Killed Monsters", "Destroyed Obstacles"]
        values = [stats[MODE_LEVEL3]["average_killed_monsters"], stats[MODE_LEVEL3]["average_destroyed_obstacles"]]
        
        plt.figure(figsize=(8, 6))
        plt.bar(metrics, values, color=['#673AB7', '#FF9800'])
        plt.title(f'Level 3 Special Metrics - {model_name}')
        plt.ylabel('Average Count')
        
        for i, val in enumerate(values):
            plt.text(i, val + 0.1, f'{val:.2f}', ha='center')
        
        plt.savefig(os.path.join(output_dir, f'{model_name}_level3_metrics.png'))
        plt.close()

def compare_models(models, results_dir, output_dir):
    """
    Compare multiple models
    
    Args:
        models: List of model names
        results_dir: Results directory
        output_dir: Output directory
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load results for all models
    all_results = {}
    all_stats = {}
    
    for model in models:
        # Try both with and without truth knowledge
        standard_results = load_results(model, results_dir, with_truth=False)
        truth_results = load_results(model, results_dir, with_truth=True)
        
        # Use results with truth if available, otherwise use standard results
        if truth_results and any(truth_results.values()):
            print(f"Using truth results for {model}")
            all_results[f"{model}-truth"] = truth_results
            all_stats[f"{model}-truth"] = calculate_statistics(truth_results)
        
        if standard_results and any(standard_results.values()):
            print(f"Using standard results for {model}")
            all_results[model] = standard_results
            all_stats[model] = calculate_statistics(standard_results)
    
    if not all_stats:
        print("No results found for any model")
        return
    
    # Generate comparison charts
    
    # 1. Success Rate Comparison
    model_names = list(all_stats.keys())
    success_rates = [stats["overall"]["success_rate"] * 100 for stats in all_stats.values()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, success_rates, color='#2196F3')
    plt.title('Success Rate Comparison')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'))
    plt.close()
    
    # 2. Average Score Comparison
    avg_scores = [stats["overall"]["average_score"] for stats in all_stats.values()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, avg_scores, color='#4CAF50')
    plt.title('Average Score Comparison')
    plt.ylabel('Average Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    
    for i, score in enumerate(avg_scores):
        plt.text(i, score + 100, f'{score:.0f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_score_comparison.png'))
    plt.close()
    
    # 3. Exploration Rate Comparison
    exploration_rates = [stats["overall"]["average_exploration"] * 100 for stats in all_stats.values()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, exploration_rates, color='#FF9800')
    plt.title('Exploration Rate Comparison')
    plt.ylabel('Exploration Rate (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    
    for i, rate in enumerate(exploration_rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exploration_rate_comparison.png'))
    plt.close()
    
    # 4. Success Rate by Level Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    modes = [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]
    mode_labels = ["Level 1", "Level 2", "Level 3"]
    x = np.arange(len(mode_labels))
    width = 0.8 / len(model_names)
    
    for i, model in enumerate(model_names):
        model_stats = all_stats[model]
        success_rates = []
        
        for mode in modes:
            if mode in model_stats and "success_rate" in model_stats[mode]:
                success_rates.append(model_stats[mode]["success_rate"] * 100)
            else:
                success_rates.append(0)
        
        ax.bar(x + (i - len(model_names)/2 + 0.5) * width, success_rates, width, label=model)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Game Level')
    ax.set_title('Success Rate by Level - Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_by_level_comparison.png'))
    plt.close()
    
    # 5. Coin Collection Rate Comparison
    coin_rates = [stats["overall"]["coin_collection_rate"] * 100 for stats in all_stats.values()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, coin_rates, color='#9C27B0')
    plt.title('Coin Collection Rate Comparison')
    plt.ylabel('Coin Collection Rate (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    
    for i, rate in enumerate(coin_rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coin_collection_comparison.png'))
    plt.close()
    
    # Generate comparison report
    generate_comparison_report(all_stats, model_names, output_dir)

def generate_comparison_report(all_stats, models, output_dir):
    """
    Generate a comparison report
    
    Args:
        all_stats: Dictionary with statistics for all models
        models: List of model names
        output_dir: Output directory
    """
    # Create comparison table
    comparison_data = []
    
    for model in models:
        if model not in all_stats:
            continue
            
        stats = all_stats[model]
        
        if "overall" not in stats:
            continue
            
        overall = stats["overall"]
        
        # Create a row for each model
        row = {
            "Model": model,
            "Success Rate": f"{overall['success_rate']*100:.1f}%",
            "Average Score": f"{overall['average_score']:.0f}",
            "Exploration Rate": f"{overall['average_exploration']*100:.1f}%",
            "Coin Collection": f"{overall['coin_collection_rate']*100:.1f}%",
            "Avg Steps": f"{overall['average_steps']:.1f}",
            "Avg Lives": f"{overall['average_lives']:.1f}"
        }
        
        # Add Level 3 specific metrics if available
        if "average_killed_monsters" in overall:
            row["Avg Monsters Killed"] = f"{overall['average_killed_monsters']:.1f}"
            row["Avg Obstacles Destroyed"] = f"{overall['average_destroyed_obstacles']:.1f}"
        
        comparison_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Save to Excel
    excel_path = os.path.join(output_dir, "model_comparison.xlsx")
    df.to_excel(excel_path, index=False)
    
    print(f"Comparison report saved to {csv_path} and {excel_path}")

def generate_report(model_name, stats, output_dir):
    """
    Generate a report for a single model
    
    Args:
        model_name: Model name
        stats: Statistics dictionary
        output_dir: Output directory
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a more detailed report for the model
    report_data = []
    
    # Add overall statistics
    if "overall" in stats:
        overall = stats["overall"]
        
        report_data.append({
            "Level": "Overall",
            "Maps Count": overall["maps_count"],
            "Success Rate": f"{overall['success_rate']*100:.1f}%",
            "Average Score": f"{overall['average_score']:.0f}",
            "Exploration Rate": f"{overall['average_exploration']*100:.1f}%",
            "Coin Collection": f"{overall['coin_collection_rate']*100:.1f}%",
            "Avg Steps": f"{overall['average_steps']:.1f}",
            "Avg Lives": f"{overall['average_lives']:.1f}"
        })
    
    # Add mode-specific statistics
    for mode in [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]:
        if mode not in stats:
            continue
            
        mode_stats = stats[mode]
        
        row = {
            "Level": mode,
            "Maps Count": mode_stats["maps_count"],
            "Success Rate": f"{mode_stats['success_rate']*100:.1f}%",
            "Average Score": f"{mode_stats['average_score']:.0f}",
            "Exploration Rate": f"{mode_stats['average_exploration']*100:.1f}%",
            "Coin Collection": f"{mode_stats['coin_collection_rate']*100:.1f}%",
            "Avg Steps": f"{mode_stats['average_steps']:.1f}",
            "Avg Lives": f"{mode_stats['average_lives']:.1f}"
        }
        
        # Add Level 3 specific metrics if available
        if mode == MODE_LEVEL3 and "average_killed_monsters" in mode_stats:
            row["Avg Monsters Killed"] = f"{mode_stats['average_killed_monsters']:.1f}"
            row["Avg Obstacles Destroyed"] = f"{mode_stats['average_destroyed_obstacles']:.1f}"
        
        report_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(report_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"{model_name}_report.csv")
    df.to_csv(csv_path, index=False)
    
    # Save to Excel
    excel_path = os.path.join(output_dir, f"{model_name}_report.xlsx")
    df.to_excel(excel_path, index=False)
    
    print(f"Report for {model_name} saved to {csv_path} and {excel_path}")

def main():
    """Main function"""
    args = parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if args.results_dir:
            output_dir = os.path.join(args.results_dir, "analysis")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, "results", "analysis")
    
    # If comparing models
    if args.compare_models:
        models = args.compare_models.split(",")
        print(f"Comparing models: {', '.join(models)}")
        compare_models(models, args.results_dir, output_dir)
        return
    
    # Analyze a single model
    model_name = args.model
    print(f"Analyzing model: {model_name}")
    
    # Try both with and without truth knowledge
    standard_results = load_results(model_name, args.results_dir, with_truth=False)
    truth_results = load_results(model_name, args.results_dir, with_truth=True)
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name)
    
    # Analyze standard results
    if standard_results and any(standard_results.values()):
        print(f"Analyzing standard results for {model_name}")
        stats = calculate_statistics(standard_results)
        generate_plots(stats, model_name, model_output_dir)
        generate_report(model_name, stats, model_output_dir)
    
    # Analyze truth results
    if truth_results and any(truth_results.values()):
        print(f"Analyzing truth results for {model_name}")
        truth_stats = calculate_statistics(truth_results)
        generate_plots(truth_stats, f"{model_name}-truth", model_output_dir)
        generate_report(f"{model_name}-truth", truth_stats, model_output_dir)
    
    if not (standard_results and any(standard_results.values())) and not (truth_results and any(truth_results.values())):
        print(f"No results found for model {model_name}")

if __name__ == "__main__":
    main() 