#!/usr/bin/env python
"""
Analysis script for Match-3 game results.
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze Match-3 game results")
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
    model_dir = os.path.join(results_dir, model_suffix, "game2")
    
    if not os.path.exists(model_dir):
        print(f"No results found for model {model_suffix}")
        return {}
    
    # Structure to hold results
    results = {
        "easy": {},
        "medium": {},
        "hard": {}
    }
    
    # Load stats for each difficulty and level
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_dir = os.path.join(model_dir, difficulty)
        if not os.path.exists(difficulty_dir):
            continue
        
        # Find all level directories
        level_dirs = glob.glob(os.path.join(difficulty_dir, "level*", "agent"))
        
        for level_dir in level_dirs:
            # Extract level number
            level_match = os.path.basename(os.path.dirname(level_dir))
            if not level_match.startswith("level"):
                continue
            
            level_num = int(level_match[5:])
            
            # Find the latest stats file
            stats_files = glob.glob(os.path.join(level_dir, "stats_*.json"))
            if not stats_files:
                continue
            
            latest_stats_file = max(stats_files, key=os.path.getmtime)
            
            # Load stats
            try:
                with open(latest_stats_file, 'r') as f:
                    stats = json.load(f)
                
                # Store stats
                results[difficulty][level_num] = stats
            except Exception as e:
                print(f"Error loading stats for {difficulty} level {level_num}: {str(e)}")
    
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
        "easy": {
            "levels_completed": 0,
            "total_levels": 0,
            "avg_score": 0,
            "avg_steps_remaining": 0,
            "avg_clear_per_step": 0,
            "completion_rate": 0
        },
        "medium": {
            "levels_completed": 0,
            "total_levels": 0,
            "avg_score": 0,
            "avg_steps_remaining": 0,
            "avg_clear_per_step": 0,
            "completion_rate": 0
        },
        "hard": {
            "levels_completed": 0,
            "total_levels": 0,
            "avg_score": 0,
            "avg_steps_remaining": 0,
            "avg_clear_per_step": 0,
            "completion_rate": 0
        },
        "overall": {
            "levels_completed": 0,
            "total_levels": 0,
            "avg_score": 0,
            "avg_steps_remaining": 0,
            "avg_clear_per_step": 0,
            "completion_rate": 0
        }
    }
    
    # Calculate statistics for each difficulty
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_results = results.get(difficulty, {})
        
        total_levels = len(difficulty_results)
        if total_levels == 0:
            continue
        
        stats[difficulty]["total_levels"] = total_levels
        
        # Accumulate values
        completed_levels = 0
        total_score = 0
        total_steps_remaining = 0
        total_clear_per_step = 0
        
        for level, level_stats in difficulty_results.items():
            if level_stats.get("cleared", False):
                completed_levels += 1
            
            total_score += level_stats.get("total_score", 0)
            total_steps_remaining += level_stats.get("steps_remaining", 0)
            total_clear_per_step += level_stats.get("avg_clear_per_step", 0)
        
        stats[difficulty]["levels_completed"] = completed_levels
        stats[difficulty]["completion_rate"] = completed_levels / total_levels if total_levels > 0 else 0
        stats[difficulty]["avg_score"] = total_score / total_levels if total_levels > 0 else 0
        stats[difficulty]["avg_steps_remaining"] = total_steps_remaining / total_levels if total_levels > 0 else 0
        stats[difficulty]["avg_clear_per_step"] = total_clear_per_step / total_levels if total_levels > 0 else 0
    
    # Calculate overall statistics
    total_levels = sum(stats[difficulty]["total_levels"] for difficulty in ["easy", "medium", "hard"])
    if total_levels > 0:
        stats["overall"]["total_levels"] = total_levels
        stats["overall"]["levels_completed"] = sum(stats[difficulty]["levels_completed"] for difficulty in ["easy", "medium", "hard"])
        stats["overall"]["completion_rate"] = stats["overall"]["levels_completed"] / total_levels
        
        # Weighted averages based on number of levels
        stats["overall"]["avg_score"] = sum(stats[difficulty]["avg_score"] * stats[difficulty]["total_levels"] 
                                         for difficulty in ["easy", "medium", "hard"]) / total_levels
        
        stats["overall"]["avg_steps_remaining"] = sum(stats[difficulty]["avg_steps_remaining"] * stats[difficulty]["total_levels"] 
                                                  for difficulty in ["easy", "medium", "hard"]) / total_levels
        
        stats["overall"]["avg_clear_per_step"] = sum(stats[difficulty]["avg_clear_per_step"] * stats[difficulty]["total_levels"] 
                                                 for difficulty in ["easy", "medium", "hard"]) / total_levels
    
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
    
    # 1. Completion Rate by Difficulty
    difficulties = ["easy", "medium", "hard"]
    completion_rates = [stats[difficulty]["completion_rate"] * 100 for difficulty in difficulties]
    
    plt.figure(figsize=(10, 6))
    plt.bar(difficulties, completion_rates, color=['green', 'orange', 'red'])
    plt.title(f'Completion Rate by Difficulty - {model_name}')
    plt.ylabel('Completion Rate (%)')
    plt.xlabel('Difficulty')
    plt.ylim(0, 100)
    
    for i, rate in enumerate(completion_rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_completion_rate.png'))
    plt.close()
    
    # 2. Average Score by Difficulty
    avg_scores = [stats[difficulty]["avg_score"] for difficulty in difficulties]
    
    plt.figure(figsize=(10, 6))
    plt.bar(difficulties, avg_scores, color=['green', 'orange', 'red'])
    plt.title(f'Average Score by Difficulty - {model_name}')
    plt.ylabel('Average Score')
    plt.xlabel('Difficulty')
    
    for i, score in enumerate(avg_scores):
        plt.text(i, score + 100, f'{score:.0f}', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_avg_score.png'))
    plt.close()
    
    # 3. Average Clear Per Step by Difficulty
    avg_clear = [stats[difficulty]["avg_clear_per_step"] for difficulty in difficulties]
    
    plt.figure(figsize=(10, 6))
    plt.bar(difficulties, avg_clear, color=['green', 'orange', 'red'])
    plt.title(f'Average Clear Per Step by Difficulty - {model_name}')
    plt.ylabel('Average Clear Per Step')
    plt.xlabel('Difficulty')
    
    for i, clear in enumerate(avg_clear):
        plt.text(i, clear + 0.1, f'{clear:.2f}', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_avg_clear.png'))
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
    
    # Load results for each model
    all_results = {}
    all_stats = {}
    
    for model in models:
        # Load baseline results
        baseline_results = load_results(model, results_dir, with_truth=False)
        baseline_stats = calculate_statistics(baseline_results)
        
        # Load truth results
        truth_results = load_results(model, results_dir, with_truth=True)
        truth_stats = calculate_statistics(truth_results)
        
        all_results[f"{model}"] = baseline_results
        all_results[f"{model}-truth"] = truth_results
        
        all_stats[f"{model}"] = baseline_stats
        all_stats[f"{model}-truth"] = truth_stats
    
    # 1. Completion Rate Comparison
    difficulties = ["easy", "medium", "hard", "overall"]
    models_with_truth = []
    
    for model in models:
        models_with_truth.append(f"{model}")
        models_with_truth.append(f"{model}-truth")
    
    # Create dataframe for plotting
    completion_data = []
    
    for model_name in models_with_truth:
        for difficulty in difficulties:
            completion_data.append({
                "Model": model_name,
                "Difficulty": difficulty.capitalize(),
                "Completion Rate": all_stats[model_name][difficulty]["completion_rate"] * 100
            })
    
    df = pd.DataFrame(completion_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart
    ax = plt.subplot(111)
    
    # Set width of bars
    bar_width = 0.15
    
    # Set position of bars on X axis
    r = np.arange(len(difficulties))
    
    # Plot bars
    for i, model in enumerate(models_with_truth):
        position = r + bar_width * (i - len(models_with_truth)/2 + 0.5)
        ax.bar(position, 
               df[(df['Model'] == model)]['Completion Rate'], 
               width=bar_width, 
               label=model)
    
    # Add labels and title
    ax.set_ylabel('Completion Rate (%)')
    ax.set_xlabel('Difficulty')
    ax.set_title('Completion Rate Comparison')
    ax.set_xticks(r)
    ax.set_xticklabels(difficulties)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, 'completion_rate_comparison.png'))
    plt.close()
    
    # 2. Average Score Comparison
    score_data = []
    
    for model_name in models_with_truth:
        for difficulty in difficulties:
            score_data.append({
                "Model": model_name,
                "Difficulty": difficulty.capitalize(),
                "Average Score": all_stats[model_name][difficulty]["avg_score"]
            })
    
    df = pd.DataFrame(score_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart
    ax = plt.subplot(111)
    
    # Plot bars
    for i, model in enumerate(models_with_truth):
        position = r + bar_width * (i - len(models_with_truth)/2 + 0.5)
        ax.bar(position, 
               df[(df['Model'] == model)]['Average Score'], 
               width=bar_width, 
               label=model)
    
    # Add labels and title
    ax.set_ylabel('Average Score')
    ax.set_xlabel('Difficulty')
    ax.set_title('Average Score Comparison')
    ax.set_xticks(r)
    ax.set_xticklabels(difficulties)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, 'avg_score_comparison.png'))
    plt.close()
    
    # 3. Average Clear Per Step Comparison
    clear_data = []
    
    for model_name in models_with_truth:
        for difficulty in difficulties:
            clear_data.append({
                "Model": model_name,
                "Difficulty": difficulty.capitalize(),
                "Average Clear Per Step": all_stats[model_name][difficulty]["avg_clear_per_step"]
            })
    
    df = pd.DataFrame(clear_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart
    ax = plt.subplot(111)
    
    # Plot bars
    for i, model in enumerate(models_with_truth):
        position = r + bar_width * (i - len(models_with_truth)/2 + 0.5)
        ax.bar(position, 
               df[(df['Model'] == model)]['Average Clear Per Step'], 
               width=bar_width, 
               label=model)
    
    # Add labels and title
    ax.set_ylabel('Average Clear Per Step')
    ax.set_xlabel('Difficulty')
    ax.set_title('Average Clear Per Step Comparison')
    ax.set_xticks(r)
    ax.set_xticklabels(difficulties)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, 'avg_clear_comparison.png'))
    plt.close()
    
    # Generate a summary report
    generate_comparison_report(all_stats, models, output_dir)

def generate_comparison_report(all_stats, models, output_dir):
    """
    Generate a comparison report
    
    Args:
        all_stats: Statistics for all models
        models: List of model names
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Match-3 Game Performance Comparison\n\n")
        
        f.write("## Overall Performance\n\n")
        
        # Create overall performance table
        f.write("| Model | Completion Rate | Average Score | Avg Steps Remaining | Avg Clear Per Step |\n")
        f.write("|-------|----------------|--------------|---------------------|--------------------|\n")
        
        for model in models:
            baseline_stats = all_stats[f"{model}"]["overall"]
            truth_stats = all_stats[f"{model}-truth"]["overall"]
            
            f.write(f"| {model} | {baseline_stats['completion_rate']*100:.1f}% | {baseline_stats['avg_score']:.0f} | {baseline_stats['avg_steps_remaining']:.1f} | {baseline_stats['avg_clear_per_step']:.2f} |\n")
            f.write(f"| {model}-truth | {truth_stats['completion_rate']*100:.1f}% | {truth_stats['avg_score']:.0f} | {truth_stats['avg_steps_remaining']:.1f} | {truth_stats['avg_clear_per_step']:.2f} |\n")
        
        # Add improvement section
        f.write("\n## Improvement with Truth Knowledge\n\n")
        
        f.write("| Model | Completion Rate Improvement | Score Improvement | Clear Per Step Improvement |\n")
        f.write("|-------|---------------------------|-------------------|---------------------------|\n")
        
        for model in models:
            baseline_stats = all_stats[f"{model}"]["overall"]
            truth_stats = all_stats[f"{model}-truth"]["overall"]
            
            completion_improvement = (truth_stats['completion_rate'] - baseline_stats['completion_rate']) * 100
            score_improvement = (truth_stats['avg_score'] - baseline_stats['avg_score'])
            clear_improvement = (truth_stats['avg_clear_per_step'] - baseline_stats['avg_clear_per_step'])
            
            f.write(f"| {model} | {completion_improvement:+.1f}% | {score_improvement:+.0f} | {clear_improvement:+.2f} |\n")
        
        # Add difficulty breakdown
        for difficulty in ["easy", "medium", "hard"]:
            f.write(f"\n## {difficulty.capitalize()} Difficulty Performance\n\n")
            
            f.write("| Model | Completion Rate | Average Score | Avg Steps Remaining | Avg Clear Per Step |\n")
            f.write("|-------|----------------|--------------|---------------------|--------------------|\n")
            
            for model in models:
                baseline_stats = all_stats[f"{model}"][difficulty]
                truth_stats = all_stats[f"{model}-truth"][difficulty]
                
                f.write(f"| {model} | {baseline_stats['completion_rate']*100:.1f}% | {baseline_stats['avg_score']:.0f} | {baseline_stats['avg_steps_remaining']:.1f} | {baseline_stats['avg_clear_per_step']:.2f} |\n")
                f.write(f"| {model}-truth | {truth_stats['completion_rate']*100:.1f}% | {truth_stats['avg_score']:.0f} | {truth_stats['avg_steps_remaining']:.1f} | {truth_stats['avg_clear_per_step']:.2f} |\n")

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
    
    report_path = os.path.join(output_dir, f'{model_name}_report.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# Match-3 Game Performance Report - {model_name}\n\n")
        
        f.write("## Overall Performance\n\n")
        
        overall_stats = stats["overall"]
        
        f.write(f"- **Total Levels Evaluated**: {overall_stats['total_levels']}\n")
        f.write(f"- **Levels Completed**: {overall_stats['levels_completed']} ({overall_stats['completion_rate']*100:.1f}%)\n")
        f.write(f"- **Average Score**: {overall_stats['avg_score']:.0f}\n")
        f.write(f"- **Average Steps Remaining**: {overall_stats['avg_steps_remaining']:.1f}\n")
        f.write(f"- **Average Clear Per Step**: {overall_stats['avg_clear_per_step']:.2f}\n\n")
        
        f.write("## Performance by Difficulty\n\n")
        
        f.write("| Difficulty | Levels | Completed | Completion Rate | Avg Score | Avg Steps Remaining | Avg Clear Per Step |\n")
        f.write("|------------|--------|-----------|----------------|-----------|---------------------|--------------------|\n")
        
        for difficulty in ["easy", "medium", "hard"]:
            diff_stats = stats[difficulty]
            if diff_stats["total_levels"] > 0:
                f.write(f"| {difficulty.capitalize()} | {diff_stats['total_levels']} | {diff_stats['levels_completed']} | {diff_stats['completion_rate']*100:.1f}% | {diff_stats['avg_score']:.0f} | {diff_stats['avg_steps_remaining']:.1f} | {diff_stats['avg_clear_per_step']:.2f} |\n")
        
        f.write("\n")
        f.write("## Analysis\n\n")
        
        # Add some analysis text
        f.write("### Key Observations\n\n")
        
        # Analyze completion rates
        if stats["easy"]["total_levels"] > 0 and stats["hard"]["total_levels"] > 0:
            easy_completion = stats["easy"]["completion_rate"]
            hard_completion = stats["hard"]["completion_rate"]
            
            if easy_completion > hard_completion:
                f.write(f"- Completion rate decreases as difficulty increases, as expected (easy: {easy_completion*100:.1f}%, hard: {hard_completion*100:.1f}%)\n")
            elif easy_completion < hard_completion:
                f.write(f"- Unexpected trend: Completion rate is higher for harder levels (easy: {easy_completion*100:.1f}%, hard: {hard_completion*100:.1f}%)\n")
            else:
                f.write(f"- Completion rate is consistent across difficulty levels ({easy_completion*100:.1f}%)\n")
        
        # Analyze efficiency
        if overall_stats["avg_clear_per_step"] > 3:
            f.write(f"- Agent shows good efficiency with {overall_stats['avg_clear_per_step']:.2f} clears per step\n")
        elif overall_stats["avg_clear_per_step"] > 2:
            f.write(f"- Agent shows moderate efficiency with {overall_stats['avg_clear_per_step']:.2f} clears per step\n")
        else:
            f.write(f"- Agent shows low efficiency with only {overall_stats['avg_clear_per_step']:.2f} clears per step\n")
        
        # Analyze overall performance
        if overall_stats["completion_rate"] > 0.8:
            f.write(f"- Overall performance is excellent with {overall_stats['completion_rate']*100:.1f}% of levels completed\n")
        elif overall_stats["completion_rate"] > 0.5:
            f.write(f"- Overall performance is good with {overall_stats['completion_rate']*100:.1f}% of levels completed\n")
        else:
            f.write(f"- Overall performance needs improvement with only {overall_stats['completion_rate']*100:.1f}% of levels completed\n")

def main():
    """Main analysis function"""
    args = parse_args()
    
    # Set default output directory if not specified
    if not args.output_dir:
        if args.results_dir:
            args.output_dir = os.path.join(args.results_dir, "analysis")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            args.output_dir = os.path.join(project_root, "results", args.model, "game2", "analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare models if requested
    if args.compare_models:
        models = args.compare_models.split(",")
        compare_models(models, args.results_dir, args.output_dir)
        print(f"Model comparison complete, output saved to {args.output_dir}")
        return
    
    # Analyze baseline results
    print(f"Analyzing baseline results for model {args.model}...")
    baseline_results = load_results(args.model, args.results_dir, with_truth=False)
    baseline_stats = calculate_statistics(baseline_results)
    
    # Generate baseline report and plots
    generate_report(args.model, baseline_stats, args.output_dir)
    generate_plots(baseline_stats, args.model, args.output_dir)
    
    # Analyze truth knowledge results
    print(f"Analyzing truth knowledge results for model {args.model}...")
    truth_results = load_results(args.model, args.results_dir, with_truth=True)
    truth_stats = calculate_statistics(truth_results)
    
    # Generate truth knowledge report and plots
    generate_report(f"{args.model}-truth", truth_stats, args.output_dir)
    generate_plots(truth_stats, f"{args.model}-truth", args.output_dir)
    
    # Compare baseline and truth knowledge
    models = [args.model]
    compare_models(models, args.results_dir, args.output_dir)
    
    print(f"Analysis complete, output saved to {args.output_dir}")

if __name__ == "__main__":
    main() 