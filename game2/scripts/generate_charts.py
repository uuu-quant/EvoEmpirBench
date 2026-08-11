#!/usr/bin/env python
"""
Chart generation script for Match-3 game results.
Creates visualizations for paper/presentation.
"""

import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate charts from Match-3 game results")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name to analyze")
    parser.add_argument("--results_dir", type=str, help="Results directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for charts")
    parser.add_argument("--compare_models", type=str, help="Compare models (comma-separated)")
    parser.add_argument("--paper_ready", action="store_true", help="Generate paper-ready charts")
    
    return parser.parse_args()

def load_stats_from_analysis(model_name, results_dir, with_truth=False):
    """
    Load stats from analysis directory
    
    Args:
        model_name: Name of the model
        results_dir: Results directory
        with_truth: Whether to load results with truth knowledge
    
    Returns:
        Dict with statistics
    """
    # Determine analysis path
    if not results_dir:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(project_root, "results")
    
    model_suffix = f"{model_name}-truth" if with_truth else model_name
    analysis_dir = os.path.join(results_dir, model_suffix, "game2", "analysis")
    
    if not os.path.exists(analysis_dir):
        print(f"No analysis found for model {model_suffix}")
        return None
    
    # Load report file
    report_path = os.path.join(analysis_dir, f"{model_suffix}_report.md")
    if not os.path.exists(report_path):
        print(f"No report found at {report_path}")
        return None
    
    # Parse statistics from report file
    stats = {
        "easy": {},
        "medium": {},
        "hard": {},
        "overall": {}
    }
    
    try:
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Parse overall stats
        overall_section = report_content.split("## Overall Performance")[1].split("##")[0]
        
        # Extract metrics using string parsing
        levels_match = re.search(r"Total Levels Evaluated:\s*(\d+)", overall_section)
        if levels_match:
            stats["overall"]["total_levels"] = int(levels_match.group(1))
        
        completed_match = re.search(r"Levels Completed:\s*(\d+)\s*\((\d+\.\d+)%\)", overall_section)
        if completed_match:
            stats["overall"]["levels_completed"] = int(completed_match.group(1))
            stats["overall"]["completion_rate"] = float(completed_match.group(2)) / 100
        
        score_match = re.search(r"Average Score:\s*(\d+)", overall_section)
        if score_match:
            stats["overall"]["avg_score"] = float(score_match.group(1))
        
        steps_match = re.search(r"Average Steps Remaining:\s*(\d+\.\d+)", overall_section)
        if steps_match:
            stats["overall"]["avg_steps_remaining"] = float(steps_match.group(1))
        
        clear_match = re.search(r"Average Clear Per Step:\s*(\d+\.\d+)", overall_section)
        if clear_match:
            stats["overall"]["avg_clear_per_step"] = float(clear_match.group(1))
        
        # Parse difficulty stats from table
        difficulty_section = report_content.split("## Performance by Difficulty")[1].split("##")[0]
        
        # Extract table rows
        table_rows = difficulty_section.strip().split("\n")[3:]  # Skip header and separator
        
        for row in table_rows:
            cells = [cell.strip() for cell in row.split("|")[1:-1]]  # Skip first and last empty cells
            if len(cells) >= 7:
                difficulty = cells[0].lower()
                if difficulty not in stats:
                    continue
                
                stats[difficulty]["total_levels"] = int(cells[1])
                stats[difficulty]["levels_completed"] = int(cells[2])
                stats[difficulty]["completion_rate"] = float(cells[3].rstrip("%")) / 100
                stats[difficulty]["avg_score"] = float(cells[4])
                stats[difficulty]["avg_steps_remaining"] = float(cells[5])
                stats[difficulty]["avg_clear_per_step"] = float(cells[6])
        
        return stats
    
    except Exception as e:
        print(f"Error parsing report: {str(e)}")
        return None

def generate_comparison_charts(models, results_dir, output_dir, paper_ready=False):
    """
    Generate comparison charts for multiple models
    
    Args:
        models: List of model names
        results_dir: Results directory
        output_dir: Output directory
        paper_ready: Whether to generate paper-ready charts
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load stats for each model
    all_stats = {}
    
    for model in models:
        # Load baseline stats
        baseline_stats = load_stats_from_analysis(model, results_dir, with_truth=False)
        if baseline_stats:
            all_stats[f"{model}"] = baseline_stats
        
        # Load truth stats
        truth_stats = load_stats_from_analysis(model, results_dir, with_truth=True)
        if truth_stats:
            all_stats[f"{model}-truth"] = truth_stats
    
    if not all_stats:
        print("No stats could be loaded for any model")
        return
    
    # Set plot style for paper-ready charts
    if paper_ready:
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    # 1. Truth Knowledge Improvement Chart
    generate_truth_improvement_chart(all_stats, models, output_dir, paper_ready)
    
    # 2. Completion Rate Comparison Across Difficulties
    generate_completion_rate_comparison(all_stats, models, output_dir, paper_ready)
    
    # 3. Performance Metrics Comparison
    generate_metrics_comparison(all_stats, models, output_dir, paper_ready)

def generate_truth_improvement_chart(all_stats, models, output_dir, paper_ready=False):
    """
    Generate chart showing improvement with truth knowledge
    
    Args:
        all_stats: Stats for all models
        models: List of model names
        output_dir: Output directory
        paper_ready: Whether to generate paper-ready charts
    """
    # Calculate improvement percentages
    improvements = []
    
    for model in models:
        if f"{model}" in all_stats and f"{model}-truth" in all_stats:
            baseline = all_stats[f"{model}"]["overall"]
            truth = all_stats[f"{model}-truth"]["overall"]
            
            completion_improvement = (truth["completion_rate"] - baseline["completion_rate"]) * 100
            score_improvement = ((truth["avg_score"] - baseline["avg_score"]) / baseline["avg_score"]) * 100 if baseline["avg_score"] > 0 else 0
            efficiency_improvement = ((truth["avg_clear_per_step"] - baseline["avg_clear_per_step"]) / baseline["avg_clear_per_step"]) * 100 if baseline["avg_clear_per_step"] > 0 else 0
            
            improvements.append({
                "Model": model,
                "Metric": "Completion Rate",
                "Improvement (%)": completion_improvement
            })
            
            improvements.append({
                "Model": model,
                "Metric": "Average Score",
                "Improvement (%)": score_improvement
            })
            
            improvements.append({
                "Model": model,
                "Metric": "Efficiency",
                "Improvement (%)": efficiency_improvement
            })
    
    if not improvements:
        print("No improvement data available")
        return
    
    df = pd.DataFrame(improvements)
    
    plt.figure(figsize=(12, 8) if not paper_ready else (10, 6))
    
    ax = plt.subplot(111)
    
    # Use a more professional color palette for paper-ready charts
    colors = sns.color_palette("muted" if paper_ready else "bright", 3)
    
    sns.barplot(x="Model", y="Improvement (%)", hue="Metric", data=df, palette=colors)
    
    plt.title("Improvement with Truth Knowledge" if not paper_ready else "Performance Improvement with Truth Knowledge")
    plt.xlabel("Model")
    plt.ylabel("Improvement (%)")
    
    # Add data labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if not np.isnan(height):
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha="center", fontsize=10)
    
    plt.legend(title="Metric")
    
    if paper_ready:
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"truth_knowledge_improvement{'_paper' if paper_ready else ''}.png"))
    plt.savefig(os.path.join(output_dir, f"truth_knowledge_improvement{'_paper' if paper_ready else ''}.pdf"))
    plt.close()

def generate_completion_rate_comparison(all_stats, models, output_dir, paper_ready=False):
    """
    Generate completion rate comparison chart across difficulties
    
    Args:
        all_stats: Stats for all models
        models: List of model names
        output_dir: Output directory
        paper_ready: Whether to generate paper-ready charts
    """
    # Prepare data
    completion_data = []
    
    for model in models:
        if f"{model}" in all_stats:
            for difficulty in ["easy", "medium", "hard"]:
                completion_data.append({
                    "Model": f"{model} (baseline)",
                    "Difficulty": difficulty.capitalize(),
                    "Completion Rate (%)": all_stats[f"{model}"][difficulty]["completion_rate"] * 100
                })
        
        if f"{model}-truth" in all_stats:
            for difficulty in ["easy", "medium", "hard"]:
                completion_data.append({
                    "Model": f"{model} (truth)",
                    "Difficulty": difficulty.capitalize(),
                    "Completion Rate (%)": all_stats[f"{model}-truth"][difficulty]["completion_rate"] * 100
                })
    
    if not completion_data:
        print("No completion rate data available")
        return
    
    df = pd.DataFrame(completion_data)
    
    plt.figure(figsize=(12, 8) if not paper_ready else (10, 6))
    
    # Use a more readable grouped bar chart
    sns.barplot(x="Difficulty", y="Completion Rate (%)", hue="Model", data=df)
    
    plt.title("Completion Rate by Difficulty" if not paper_ready else "Level Completion Rate by Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Completion Rate (%)")
    
    # Set y-axis to start at 0 and end at 100
    plt.ylim(0, 100)
    
    # Add horizontal lines at 25%, 50%, 75% for reference
    if paper_ready:
        for y in [25, 50, 75]:
            plt.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
    
    plt.legend(title="Model")
    
    if paper_ready:
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"completion_rate_by_difficulty{'_paper' if paper_ready else ''}.png"))
    plt.savefig(os.path.join(output_dir, f"completion_rate_by_difficulty{'_paper' if paper_ready else ''}.pdf"))
    plt.close()

def generate_metrics_comparison(all_stats, models, output_dir, paper_ready=False):
    """
    Generate metrics comparison chart
    
    Args:
        all_stats: Stats for all models
        models: List of model names
        output_dir: Output directory
        paper_ready: Whether to generate paper-ready charts
    """
    # Prepare data for efficiency comparison
    efficiency_data = []
    
    for model in models:
        if f"{model}" in all_stats:
            efficiency_data.append({
                "Model": model,
                "Version": "Baseline",
                "Avg. Clear Per Step": all_stats[f"{model}"]["overall"]["avg_clear_per_step"]
            })
        
        if f"{model}-truth" in all_stats:
            efficiency_data.append({
                "Model": model,
                "Version": "With Truth Knowledge",
                "Avg. Clear Per Step": all_stats[f"{model}-truth"]["overall"]["avg_clear_per_step"]
            })
    
    if not efficiency_data:
        print("No efficiency data available")
        return
    
    df_efficiency = pd.DataFrame(efficiency_data)
    
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    sns.barplot(x="Model", y="Avg. Clear Per Step", hue="Version", data=df_efficiency, 
                palette=["#3498db", "#e74c3c"])
    
    plt.title("Agent Efficiency Comparison" if not paper_ready else "Agent Efficiency With and Without Truth Knowledge")
    plt.xlabel("Model")
    plt.ylabel("Average Clear Per Step")
    
    plt.legend(title="")
    
    if paper_ready:
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"efficiency_comparison{'_paper' if paper_ready else ''}.png"))
    plt.savefig(os.path.join(output_dir, f"efficiency_comparison{'_paper' if paper_ready else ''}.pdf"))
    plt.close()
    
    # Generate score comparison
    score_data = []
    
    for model in models:
        if f"{model}" in all_stats:
            score_data.append({
                "Model": model,
                "Version": "Baseline",
                "Average Score": all_stats[f"{model}"]["overall"]["avg_score"]
            })
        
        if f"{model}-truth" in all_stats:
            score_data.append({
                "Model": model,
                "Version": "With Truth Knowledge",
                "Average Score": all_stats[f"{model}-truth"]["overall"]["avg_score"]
            })
    
    if not score_data:
        print("No score data available")
        return
    
    df_score = pd.DataFrame(score_data)
    
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    sns.barplot(x="Model", y="Average Score", hue="Version", data=df_score, 
                palette=["#3498db", "#e74c3c"])
    
    plt.title("Average Score Comparison" if not paper_ready else "Score Comparison With and Without Truth Knowledge")
    plt.xlabel("Model")
    plt.ylabel("Average Score")
    
    plt.legend(title="")
    
    if paper_ready:
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"score_comparison{'_paper' if paper_ready else ''}.png"))
    plt.savefig(os.path.join(output_dir, f"score_comparison{'_paper' if paper_ready else ''}.pdf"))
    plt.close()

def generate_model_efficiency_chart(models, results_dir, output_dir, paper_ready=False):
    """
    Generate efficiency chart comparing all models
    
    Args:
        models: List of model names
        results_dir: Results directory
        output_dir: Output directory
        paper_ready: Whether to generate paper-ready charts
    """
    # Prepare data
    model_efficiency_data = []
    
    for model in models:
        if f"{model}" in all_stats:
            model_efficiency_data.append({
                "Model": model,
                "Version": "Baseline",
                "Avg. Clear Per Step": all_stats[f"{model}"]["overall"]["avg_clear_per_step"],
                "Completion Rate (%)": all_stats[f"{model}"]["overall"]["completion_rate"] * 100
            })
        
        if f"{model}-truth" in all_stats:
            model_efficiency_data.append({
                "Model": model,
                "Version": "With Truth",
                "Avg. Clear Per Step": all_stats[f"{model}-truth"]["overall"]["avg_clear_per_step"],
                "Completion Rate (%)": all_stats[f"{model}-truth"]["overall"]["completion_rate"] * 100
            })
    
    if not model_efficiency_data:
        print("No model efficiency data available")
        return
    
    df = pd.DataFrame(model_efficiency_data)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    scatter = sns.scatterplot(x="Avg. Clear Per Step", y="Completion Rate (%)", 
                             hue="Model", style="Version", s=150, data=df)
    
    # Add labels to points
    for i, row in df.iterrows():
        plt.text(row["Avg. Clear Per Step"]+0.05, row["Completion Rate (%)"], 
                f"{row['Model']}-{row['Version']}", fontsize=8)
    
    plt.title("Model Efficiency vs. Completion Rate" if not paper_ready else "Efficiency vs. Success Rate")
    plt.xlabel("Average Clear Per Step")
    plt.ylabel("Completion Rate (%)")
    
    # Set y-axis to start at 0 and end at 100
    plt.ylim(0, 100)
    
    if paper_ready:
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"model_efficiency_completion{'_paper' if paper_ready else ''}.png"))
    plt.savefig(os.path.join(output_dir, f"model_efficiency_completion{'_paper' if paper_ready else ''}.pdf"))
    plt.close()

def main():
    """Main function"""
    args = parse_args()
    
    # Set default output directory if not specified
    if not args.output_dir:
        if args.results_dir:
            args.output_dir = os.path.join(args.results_dir, "charts")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            args.output_dir = os.path.join(project_root, "results", args.model, "game2", "charts")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate charts
    if args.compare_models:
        models = args.compare_models.split(",")
        generate_comparison_charts(models, args.results_dir, args.output_dir, args.paper_ready)
    else:
        models = [args.model]
        generate_comparison_charts(models, args.results_dir, args.output_dir, args.paper_ready)
    
    print(f"Charts generated in {args.output_dir}")

if __name__ == "__main__":
    import re  # Import needed for regex in functions
    main() 