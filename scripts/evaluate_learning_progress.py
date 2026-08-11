#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse
import matplotlib as mpl

# Set English font for plots
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = True

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from src.config.paths import AGENT_MEMORY_DIR, AGENT_SESSIONS_DIR, RESULTS_DIR

def load_session_data(agent_sessions_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all session data, grouped by level ID
    
    Args:
        agent_sessions_dir: Session data directory
        
    Returns:
        Dictionary of session data grouped by level ID
    """
    session_data = {}
    
    # Traverse session directory
    level_dirs = [d for d in os.listdir(agent_sessions_dir) 
                 if os.path.isdir(os.path.join(agent_sessions_dir, d))]
    
    for level_id in level_dirs:
        level_dir = os.path.join(agent_sessions_dir, level_id)
        
        # Find all metrics files for this level
        metrics_pattern = os.path.join(level_dir, "session_metrics_*.json")
        metrics_files = sorted(glob.glob(metrics_pattern))
        
        # Load all metrics files
        level_sessions = []
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    # Add file creation time as session timestamp
                    file_time = os.path.getmtime(metrics_file)
                    metrics['file_timestamp'] = file_time
                    metrics['filename'] = os.path.basename(metrics_file)
                    level_sessions.append(metrics)
            except Exception as e:
                print(f"Failed to load metrics file {metrics_file}: {str(e)}")
        
        # Sort by timestamp
        level_sessions.sort(key=lambda x: x.get('file_timestamp', 0))
        
        # Store all session data for this level
        session_data[level_id] = level_sessions
    
    return session_data

def load_truth_knowledge_history(memory_dir: str) -> List[Dict[str, Any]]:
    """
    Load truth knowledge history
    
    Args:
        memory_dir: Memory data directory
        
    Returns:
        List of truth knowledge entries with timestamps
    """
    truth_file = os.path.join(memory_dir, 'truth_knowledge.json')
    
    if not os.path.exists(truth_file):
        print(f"Truth knowledge file not found: {truth_file}")
        return []
    
    try:
        with open(truth_file, 'r', encoding='utf-8') as f:
            truth_knowledge = json.load(f)
            
        # Sort by timestamp
        truth_knowledge.sort(key=lambda x: datetime.strptime(x.get('timestamp', '2000-01-01 00:00:00'), '%Y-%m-%d %H:%M:%S'))
        
        return truth_knowledge
    except Exception as e:
        print(f"Failed to load truth knowledge: {str(e)}")
        return []

def analyze_validation_issues(session_data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Analyze validation issues where subjective memories were promoted to truth despite degraded performance
    
    Args:
        session_data: Session data grouped by level ID
        
    Returns:
        DataFrame with validation issues
    """
    issues = []
    
    for level_id, sessions in session_data.items():
        # Skip if there are not enough sessions for comparison
        if len(sessions) < 2:
            continue
            
        # Check consecutive pairs of sessions
        for i in range(len(sessions) - 1):
            current_session = sessions[i]
            next_session = sessions[i+1]
            
            # Create timestamp strings
            current_time = datetime.fromtimestamp(current_session['file_timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            next_time = datetime.fromtimestamp(next_session['file_timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if performance degraded
            score_change = next_session['score'] - current_session['score']
            steps_change = next_session['steps'] - current_session['steps']
            success_degraded = current_session['success'] and not next_session['success']
            
            # If any metric got worse
            if score_change < 0 or (steps_change > 0 and next_session['success']) or success_degraded:
                issues.append({
                    'level_id': level_id,
                    'mode': current_session.get('mode', 'Unknown'),
                    'map_index': current_session.get('map_index', -1),
                    'session_1_time': current_time,
                    'session_2_time': next_time,
                    'score_1': current_session['score'],
                    'score_2': next_session['score'],
                    'score_change': score_change,
                    'steps_1': current_session['steps'],
                    'steps_2': next_session['steps'],
                    'steps_change': steps_change,
                    'success_1': current_session['success'],
                    'success_2': next_session['success'],
                    'success_degraded': success_degraded,
                    'validation_should_fail': True
                })
    
    # Create DataFrame
    df = pd.DataFrame(issues)
    
    if not df.empty:
        mode_order = {MODE_LEVEL1: 0, MODE_LEVEL2: 1, MODE_LEVEL3: 2}
        df['mode_order'] = df['mode'].map(lambda x: mode_order.get(x, 999))
        df = df.sort_values(['mode_order', 'map_index']).drop(columns=['mode_order'])
    
    return df

def analyze_session_metrics(session_data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Analyze session metrics for each level, comparing first and last sessions
    
    Args:
        session_data: Session data grouped by level ID
        
    Returns:
        DataFrame with comparison of metrics before and after training
    """
    results = []
    
    for level_id, sessions in session_data.items():
        if len(sessions) < 1:
            continue
            
        # Get first session (without memory enhancement)
        first_session = sessions[0]
        
        # Get last session (with memory enhancement)
        last_session = sessions[-1]
        
        # Calculate metric changes
        score_change = last_session['score'] - first_session['score']
        score_change_percent = (score_change / abs(first_session['score'])) * 100 if first_session['score'] != 0 else float('inf')
        
        steps_change = last_session['steps'] - first_session['steps']
        steps_change_percent = (steps_change / first_session['steps']) * 100 if first_session['steps'] > 0 else 0
        
        exploration_change = last_session.get('exploration_rate', 0) - first_session.get('exploration_rate', 0)
        
        # Add results to list
        results.append({
            'level_id': level_id,
            'mode': first_session.get('mode', 'Unknown'),
            'map_index': first_session.get('map_index', -1),
            
            # First session metrics
            'first_score': first_session['score'],
            'first_steps': first_session['steps'],
            'first_success': first_session['success'],
            'first_lives': first_session.get('lives_remaining', 0),
            'first_exploration': first_session.get('exploration_rate', 0),
            'first_coins': first_session.get('collected_coins', 0),
            'first_timestamp': datetime.fromtimestamp(first_session['file_timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            
            # Last session metrics
            'last_score': last_session['score'],
            'last_steps': last_session['steps'],
            'last_success': last_session['success'],
            'last_lives': last_session.get('lives_remaining', 0),
            'last_exploration': last_session.get('exploration_rate', 0),
            'last_coins': last_session.get('collected_coins', 0),
            'last_timestamp': datetime.fromtimestamp(last_session['file_timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            
            # Changes
            'score_change': score_change,
            'score_change_percent': score_change_percent,
            'steps_change': steps_change,
            'steps_change_percent': steps_change_percent,
            'success_improved': last_session['success'] and not first_session['success'],
            'exploration_change': exploration_change,
            
            # Session count
            'session_count': len(sessions)
        })
    
    # Create DataFrame and sort by mode and map index
    df = pd.DataFrame(results)
    
    if not df.empty:
        mode_order = {MODE_LEVEL1: 0, MODE_LEVEL2: 1, MODE_LEVEL3: 2}
        df['mode_order'] = df['mode'].map(lambda x: mode_order.get(x, 999))
        df = df.sort_values(['mode_order', 'map_index']).drop(columns=['mode_order'])
    
    return df

def generate_before_after_charts(df: pd.DataFrame, output_dir: str):
    """
    Generate before-after comparison charts
    
    Args:
        df: Analysis results DataFrame
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by mode
    for mode, mode_df in df.groupby('mode'):
        # Set figure size
        plt.figure(figsize=(14, 8))
        
        # Set bar positions
        indices = np.arange(len(mode_df))
        width = 0.35
        
        # Draw score comparison chart
        plt.subplot(2, 2, 1)
        plt.bar(indices - width/2, mode_df['first_score'], width, label='Before Training', color='#5A9BD5')
        plt.bar(indices + width/2, mode_df['last_score'], width, label='After Training', color='#ED7D31')
        plt.title(f'{mode} - Score Comparison')
        plt.xlabel('Map Number')
        plt.ylabel('Score')
        plt.xticks(indices, [f'Map {i+1}' for i in mode_df['map_index']])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Draw steps comparison chart
        plt.subplot(2, 2, 2)
        plt.bar(indices - width/2, mode_df['first_steps'], width, label='Before Training', color='#5A9BD5')
        plt.bar(indices + width/2, mode_df['last_steps'], width, label='After Training', color='#ED7D31')
        plt.title(f'{mode} - Steps Comparison')
        plt.xlabel('Map Number')
        plt.ylabel('Steps')
        plt.xticks(indices, [f'Map {i+1}' for i in mode_df['map_index']])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Draw exploration rate comparison chart
        plt.subplot(2, 2, 3)
        plt.bar(indices - width/2, mode_df['first_exploration'] * 100, width, label='Before Training', color='#5A9BD5')
        plt.bar(indices + width/2, mode_df['last_exploration'] * 100, width, label='After Training', color='#ED7D31')
        plt.title(f'{mode} - Exploration Rate Comparison (%)')
        plt.xlabel('Map Number')
        plt.ylabel('Exploration Rate (%)')
        plt.xticks(indices, [f'Map {i+1}' for i in mode_df['map_index']])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Draw success rate comparison chart
        plt.subplot(2, 2, 4)
        plt.bar(indices - width/2, mode_df['first_success'].astype(int) * 100, width, label='Before Training', color='#5A9BD5')
        plt.bar(indices + width/2, mode_df['last_success'].astype(int) * 100, width, label='After Training', color='#ED7D31')
        plt.title(f'{mode} - Success Rate Comparison (%)')
        plt.xlabel('Map Number')
        plt.ylabel('Success Rate (%)')
        plt.xticks(indices, [f'Map {i+1}' for i in mode_df['map_index']])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{mode}_training_comparison.png'))
        plt.close()
    
    # Generate overall improvement bar chart
    plt.figure(figsize=(12, 8))
    
    # Calculate average improvement rates by mode
    mode_improvement = df.groupby('mode').agg({
        'score_change_percent': 'mean',
        'steps_change_percent': 'mean',
        'exploration_change': 'mean',
        'success_improved': 'mean'
    }).reset_index()
    
    # Set bar positions
    indices = np.arange(len(mode_improvement))
    width = 0.2
    
    # Draw average improvement rate comparison chart
    plt.bar(indices - width*1.5, mode_improvement['score_change_percent'], width, label='Score Improvement (%)', color='#5A9BD5')
    plt.bar(indices - width/2, -mode_improvement['steps_change_percent'], width, label='Steps Reduction (%)', color='#ED7D31')
    plt.bar(indices + width/2, mode_improvement['exploration_change'] * 100, width, label='Exploration Increase (%)', color='#70AD47')
    plt.bar(indices + width*1.5, mode_improvement['success_improved'] * 100, width, label='Success Rate Increase (%)', color='#FFC000')
    
    plt.title('Training Effect Comparison by Mode')
    plt.xlabel('Game Mode')
    plt.ylabel('Improvement Percentage (%)')
    plt.xticks(indices, mode_improvement['mode'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_improvement.png'))
    plt.close()

def analyze_truth_growth(truth_knowledge: List[Dict[str, Any]], output_dir: str):
    """
    Analyze truth knowledge growth
    
    Args:
        truth_knowledge: Truth knowledge history
        output_dir: Output directory
    """
    if not truth_knowledge:
        print("Truth knowledge is empty, cannot analyze growth")
        return None, None
    
    # Group by timestamp and count truth knowledge size
    timestamps = [datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S') for item in truth_knowledge]
    unique_dates = sorted(list(set([ts.date() for ts in timestamps])))
    
    growth_data = []
    knowledge_by_date = {}
    
    # Count truth entries by date
    for i, date in enumerate(unique_dates):
        # Calculate truth entries up to this date
        count = sum(1 for ts in timestamps if ts.date() <= date)
        growth_data.append((date, count))
        
        # Collect truth entries by date
        knowledge_on_date = [item for item, ts in zip(truth_knowledge, timestamps) if ts.date() == date]
        knowledge_by_date[date] = knowledge_on_date
    
    # Generate truth knowledge growth chart
    plt.figure(figsize=(12, 6))
    
    dates = [item[0] for item in growth_data]
    counts = [item[1] for item in growth_data]
    
    plt.plot(dates, counts, 'o-', linewidth=2, markersize=8)
    plt.title('Truth Knowledge Growth Trend')
    plt.xlabel('Date')
    plt.ylabel('Number of Truth Entries')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add data labels
    for i, (date, count) in enumerate(growth_data):
        if i == 0 or i == len(growth_data) - 1 or count != growth_data[i-1][1]:
            plt.annotate(str(count), (date, count), textcoords="offset points", 
                          xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'truth_knowledge_growth.png'))
    plt.close()
    
    # Generate truth knowledge source distribution chart
    sources = [item.get('source', 'Unknown') for item in truth_knowledge]
    source_counts = {}
    
    for source in sources:
        if source in source_counts:
            source_counts[source] += 1
        else:
            source_counts[source] = 1
    
    plt.figure(figsize=(10, 6))
    
    source_names = list(source_counts.keys())
    source_values = list(source_counts.values())
    
    # Sort by value
    sorted_indices = np.argsort(source_values)[::-1]
    sorted_names = [source_names[i] for i in sorted_indices]
    sorted_values = [source_values[i] for i in sorted_indices]
    
    plt.bar(sorted_names, sorted_values, color='#5A9BD5')
    plt.title('Truth Knowledge Source Distribution')
    plt.xlabel('Knowledge Source')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add data labels
    for i, v in enumerate(sorted_values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'truth_knowledge_sources.png'))
    plt.close()
    
    # Return truth knowledge growth data and truth entries grouped by date
    return growth_data, knowledge_by_date

def visualize_validation_issues(validation_df: pd.DataFrame, output_dir: str):
    """
    Visualize validation issues
    
    Args:
        validation_df: DataFrame with validation issues
        output_dir: Output directory
    """
    if validation_df.empty:
        print("No validation issues to visualize")
        return
        
    # Count issues by mode
    issues_by_mode = validation_df.groupby('mode').size().reset_index(name='issue_count')
    
    plt.figure(figsize=(10, 6))
    
    plt.bar(issues_by_mode['mode'], issues_by_mode['issue_count'], color='#FF6B6B')
    plt.title('Memory Validation Issues by Mode')
    plt.xlabel('Game Mode')
    plt.ylabel('Count of Validation Issues')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add data labels
    for i, v in enumerate(issues_by_mode['issue_count']):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_issues_by_mode.png'))
    plt.close()
    
    # Visualize score changes in validation issues
    plt.figure(figsize=(12, 8))
    
    # Sort by score change
    validation_df_sorted = validation_df.sort_values('score_change')
    
    # Create labels
    labels = [f"{row['mode']} Map {row['map_index']+1}" for _, row in validation_df_sorted.iterrows()]
    
    plt.bar(labels, validation_df_sorted['score_change'], color='#FF6B6B')
    plt.title('Score Degradation in Memory Validation Issues')
    plt.xlabel('Level')
    plt.ylabel('Score Change')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_score_changes.png'))
    plt.close()

def create_excel_report(metrics_df: pd.DataFrame, validation_df: pd.DataFrame, 
                      truth_growth: List[Tuple], knowledge_by_date: Dict, output_file: str):
    """
    Create Excel report
    
    Args:
        metrics_df: Metrics analysis DataFrame
        validation_df: Validation issues DataFrame
        truth_growth: Truth knowledge growth data
        knowledge_by_date: Truth entries grouped by date
        output_file: Output file path
    """
    with pd.ExcelWriter(output_file) as writer:
        # Write training effect summary table
        metrics_df.to_excel(writer, sheet_name='Training Effects', index=False)
        
        # Write validation issues
        if not validation_df.empty:
            validation_df.to_excel(writer, sheet_name='Validation Issues', index=False)
        
        # Write detailed tables by mode
        for mode, mode_df in metrics_df.groupby('mode'):
            sheet_name = f'{mode[:28]}_Details'
            mode_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Write truth knowledge growth data
        if truth_growth:
            growth_df = pd.DataFrame(truth_growth, columns=['Date', 'Truth Entry Count'])
            growth_df.to_excel(writer, sheet_name='Truth Growth', index=False)
        
        # Write truth knowledge content table
        if knowledge_by_date:
            truth_data = []
            
            for date, items in knowledge_by_date.items():
                for item in items:
                    truth_data.append({
                        'Date': date,
                        'Source': item.get('source', 'Unknown'),
                        'Knowledge': item.get('knowledge', '')
                    })
            
            truth_df = pd.DataFrame(truth_data)
            truth_df = truth_df.sort_values('Date')
            truth_df.to_excel(writer, sheet_name='Truth Content', index=False)

def main():
    parser = argparse.ArgumentParser(description='Evaluate learning agent training effects')
    parser.add_argument('--sessions_dir', type=str, default=None, 
                        help='Session data directory path (default: outputs/agent_sessions)')
    parser.add_argument('--memory_dir', type=str, default=None,
                        help='Memory data directory path (default: outputs/memory/agent_memory)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory path (default: outputs/results/learning_evaluation)')
    
    args = parser.parse_args()
    
    # Set default paths
    sessions_dir = args.sessions_dir or str(AGENT_SESSIONS_DIR)
    memory_dir = args.memory_dir or str(AGENT_MEMORY_DIR)
    output_dir = args.output_dir or os.path.join(str(RESULTS_DIR), 'learning_evaluation')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting evaluation of learning agent training effects...")
    print(f"Session data directory: {sessions_dir}")
    print(f"Memory data directory: {memory_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load session data
    print("Loading session data...")
    session_data = load_session_data(sessions_dir)
    print(f"Loaded session data for {len(session_data)} levels")
    
    # Analyze session metrics
    print("Analyzing session metrics...")
    metrics_df = analyze_session_metrics(session_data)
    
    if metrics_df.empty:
        print("No valid session data found, cannot generate evaluation report")
        return
    
    # Analyze validation issues
    print("Analyzing memory validation issues...")
    validation_df = analyze_validation_issues(session_data)
    print(f"Found {len(validation_df)} potential validation issues")
    
    # Visualize validation issues
    if not validation_df.empty:
        print("Generating validation issue visualizations...")
        visualize_validation_issues(validation_df, output_dir)
    
    # Generate before-after comparison charts
    print("Generating before-after comparison charts...")
    generate_before_after_charts(metrics_df, output_dir)
    
    # Load truth knowledge history
    print("Loading truth knowledge history...")
    truth_knowledge = load_truth_knowledge_history(memory_dir)
    print(f"Loaded {len(truth_knowledge)} truth knowledge entries")
    
    # Analyze truth knowledge growth
    truth_growth = None
    knowledge_by_date = None
    
    if truth_knowledge:
        print("Analyzing truth knowledge growth...")
        truth_growth, knowledge_by_date = analyze_truth_growth(truth_knowledge, output_dir)
    
    # Create Excel report
    excel_file = os.path.join(output_dir, 'learning_evaluation_report.xlsx')
    print(f"Creating Excel report: {excel_file}")
    create_excel_report(metrics_df, validation_df, truth_growth, knowledge_by_date, excel_file)
    
    print(f"Evaluation complete! Report saved to: {output_dir}")
    print(f"Excel report: {excel_file}")
    print(f"Chart files:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {os.path.join(output_dir, file)}")

if __name__ == "__main__":
    main() 
