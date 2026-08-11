import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from src.config.paths import AGENT_SESSIONS_DIR, RESULTS_DIR

def load_session_data(agent_sessions_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    加载所有会话数据，按关卡ID分组
    
    Args:
        agent_sessions_dir: 会话数据目录
        
    Returns:
        按关卡ID分组的会话数据字典
    """
    session_data = {}
    
    # 遍历会话目录
    level_dirs = [d for d in os.listdir(agent_sessions_dir) 
                 if os.path.isdir(os.path.join(agent_sessions_dir, d))]
    
    for level_id in level_dirs:
        level_dir = os.path.join(agent_sessions_dir, level_id)
        
        # 查找该关卡的所有指标文件
        metrics_pattern = os.path.join(level_dir, "session_metrics_*.json")
        metrics_files = sorted(glob.glob(metrics_pattern))
        
        # 加载所有指标文件
        level_sessions = []
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    # 添加文件创建时间作为会话时间戳
                    file_time = os.path.getmtime(metrics_file)
                    metrics['file_timestamp'] = file_time
                    metrics['filename'] = os.path.basename(metrics_file)
                    level_sessions.append(metrics)
            except Exception as e:
                print(f"加载指标文件 {metrics_file} 失败: {str(e)}")
        
        # 按时间戳排序
        level_sessions.sort(key=lambda x: x.get('file_timestamp', 0))
        
        # 存储该关卡的所有会话数据
        session_data[level_id] = level_sessions
    
    return session_data

def analyze_learning_progress(session_data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    分析学习进度，生成包含初始得分和最终得分的DataFrame
    
    Args:
        session_data: 按关卡ID分组的会话数据字典
        
    Returns:
        学习进度DataFrame
    """
    progress_data = []
    
    # 提取每个关卡的学习进度
    for level_id, sessions in session_data.items():
        if not sessions:
            continue
        
        # 获取该关卡的模式
        mode = sessions[0].get('mode', 'Unknown')
        
        # 获取关卡索引
        map_index = sessions[0].get('map_index', -1)
        
        # 获取初始会话（无记忆增强）和最终会话（有记忆增强）
        initial_session = sessions[0]  # 第一次尝试，无记忆增强
        final_session = sessions[-1] if len(sessions) > 1 else initial_session  # 最后一次尝试，有记忆增强
        
        # 提取学习进度数据
        progress_entry = {
            'level_id': level_id,
            'mode': mode,
            'map_index': map_index,
            'initial_score': initial_session.get('score', 0),
            'final_score': final_session.get('score', 0),
            'score_improvement': final_session.get('score', 0) - initial_session.get('score', 0),
            'score_improvement_percent': ((final_session.get('score', 0) / initial_session.get('score', 1)) - 1) * 100,
            'initial_success': initial_session.get('success', False),
            'final_success': final_session.get('success', False),
            'initial_steps': initial_session.get('steps', 0),
            'final_steps': final_session.get('steps', 0),
            'step_reduction': initial_session.get('steps', 0) - final_session.get('steps', 0),
            'step_reduction_percent': (1 - (final_session.get('steps', 0) / initial_session.get('steps', 1))) * 100,
            'initial_exploration_rate': initial_session.get('exploration_rate', 0),
            'final_exploration_rate': final_session.get('exploration_rate', 0),
            'attempts_count': len(sessions)
        }
        
        progress_data.append(progress_entry)
    
    # 创建DataFrame
    df = pd.DataFrame(progress_data)
    
    # 按模式和地图索引排序
    mode_order = {MODE_LEVEL1: 0, MODE_LEVEL2: 1, MODE_LEVEL3: 2}
    df['mode_order'] = df['mode'].map(lambda x: mode_order.get(x, 999))
    df = df.sort_values(['mode_order', 'map_index']).drop(columns=['mode_order'])
    
    return df

def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成按模式分组的汇总统计
    
    Args:
        df: 学习进度DataFrame
        
    Returns:
        汇总统计DataFrame
    """
    # 按模式分组计算统计
    summary = df.groupby('mode').agg({
        'score_improvement': ['mean', 'min', 'max'],
        'score_improvement_percent': ['mean', 'min', 'max'],
        'initial_success': 'mean',
        'final_success': 'mean',
        'step_reduction': ['mean', 'min', 'max'],
        'step_reduction_percent': ['mean', 'min', 'max']
    })
    
    # 重命名列
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # 添加成功率改进
    summary['success_rate_improvement'] = summary['final_success_mean'] - summary['initial_success_mean']
    
    return summary

def generate_charts(df: pd.DataFrame, output_dir: str):
    """
    生成可视化图表
    
    Args:
        df: 学习进度DataFrame
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按模式分组
    for mode, mode_df in df.groupby('mode'):
        # 得分对比图
        plt.figure(figsize=(12, 6))
        indices = np.arange(len(mode_df))
        width = 0.35
        
        plt.bar(indices - width/2, mode_df['initial_score'], width, label='初始得分')
        plt.bar(indices + width/2, mode_df['final_score'], width, label='优化后得分')
        
        plt.title(f'{mode} 学习前后得分对比')
        plt.xlabel('地图索引')
        plt.ylabel('得分')
        plt.xticks(indices, [f'Map {i+1}' for i in mode_df['map_index']])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加得分提升百分比标签
        for i, row in enumerate(mode_df.itertuples()):
            improvement = row.score_improvement
            if improvement > 0:
                plt.text(i, max(row.initial_score, row.final_score) + 100, 
                        f'+{improvement:.0f}\n({row.score_improvement_percent:.1f}%)', 
                        ha='center', va='bottom', color='green')
            elif improvement < 0:
                plt.text(i, max(row.initial_score, row.final_score) + 100, 
                        f'{improvement:.0f}\n({row.score_improvement_percent:.1f}%)', 
                        ha='center', va='bottom', color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{mode.replace(" ", "_")}_score_comparison.png'))
        plt.close()
        
        # 步数对比图
        plt.figure(figsize=(12, 6))
        
        plt.bar(indices - width/2, mode_df['initial_steps'], width, label='初始步数')
        plt.bar(indices + width/2, mode_df['final_steps'], width, label='优化后步数')
        
        plt.title(f'{mode} 学习前后步数对比')
        plt.xlabel('地图索引')
        plt.ylabel('步数')
        plt.xticks(indices, [f'Map {i+1}' for i in mode_df['map_index']])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加步数减少百分比标签
        for i, row in enumerate(mode_df.itertuples()):
            reduction = row.step_reduction
            if reduction > 0:
                plt.text(i, max(row.initial_steps, row.final_steps) + 5, 
                        f'-{reduction:.0f}\n({row.step_reduction_percent:.1f}%)', 
                        ha='center', va='bottom', color='green')
            elif reduction < 0:
                plt.text(i, max(row.initial_steps, row.final_steps) + 5, 
                        f'{reduction:.0f}\n({row.step_reduction_percent:.1f}%)', 
                        ha='center', va='bottom', color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{mode.replace(" ", "_")}_steps_comparison.png'))
        plt.close()
        
    # 生成模式间的比较图
    mode_summary = df.groupby('mode').agg({
        'score_improvement_percent': 'mean',
        'step_reduction_percent': 'mean',
        'final_success': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # 排序模式
    mode_order = {MODE_LEVEL1: 0, MODE_LEVEL2: 1, MODE_LEVEL3: 2}
    mode_summary['mode_order'] = mode_summary['mode'].map(lambda x: mode_order.get(x, 999))
    mode_summary = mode_summary.sort_values('mode_order').drop(columns=['mode_order'])
    
    # 绘制模式间比较柱状图
    indices = np.arange(len(mode_summary))
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    width = 0.3
    ax1.bar(indices - width, mode_summary['score_improvement_percent'], width, label='得分提升百分比', color='green')
    ax1.bar(indices, mode_summary['step_reduction_percent'], width, label='步数减少百分比', color='blue')
    
    ax1.set_xlabel('游戏模式')
    ax1.set_ylabel('百分比 (%)')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(mode_summary['mode'])
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(indices, mode_summary['final_success'] * 100, 'ro-', label='最终通关率')
    ax2.set_ylabel('通关率 (%)')
    ax2.legend(loc='upper right')
    
    plt.title('不同模式的学习效果比较')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mode_comparison.png'))
    plt.close()

def main():
    # 设置会话数据目录
    agent_sessions_dir = str(AGENT_SESSIONS_DIR)
    
    # 设置输出目录
    output_dir = os.path.join(str(RESULTS_DIR), 'learning_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"加载会话数据...")
    session_data = load_session_data(agent_sessions_dir)
    print(f"找到 {len(session_data)} 个关卡的会话数据")
    
    print(f"分析学习进度...")
    progress_df = analyze_learning_progress(session_data)
    
    print(f"生成汇总统计...")
    summary_df = generate_summary_statistics(progress_df)
    
    print(f"生成图表...")
    generate_charts(progress_df, output_dir)
    
    # 保存Excel报告
    excel_file = os.path.join(output_dir, 'learning_progress_report.xlsx')
    with pd.ExcelWriter(excel_file) as writer:
        # 学习进度表
        progress_df.to_excel(writer, sheet_name='学习进度详情', index=False)
        
        # 汇总统计表
        summary_df.to_excel(writer, sheet_name='模式汇总统计')
        
        # 按模式分组的详细表格
        for mode, mode_df in progress_df.groupby('mode'):
            mode_name = mode.replace(' ', '_')
            mode_df.to_excel(writer, sheet_name=f'{mode_name[:28]}_详情', index=False)
    
    print(f"学习进度报告已生成: {excel_file}")
    print(f"图表已保存到: {output_dir}")

if __name__ == "__main__":
    main() 
