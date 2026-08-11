import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def load_results(results_file: str) -> Dict:
    """加载评估结果文件"""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_summary_dataframe(results: Dict) -> pd.DataFrame:
    """创建汇总数据框"""
    summary_data = []
    
    for mode, mode_data in results["results"].items():
        if isinstance(mode_data, list):  # 旧格式
            raw_results = mode_data
            success_count = sum(1 for r in raw_results if r.get("success", False))
            success_rate = success_count / len(raw_results) if raw_results else 0
            avg_score = sum(r.get("custom_score", 0) for r in raw_results) / len(raw_results) if raw_results else 0
            avg_exploration = sum(r.get("exploration_rate", 0) for r in raw_results) / len(raw_results) if raw_results else 0
            avg_coins = sum(r.get("collected_coins", 0) for r in raw_results) / len(raw_results) if raw_results else 0
            avg_steps = sum(r.get("steps", 0) for r in raw_results) / len(raw_results) if raw_results else 0
            avg_monsters = sum(r.get("killed_monsters", 0) for r in raw_results) / len(raw_results) if raw_results else 0
            avg_obstacles = sum(r.get("destroyed_obstacles", 0) for r in raw_results) / len(raw_results) if raw_results else 0
        else:  # 新格式
            stats = mode_data.get("stats", {})
            success_rate = stats.get("success_rate", 0)
            avg_score = stats.get("avg_score", 0)
            avg_exploration = stats.get("avg_exploration", 0)
            avg_coins = stats.get("avg_collected_coins", 0)
            avg_steps = stats.get("avg_steps", 0)
            avg_monsters = stats.get("avg_killed_monsters", 0)
            avg_obstacles = stats.get("avg_destroyed_obstacles", 0)
        
        summary_data.append({
            "模式": mode,
            "通关率": success_rate,
            "平均得分": avg_score,
            "平均探索率": avg_exploration,
            "平均金币收集": avg_coins,
            "平均步数": avg_steps,
            "平均击杀怪物": avg_monsters,
            "平均破坏障碍": avg_obstacles
        })
    
    return pd.DataFrame(summary_data)

def create_detailed_dataframes(results: Dict) -> Dict[str, pd.DataFrame]:
    """创建每个模式的详细数据框"""
    detailed_dfs = {}
    
    for mode, mode_data in results["results"].items():
        if isinstance(mode_data, list):  # 旧格式
            raw_results = mode_data
        else:  # 新格式
            raw_results = mode_data.get("raw_results", [])
        
        if raw_results:
            # 创建数据框
            df = pd.DataFrame(raw_results)
            # 添加模式列
            df["模式"] = mode
            detailed_dfs[mode] = df
    
    return detailed_dfs

def create_visualizations(results_file: str, output_dir: Optional[str] = None):
    """创建可视化图表"""
    # 加载结果
    results = load_results(results_file)
    
    # 设置输出目录
    if output_dir is None:
        # 使用结果文件所在目录
        output_dir = os.path.dirname(os.path.abspath(results_file))
    
    # 创建汇总数据框
    summary_df = create_summary_dataframe(results)
    
    # 创建详细数据框
    detailed_dfs = create_detailed_dataframes(results)
    
    # 设置风格
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 生成文件名前缀
    timestamp = results.get("timestamp", "unknown")
    prefix = f"viz_{timestamp}"
    
    # 1. 绘制各模式通关率对比条形图
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="模式", y="通关率", data=summary_df, palette="viridis")
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.1%}", 
                   (p.get_x() + p.get_width()/2., p.get_height()), 
                   ha='center', va='bottom', fontsize=12)
    plt.title("各难度级别通关率对比")
    plt.ylabel("通关率")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_success_rate.png"))
    plt.close()
    
    # 2. 绘制平均得分对比条形图
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="模式", y="平均得分", data=summary_df, palette="viridis")
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.0f}", 
                   (p.get_x() + p.get_width()/2., p.get_height()), 
                   ha='center', va='bottom', fontsize=12)
    plt.title("各难度级别平均得分对比")
    plt.ylabel("平均得分")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_avg_score.png"))
    plt.close()
    
    # 3. 绘制平均探索率对比条形图
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="模式", y="平均探索率", data=summary_df, palette="viridis")
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.1%}", 
                   (p.get_x() + p.get_width()/2., p.get_height()), 
                   ha='center', va='bottom', fontsize=12)
    plt.title("各难度级别平均探索率对比")
    plt.ylabel("平均探索率")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_avg_exploration.png"))
    plt.close()
    
    # 4. 绘制多指标雷达图
    plt.figure(figsize=(10, 8))
    
    # 准备雷达图数据
    categories = ['通关率', '平均探索率', '平均金币收集/5', '平均步数/200']
    
    # 转换数据到0-1范围便于雷达图显示
    for mode in summary_df['模式']:
        row = summary_df[summary_df['模式'] == mode].iloc[0]
        values = [
            row['通关率'],  # 通关率已经是0-1范围
            row['平均探索率'],  # 探索率已经是0-1范围
            row['平均金币收集'] / 5,  # 金币收集率（总共5个金币）
            row['平均步数'] / 200  # 步数比例（相对于最大200步）
        ]
        
        # 确保闭合的雷达图
        values = np.concatenate((values, [values[0]]))
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 绘制雷达图
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2, label=mode)
        ax.fill(angles, values, alpha=0.25)
    
    # 设置雷达图样式
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["25%", "50%", "75%", "100%"], color="grey", size=8)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("各难度级别性能雷达图")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_radar_chart.png"))
    plt.close()
    
    # 5. 为每个模式创建详细分析
    for mode, df in detailed_dfs.items():
        # 5.1 成功与失败比例饼图
        plt.figure(figsize=(8, 8))
        success_count = df['success'].sum()
        failure_count = len(df) - success_count
        plt.pie([success_count, failure_count], 
                labels=['成功', '失败'], 
                autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'],
                explode=(0.1, 0))
        plt.title(f'{mode} 通关情况')
        plt.savefig(os.path.join(output_dir, f"{prefix}_{mode}_success_pie.png"))
        plt.close()
        
        # 5.2 探索率分布直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(df['exploration_rate'], bins=10, kde=True)
        plt.axvline(df['exploration_rate'].mean(), color='r', linestyle='--', label=f'平均值: {df["exploration_rate"].mean():.2f}')
        plt.title(f'{mode} 探索率分布')
        plt.xlabel('探索率')
        plt.ylabel('地图数量')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_{mode}_exploration_hist.png"))
        plt.close()
        
        # 5.3 得分与步数关系散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='steps', y='custom_score', hue='success', data=df, palette={True: '#2ecc71', False: '#e74c3c'}, s=100)
        plt.title(f'{mode} 得分与步数关系')
        plt.xlabel('步数')
        plt.ylabel('得分')
        plt.grid(True, alpha=0.3)
        plt.legend(title='是否通关')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_{mode}_score_steps.png"))
        plt.close()
        
        # 5.4 金币收集直方图
        plt.figure(figsize=(8, 6))
        sns.countplot(x='collected_coins', data=df, palette='viridis')
        plt.title(f'{mode} 金币收集情况')
        plt.xlabel('收集金币数量')
        plt.ylabel('地图数量')
        plt.xticks(range(6))  # 0-5个金币
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_{mode}_coins_collected.png"))
        plt.close()
        
        # 特定于Level 2和Level 3的图表
        if mode in ["Level 2", "Level 3"]:
            # 怪物击杀分布
            plt.figure(figsize=(8, 6))
            sns.countplot(x='killed_monsters', data=df, palette='viridis')
            plt.title(f'{mode} 怪物击杀情况')
            plt.xlabel('击杀怪物数量')
            plt.ylabel('地图数量')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_{mode}_monsters_killed.png"))
            plt.close()
        
        # 特定于Level 3的图表
        if mode == "Level 3":
            # 障碍物破坏分布
            plt.figure(figsize=(8, 6))
            sns.countplot(x='destroyed_obstacles', data=df, palette='viridis')
            plt.title(f'{mode} 障碍物破坏情况')
            plt.xlabel('破坏障碍物数量')
            plt.ylabel('地图数量')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_{mode}_obstacles_destroyed.png"))
            plt.close()
    
    # 6. 创建综合比较图
    key_metrics = ['通关率', '平均探索率', '平均得分']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(key_metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x='模式', y=metric, data=summary_df, palette='viridis')
        plt.title(f'各难度级别 {metric} 比较')
        
        # 添加数值标签
        ax = plt.gca()
        for p in ax.patches:
            value = p.get_height()
            if metric in ['通关率', '平均探索率']:
                ax.annotate(f"{value:.1%}", 
                           (p.get_x() + p.get_width()/2., value), 
                           ha='center', va='bottom', fontsize=10)
            else:
                ax.annotate(f"{value:.0f}", 
                           (p.get_x() + p.get_width()/2., value), 
                           ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_key_metrics_comparison.png"))
    plt.close()
    
    # 7. 保存概要报告为CSV
    summary_df.to_csv(os.path.join(output_dir, f"{prefix}_summary.csv"), index=False)
    
    # 打印完成消息
    print(f"可视化图表已保存到: {output_dir}")
    print(f"生成了 {mode} 模式的详细分析图和总体对比图")
    print(f"概要报告已保存为CSV: {os.path.join(output_dir, f'{prefix}_summary.csv')}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估结果可视化工具")
    parser.add_argument("results_file", type=str, help="评估结果JSON文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认为结果文件所在目录")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查结果文件是否存在
    if not os.path.exists(args.results_file):
        print(f"错误: 找不到结果文件 {args.results_file}")
        return
    
    print(f"正在为结果文件 {args.results_file} 创建可视化图表...")
    create_visualizations(args.results_file, args.output_dir)

if __name__ == "__main__":
    main() 