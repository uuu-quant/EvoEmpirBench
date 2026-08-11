#!/usr/bin/env python
"""
迷宫游戏 - AI代理模式启动脚本
使用LLM API控制角色探索迷宫
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from examples.play_with_agent import main

def parse_args():
    parser = argparse.ArgumentParser(description="AI迷宫游戏 - LLM代理模式")
    parser.add_argument("--hide_prompt", action="store_true",
                        help="隐藏发送给模型的提示文本")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    print("=" * 60)
    print("欢迎使用AI迷宫游戏 - LLM代理模式")
    print("=" * 60)
    print("本程序将使用LLM API控制角色探索迷宫")
    print("您可以观察AI代理如何分析地图并做出决策")
    print("未探索区域将用?表示，代理会逐步探索地图")
    print("\n游戏规则:")
    print("- 不能穿过墙壁(#)，否则扣除生命值并回到起点")
    print("- 碰到怪物(M)将扣除生命值并回到起点")
    print("- 收集金币(C)可获得分数")
    print("- 每步移动消耗少量分数(-0.1)")
    print("- 生命值耗尽则游戏结束")
    print("\n控制:")
    print("- 按ESC键可随时退出游戏")
    print("\n提示显示:")
    print(f"- 当前设置: {'隐藏' if args.hide_prompt else '显示'}")
    print("- 可通过 --hide_prompt 参数隐藏提示")
    print("=" * 60)
    
    # 传递参数并运行主程序
    sys.argv = [sys.argv[0]]
    if args.hide_prompt:
        sys.argv.append("--hide_prompt")
    
    # 运行主程序
    main() 
