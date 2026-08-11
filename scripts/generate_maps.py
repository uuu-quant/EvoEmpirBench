import os
import sys
import argparse

# 添加项目根目录到Python路径，以便能正确导入src模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 现在导入项目模块
from src.game.map_generator import MapGenerator
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from src.config.paths import MAZE_EVAL_MAPS_DIR

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成迷宫地图文件")
    parser.add_argument("--save_dir", type=str, help="地图保存目录的自定义路径")
    parser.add_argument("--count", type=int, default=30, help="每个难度级别生成的地图数量 (默认: 30)")
    args = parser.parse_args()
    
    # 设置保存目录（使用命令行指定的目录或默认目录）
    if args.save_dir:
        save_dir = args.save_dir
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = str(MAZE_EVAL_MAPS_DIR)
    
    # 设置每个难度的地图数量
    maps_per_level = args.count
    
    print(f"地图将保存到: {save_dir}")
    print(f"每个难度级别将生成 {maps_per_level} 个地图")
    
    # 生成Level 1地图
    print(f"正在生成 {MODE_LEVEL1} 地图...")
    level1_maps = MapGenerator.generate_maps(maps_per_level, MODE_LEVEL1, save_dir)
    print(f"已生成 {len(level1_maps)} 个 {MODE_LEVEL1} 地图")
    
    # 生成Level 2地图
    print(f"正在生成 {MODE_LEVEL2} 地图...")
    level2_maps = MapGenerator.generate_maps(maps_per_level, MODE_LEVEL2, save_dir)
    print(f"已生成 {len(level2_maps)} 个 {MODE_LEVEL2} 地图")
    
    # 生成Level 3地图
    print(f"正在生成 {MODE_LEVEL3} 地图...")
    level3_maps = MapGenerator.generate_maps(maps_per_level, MODE_LEVEL3, save_dir)
    print(f"已生成 {len(level3_maps)} 个 {MODE_LEVEL3} 地图")
    
    print(f"\n所有地图已保存到: {save_dir}")
    print("地图文件说明：")
    print("- Level_1_collection.json: 所有Level 1地图的集合 (7x7, 无怪物)")
    print("- Level_2_collection.json: 所有Level 2地图的集合 (9x9, 有怪物)")
    print("- Level_3_collection.json: 所有Level 3地图的集合 (11x11, 有怪物和道具)")
    print("- Level_1_X.json: 单个Level 1地图")
    print("- Level_2_X.json: 单个Level 2地图")
    print("- Level_3_X.json: 单个Level 3地图")
    print("\n游戏说明:")
    print("- Level 1: 基础迷宫，7x7地图，5个金币，没有怪物")
    print("- Level 2: 怪物迷宫，9x9地图，5个金币，2个会移动的怪物")
    print("- Level 3: 道具迷宫，11x11地图，5个金币，2个怪物，4种道具:")
    print("  * 铲子: 可破坏障碍3次")
    print("  * 剑: 可无限次击杀怪物")
    print("  * 磁铁: 可吸取5x5范围内的金币")
    print("  * 钥匙: 必须拾取才能通关")
    
    # 打印使用示例
    print("\n使用示例:")
    print("1. 使用默认路径生成30个地图:")
    print("   python scripts/generate_maps.py")
    print("2. 指定自定义保存路径:")
    print("   python scripts/generate_maps.py --save_dir /path/to/your/maps")
    print("3. 自定义地图数量:")
    print("   python scripts/generate_maps.py --count 50")
    print("4. 同时指定路径和数量:")
    print("   python scripts/generate_maps.py --save_dir /path/to/your/maps --count 50")

if __name__ == "__main__":
    main() 
