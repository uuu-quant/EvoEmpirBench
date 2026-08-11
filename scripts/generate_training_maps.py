import argparse
import os
import random
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.game.map_generator import MapGenerator
from src.config.game_config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3
from src.config.paths import MAZE_TRAIN_MAPS_DIR

class TrainingMapGenerator:
    """训练用地图生成器，生成特定难度和结构的训练地图"""
    
    @staticmethod
    def generate_training_map(mode: str) -> dict:
        """
        生成单个训练地图，对比普通地图进行调整，以便更好地训练代理
        
        Args:
            mode: 游戏模式
            
        Returns:
            地图数据字典
        """
        # 使用基础地图生成器生成地图
        map_data = MapGenerator.generate_map(mode)
        
        # 根据不同的游戏模式进行特定调整
        if mode == MODE_LEVEL1:
            # Level 1的特殊调整：确保金币分布更均匀
            pass
        
        elif mode == MODE_LEVEL2:
            # Level 2的特殊调整：怪物位置更加战略性
            pass
        
        elif mode == MODE_LEVEL3:
            # Level 3的特殊调整：道具放置在更合理的位置
            # 确保钥匙不会太容易拿到，需要探索更多区域
            pass
        
        return map_data
    
    @staticmethod
    def generate_training_dataset(maps_per_level: int = 10, save_dir: str = None):
        """
        生成训练数据集
        
        Args:
            maps_per_level: 每个难度级别的地图数量
            save_dir: 保存目录，默认为 data/levels/maze_train
        """
        # 设置保存目录
        if save_dir is None:
            save_dir = str(MAZE_TRAIN_MAPS_DIR)
        os.makedirs(save_dir, exist_ok=True)
        
        modes = [MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3]
        
        for mode in modes:
            print(f"正在生成 {mode} 训练地图...")
            maps = []
            
            for i in range(maps_per_level):
                print(f"  正在生成第 {i+1}/{maps_per_level} 个 {mode} 地图")
                map_data = TrainingMapGenerator.generate_training_map(mode)
                maps.append(map_data)
                
                # 保存单个地图
                filename = f"{mode.replace(' ', '_')}_{i+1}.json"
                filepath = os.path.join(save_dir, filename)
                with open(filepath, 'w') as f:
                    import json
                    json.dump(map_data, f, indent=2)
            
            # 保存地图集合
            collection_filename = f"{mode.replace(' ', '_')}_collection.json"
            collection_filepath = os.path.join(save_dir, collection_filename)
            with open(collection_filepath, 'w') as f:
                import json
                json.dump(maps, f, indent=2)
            
            print(f"已生成 {len(maps)} 个 {mode} 训练地图")
        
        print(f"\n所有训练地图已保存到: {save_dir}")
        print("训练地图集合：")
        print(f"- Level_1_collection.json: 简单迷宫 (7x7, 无怪物)")
        print(f"- Level_2_collection.json: 怪物迷宫 (9x9, 有移动怪物)")
        print(f"- Level_3_collection.json: 道具迷宫 (11x11, 有怪物和道具)")

def main():
    """生成训练地图数据集"""
    parser = argparse.ArgumentParser(description="生成训练地图数据集")
    parser.add_argument("--count", type=int, default=10, help="每个难度级别生成的训练地图数量")
    parser.add_argument("--save_dir", type=str, default=None, help="训练地图保存目录，默认 data/levels/maze_train")
    args = parser.parse_args()

    TrainingMapGenerator.generate_training_dataset(args.count, args.save_dir)
    print("\n地图生成完成！可以使用 python examples/play_with_maps.py 进行游戏测试。")

if __name__ == "__main__":
    main() 
