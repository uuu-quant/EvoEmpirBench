import numpy as np
from typing import List, Tuple

class Obstacle:
    @staticmethod
    def create_cross(center: Tuple[int, int]) -> List[Tuple[int, int]]:
        """创建十字形障碍物"""
        x, y = center
        return [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]

    @staticmethod
    def create_l_shape(corner: Tuple[int, int], orientation: int = 0) -> List[Tuple[int, int]]:
        """创建L形障碍物"""
        x, y = corner
        orientations = [
            [(x, y), (x-1, y), (x, y+1)],  # 基础方向
            [(x, y), (x+1, y), (x, y+1)],  # 旋转90度
            [(x, y), (x+1, y), (x, y-1)],  # 旋转180度
            [(x, y), (x-1, y), (x, y-1)]   # 旋转270度
        ]
        return orientations[orientation % 4]

    @staticmethod
    def create_rectangle(top_left: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
        """创建矩形障碍物"""
        x, y = top_left
        return [(i, j) for i in range(x, x + height) for j in range(y, y + width)]

    @staticmethod
    def create_line(start: Tuple[int, int], length: int, horizontal: bool = True) -> List[Tuple[int, int]]:
        """创建直线障碍物"""
        x, y = start
        if horizontal:
            return [(x, y + i) for i in range(length)]
        return [(x + i, y) for i in range(length)]

    @staticmethod
    def validate_positions(positions: List[Tuple[int, int]], grid_size: int) -> bool:
        """验证障碍物位置是否有效"""
        return all(0 <= x < grid_size and 0 <= y < grid_size for x, y in positions)

    @staticmethod
    def get_random_obstacle_layout(grid_size: int) -> List[Tuple[int, int]]:
        """生成随机障碍物布局，根据地图大小调整"""
        obstacles = []
        
        # 确保起点和终点不被占用
        start_pos = (grid_size-1, 0)  # 左下角
        goal_pos = (0, grid_size-1)   # 右上角
        forbidden_positions = [start_pos, goal_pos]
        
        # 根据地图大小调整障碍物数量
        if grid_size <= 7:  # Level 1
            num_obstacles = np.random.randint(2, 4)
            max_obstacle_size = 2  # 较小的障碍物尺寸
        elif grid_size <= 9:  # Level 2
            num_obstacles = np.random.randint(3, 5)
            max_obstacle_size = 3
        else:  # Level 3
            num_obstacles = np.random.randint(4, 7)
            max_obstacle_size = 3
        
        # 添加一些随机障碍物
        for _ in range(num_obstacles):
            obstacle_type = np.random.choice(['cross', 'l_shape', 'rectangle', 'line'])
            
            while True:
                if obstacle_type == 'cross':
                    center = (np.random.randint(1, grid_size-1), 
                            np.random.randint(1, grid_size-1))
                    new_obstacles = Obstacle.create_cross(center)
                
                elif obstacle_type == 'l_shape':
                    corner = (np.random.randint(0, grid_size-1),
                            np.random.randint(0, grid_size-1))
                    orientation = np.random.randint(0, 4)
                    new_obstacles = Obstacle.create_l_shape(corner, orientation)
                
                elif obstacle_type == 'rectangle':
                    top_left = (np.random.randint(0, grid_size-2),
                              np.random.randint(0, grid_size-2))
                    width = np.random.randint(1, min(max_obstacle_size, grid_size-top_left[1]))
                    height = np.random.randint(1, min(max_obstacle_size, grid_size-top_left[0]))
                    new_obstacles = Obstacle.create_rectangle(top_left, width, height)
                
                else:  # line
                    start = (np.random.randint(0, grid_size-max_obstacle_size),
                           np.random.randint(0, grid_size-max_obstacle_size))
                    length = np.random.randint(2, min(max_obstacle_size+1, grid_size-max(start)))
                    horizontal = np.random.choice([True, False])
                    new_obstacles = Obstacle.create_line(start, length, horizontal)
                
                # 验证新障碍物的位置
                if (Obstacle.validate_positions(new_obstacles, grid_size) and
                    not any(pos in forbidden_positions for pos in new_obstacles) and
                    not any(pos in obstacles for pos in new_obstacles)):
                    obstacles.extend(new_obstacles)
                    break
        
        return obstacles 