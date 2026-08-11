import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
from typing import Tuple, Optional, Dict, Any, List, Set
import random
import time

from src.config.game_config import *
from src.game.obstacles import Obstacle

def _apply_grid_config(mode, grid_size=None):
    """Sync dynamic grid config into this module's imported constants."""
    global GRID_SIZE, START_POS, GOAL_POS, WINDOW_SIZE
    GRID_SIZE, START_POS, GOAL_POS, WINDOW_SIZE = update_grid_config(mode, grid_size)

class Monster:
    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos
        self.next_pos = pos  # 添加下一步位置属性
        self.is_moving = False  # 添加移动状态
    
    def plan_move(self, grid: np.ndarray) -> bool:
        """计划下一步移动，返回是否有可用的移动"""
        possible_moves = []
        for dx, dy in DIRECTIONS.values():
            new_x = self.pos[0] + dx
            new_y = self.pos[1] + dy
            if (0 <= new_x < GRID_SIZE and 
                0 <= new_y < GRID_SIZE and 
                grid[new_x, new_y] != OBSTACLE and
                grid[new_x, new_y] != MONSTER):
                possible_moves.append((new_x, new_y))
        
        if possible_moves:
            self.next_pos = random.choice(possible_moves)
            self.is_moving = True
            return True
        return False
    
    def complete_move(self):
        """完成移动"""
        self.pos = self.next_pos
        self.is_moving = False

class PathFindingEnv(gym.Env):
    def __init__(self, mode=MODE_LEVEL2):
        super().__init__()
        self.mode = mode
        self.map_index = 0  # 添加地图编号属性
        
        # 根据模式更新地图配置
        _apply_grid_config(mode)
        
        # 动作空间：4个方向 × 3种步长
        self.action_space = spaces.Discrete(12)  # 4 * 3 = 12种可能的动作
        
        # 观察空间：根据网格大小调整
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int32
        )
        
        self.grid = None
        self.agent_pos = None
        self.obstacles = None
        self.lives = INITIAL_LIVES
        self.score = 0
        self.coins = set()
        self.monsters = []
        self.vision_map = None  # 视野地图
        self.steps_count = 0  # 添加步数计数器
        
        # 添加动作历史
        self.action_history = []
        self.reward_history = []
        
        # 添加探索记录
        self.initial_discovered = 0
        
        # 添加道具状态
        self.has_shovel = False
        self.shovel_uses = 0
        self.has_sword = False
        self.has_magnet = False
        self.has_key = False
        
        # 道具位置
        self.items = {
            "shovel": None, 
            "sword": None, 
            "magnet": None, 
            "key": None
        }
        
        # 初始化pygame显示
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(f"寻路游戏 - {mode}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.shadow_surface = pygame.Surface(
            (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), 
            pygame.SRCALPHA
        )
    
    def _action_to_direction(self, action: int) -> Tuple[Tuple[int, int], int]:
        """将动作转换为方向和步数"""
        direction_idx = action // 3
        steps = (action % 3) + 1
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        return directions[direction_idx], steps
    
    def _is_valid_move(self, new_pos: Tuple[int, int]) -> bool:
        """检查移动是否有效"""
        x, y = new_pos
        # 检查是否在网格内
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        return True
    
    def _generate_coins(self):
        """生成金币"""
        self.coins = set()
        available_positions = [
            (i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
            if (i, j) not in self.obstacles and 
               (i, j) != START_POS and 
               (i, j) != GOAL_POS and 
               (i, j) != self.agent_pos
        ]
        
        if len(available_positions) >= COINS_COUNT:
            coin_positions = np.random.choice(
                len(available_positions), 
                COINS_COUNT, 
                replace=False
            )
            self.coins = {available_positions[i] for i in coin_positions}
            for coin_pos in self.coins:
                self.grid[coin_pos] = COIN
    
    def _generate_monsters(self):
        """生成怪物"""
        if self.mode == MODE_LEVEL1:
            return
        
        self.monsters = []
        available_positions = [
            (i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
            if (i, j) not in self.obstacles and 
               (i, j) != START_POS and 
               (i, j) != GOAL_POS and 
               (i, j) != self.agent_pos and
               (i, j) not in self.coins
        ]
        
        num_monsters = random.randint(MIN_MONSTERS, MAX_MONSTERS)
        if len(available_positions) >= num_monsters:
            monster_positions = random.sample(available_positions, num_monsters)
            self.monsters = [Monster(pos) for pos in monster_positions]
            for monster in self.monsters:
                self.grid[monster.pos] = MONSTER
    
    def _move_monsters(self):
        """移动所有怪物"""
        if self.mode == MODE_LEVEL1:
            return
        
        # 清除旧的怪物位置
        for monster in self.monsters:
            self.grid[monster.pos] = EMPTY
            
        # 为每个怪物规划移动
        for monster in self.monsters:
            if not monster.is_moving:  # 如果怪物不在移动中，规划新的移动
                monster.plan_move(self.grid)
            else:  # 如果怪物正在移动，完成移动
                monster.complete_move()
                monster.is_moving = False
        
        # 更新怪物位置
        for monster in self.monsters:
            self.grid[monster.pos] = MONSTER
    
    def _check_monster_collision(self) -> bool:
        """检查是否与怪物碰撞（包括移动路径）"""
        # 检查当前位置是否与怪物重叠
        if self.grid[self.agent_pos] == MONSTER:
            # 如果有剑，可以击杀怪物
            if self.has_sword:
                # 找到对应的怪物并移除
                for i, monster in enumerate(self.monsters):
                    if monster.pos == self.agent_pos:
                        self.monsters.pop(i)
                        self.grid[self.agent_pos] = EMPTY  # 移除怪物
                        return False  # 不算作碰撞
            return True
        
        # 检查是否与怪物的移动路径重叠
        for monster in self.monsters:
            if monster.is_moving:
                # 如果玩家位置与怪物的当前位置或下一个位置重叠
                if self.agent_pos == monster.pos or self.agent_pos == monster.next_pos:
                    if self.has_sword:
                        # 找到对应的怪物并移除
                        self.monsters.remove(monster)
                        self.grid[monster.pos] = EMPTY  # 移除怪物
                        return False  # 不算作碰撞
                    return True
        
        return False
    
    def _collect_nearby_coins(self):
        """使用磁铁收集附近的金币"""
        if not self.has_magnet:
            return 0
        
        x, y = self.agent_pos
        coin_reward = 0
        
        # 检查磁铁范围内的所有位置
        for i in range(max(0, x - MAGNET_RANGE), min(GRID_SIZE, x + MAGNET_RANGE + 1)):
            for j in range(max(0, y - MAGNET_RANGE), min(GRID_SIZE, y + MAGNET_RANGE + 1)):
                pos = (i, j)
                if pos in self.coins:
                    self.coins.remove(pos)
                    self.grid[pos] = EMPTY
                    coin_reward += COIN_VALUE
        
        return coin_reward
    
    def _update_vision(self):
        """更新视野地图，将当前视野范围内的区域标记为已探索"""
        x, y = self.agent_pos
        for i in range(max(0, x - VISION_RANGE), min(GRID_SIZE, x + VISION_RANGE + 1)):
            for j in range(max(0, y - VISION_RANGE), min(GRID_SIZE, y + VISION_RANGE + 1)):
                self.vision_map[i, j] = DISCOVERED
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # 初始化网格
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        
        # 初始化视野地图
        self.vision_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        
        # 重置动作和奖励历史
        self.action_history = []
        self.reward_history = []
        
        # 设置起点和终点
        self.grid[START_POS] = START
        self.grid[GOAL_POS] = GOAL
        
        # 放置障碍物
        self.obstacles = Obstacle.get_random_obstacle_layout(GRID_SIZE)
        for obs_pos in self.obstacles:
            self.grid[obs_pos] = OBSTACLE
        
        # 设置智能体初始位置
        self.agent_pos = START_POS
        
        # 重置生命值和分数
        self.lives = INITIAL_LIVES
        self.score = 0
        
        # 重置道具状态
        self.has_shovel = False
        self.shovel_uses = 0
        self.has_sword = False
        self.has_magnet = False
        self.has_key = False
        self.items = {"shovel": None, "sword": None, "magnet": None, "key": None}
        
        # 生成金币
        self._generate_coins()
        
        # 生成怪物（仅在Level 2和Level 3模式）
        self._generate_monsters()
        
        # 更新初始视野
        self._update_vision()
        
        # 记录初始已探索区域
        self.initial_discovered = np.sum(self.vision_map == DISCOVERED)
        
        # 重置步数计数器
        self.steps_count = 0
        
        return self.grid.copy(), {}
    
    def _compute_exploration_rate(self, env) -> float:
        """计算探索率"""
        grid_size = env.grid.shape[0]
        total_cells = grid_size * grid_size
        obstacles_count = len(env.obstacles)
        explorable_cells = total_cells - obstacles_count
        
        # 统计已探索的单元格数量
        discovered_count = np.sum(env.vision_map == DISCOVERED)
        
        # 确保不会超过总可探索单元格数
        discovered_count = min(discovered_count, explorable_cells)
        
        # 计算探索率
        exploration_rate = discovered_count / explorable_cells if explorable_cells > 0 else 0
        
        # 确保探索率不超过1
        exploration_rate = min(exploration_rate, 1.0)
        
        return exploration_rate
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 增加步数计数 - 无论动作是否有效都计算
        self.steps_count += 1
        
        direction, steps = self._action_to_direction(action)
        
        # 记录动作
        self.action_history.append(action)
        
        # 记录探索前的状态
        previously_discovered = np.sum(self.vision_map == DISCOVERED)
        
        # 计算新位置
        new_x = self.agent_pos[0] + direction[0] * steps
        new_y = self.agent_pos[1] + direction[1] * steps
        new_pos = (new_x, new_y)
        
        # 检查移动是否有效
        if not self._is_valid_move(new_pos):
            self.reward_history.append(-50)  # 移动惩罚
            return self.grid.copy(), -50, False, False, {}
        
        # 检查是否撞到障碍物
        if self.grid[new_x, new_y] == OBSTACLE:
            # 如果有铲子，可以破坏障碍物
            if self.has_shovel and self.shovel_uses > 0:
                self.grid[new_x, new_y] = EMPTY
                self.obstacles.remove((new_x, new_y))
                self.shovel_uses -= 1
                if self.shovel_uses <= 0:
                    self.has_shovel = False
                obstacle_reward = 500  # 正确使用铲子破坏障碍物奖励500分
            else:
                self.lives -= 1
                life_loss_penalty = -1000  # 损失一条命扣1000分
                self.agent_pos = START_POS  # 回到起点
                self._update_vision()  # 更新视野
                self.reward_history.append(life_loss_penalty)  # 生命值损失惩罚
                return self.grid.copy(), life_loss_penalty, self.lives <= 0, False, {}
        else:
            obstacle_reward = 0  # 没有破坏障碍物
        
        # 更新位置
        self.agent_pos = new_pos
        
        # 检查道具收集
        item_reward = 0
        if self.mode == MODE_LEVEL3:
            # 检查是否获得铲子
            if self.grid[new_pos] == SHOVEL:
                self.has_shovel = True
                self.shovel_uses = SHOVEL_USES
                self.grid[new_pos] = EMPTY
                item_reward += 100  # 获得道具奖励
            
            # 检查是否获得剑
            elif self.grid[new_pos] == SWORD:
                self.has_sword = True
                self.grid[new_pos] = EMPTY
                item_reward += 200  # 获得道具奖励
            
            # 检查是否获得磁铁
            elif self.grid[new_pos] == MAGNET:
                self.has_magnet = True
                self.grid[new_pos] = EMPTY
                item_reward += 150  # 获得道具奖励
            
            # 检查是否获得钥匙
            elif self.grid[new_pos] == KEY:
                self.has_key = True
                self.grid[new_pos] = EMPTY
                item_reward += 300  # 获得道具奖励
        
        # 更新视野
        self._update_vision()
        
        # 计算新探索的区域
        current_discovered = np.sum(self.vision_map == DISCOVERED)
        newly_discovered = current_discovered - previously_discovered
        
        # 确保新探索的区域不会超过地图上未探索的区域
        grid_size = self.grid.shape[0]
        total_cells = grid_size * grid_size
        obstacles_count = len(self.obstacles)
        explorable_cells = total_cells - obstacles_count
        
        # 确保总已探索区域不超过可探索区域
        if current_discovered > explorable_cells:
            # 如果超出了可探索区域，调整新发现的区域数量
            adjustment = current_discovered - explorable_cells
            newly_discovered = max(0, newly_discovered - adjustment)
        
        # 限制新探索区域不为负数
        newly_discovered = max(0, newly_discovered)
        
        exploration_reward = newly_discovered * 10  # 每探索一个新区域奖励10分
        
        # 如果有磁铁，收集附近的金币
        magnet_reward = self._collect_nearby_coins() if self.has_magnet else 0
        
        # 移动怪物（在玩家移动后）
        self._move_monsters()
        
        # 检查是否与怪物碰撞
        monster_kill_reward = 0  # 初始化怪物击杀奖励
        if self._check_monster_collision():
            if self.has_sword:
                # 使用剑击杀怪物奖励
                monster_kill_reward = 500  
                reward = monster_kill_reward - 50  # 移动惩罚
            else:
                self.lives -= 1
                life_loss_penalty = -1000  # 损失一条命扣1000分
                self.agent_pos = START_POS  # 回到起点
                self._update_vision()  # 更新视野
                self.reward_history.append(life_loss_penalty)  # 生命值损失惩罚
                return self.grid.copy(), life_loss_penalty, self.lives <= 0, False, {}
        
        # 检查是否吃到金币
        coin_reward = 0
        if new_pos in self.coins:
            self.coins.remove(new_pos)
            self.grid[new_x, new_y] = EMPTY
            coin_reward = COIN_VALUE  # 金币奖励500分
        
        # 计算总奖励
        reward = (exploration_reward + 
                 coin_reward + 
                 magnet_reward + 
                 obstacle_reward + 
                 monster_kill_reward + 
                 item_reward - 
                 50)  # 基础移动成本
        
        done = False
        
        # 检查是否到达目标
        if self.agent_pos == GOAL_POS:
            # 在Level 3中，需要钥匙才能通关
            if self.mode == MODE_LEVEL3 and not self.has_key:
                # 不扣分，只是不让进入终点
                self.agent_pos = (new_x - direction[0], new_y - direction[1])  # 返回原位置
                reward = -50  # 只扣移动的基础消耗
            else:
                # 通关奖励2000分
                goal_reward = 2000
                reward += goal_reward  # 确保是累加而不是替换
                done = True
        
        # 检查是否失败（生命值耗尽）
        if self.lives <= 0:
            done = True
            # 移除额外的失败惩罚，因为每损失一条命都已经扣了1000分了
        
        self.reward_history.append(reward)
        self.score += reward  # 更新总分
        
        return self.grid.copy(), reward, done, False, {}
    
    def render(self):
        """渲染当前游戏状态"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill(WHITE)
        
        # 绘制网格线
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, BLACK, (i * CELL_SIZE, 0), 
                           (i * CELL_SIZE, WINDOW_SIZE[1]-40))
            pygame.draw.line(self.screen, BLACK, (0, i * CELL_SIZE), 
                           (WINDOW_SIZE[0], i * CELL_SIZE))
        
        # 绘制游戏元素
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # 终点始终可见
                if (i, j) == GOAL_POS:
                    pygame.draw.rect(self.screen, RED, rect)
                    continue
                
                if self.vision_map[i, j] == NOT_DISCOVERED:
                    # 绘制黑色方块表示未探索区域
                    pygame.draw.rect(self.screen, FOG_OF_WAR, rect)
                else:
                    # 绘制已探索区域的游戏元素
                    if self.grid[i, j] == OBSTACLE:
                        pygame.draw.rect(self.screen, GRAY, rect)
                    elif (i, j) == START_POS:
                        pygame.draw.rect(self.screen, GREEN, rect)
                    elif (i, j) in self.coins:
                        # 绘制金币（圆形）
                        center = (j * CELL_SIZE + CELL_SIZE//2, i * CELL_SIZE + CELL_SIZE//2)
                        pygame.draw.circle(self.screen, YELLOW, center, CELL_SIZE//3)
                    elif self.grid[i, j] == MONSTER:
                        # 绘制怪物（菱形）
                        center = (j * CELL_SIZE + CELL_SIZE//2, i * CELL_SIZE + CELL_SIZE//2)
                        size = CELL_SIZE//3
                        points = [
                            (center[0], center[1] - size),  # 上
                            (center[0] + size, center[1]),  # 右
                            (center[0], center[1] + size),  # 下
                            (center[0] - size, center[1])   # 左
                        ]
                        pygame.draw.polygon(self.screen, PURPLE, points)
                        
                        # 如果怪物正在移动，绘制移动路径
                        for monster in self.monsters:
                            if monster.pos == (i, j) and monster.is_moving:
                                next_center = (
                                    monster.next_pos[1] * CELL_SIZE + CELL_SIZE//2,
                                    monster.next_pos[0] * CELL_SIZE + CELL_SIZE//2
                                )
                                # 只有当下一个位置也在已探索区域时才绘制路径
                                if self.vision_map[monster.next_pos[0], monster.next_pos[1]] == DISCOVERED:
                                    pygame.draw.line(
                                        self.screen,
                                        (*PURPLE, 128),
                                        center,
                                        next_center,
                                        3
                                    )
                    elif self.grid[i, j] == SHOVEL:
                        # 绘制铲子（棕色方块）
                        pygame.draw.rect(self.screen, BROWN, rect)
                    elif self.grid[i, j] == SWORD:
                        # 绘制剑（银色方块）
                        pygame.draw.rect(self.screen, SILVER, rect)
                    elif self.grid[i, j] == MAGNET:
                        # 绘制磁铁（青色方块）
                        pygame.draw.rect(self.screen, CYAN, rect)
                    elif self.grid[i, j] == KEY:
                        # 绘制钥匙（金色方块）
                        pygame.draw.rect(self.screen, GOLD, rect)
        
        # 绘制智能体（始终可见）
        agent_rect = pygame.Rect(
            self.agent_pos[1] * CELL_SIZE, 
            self.agent_pos[0] * CELL_SIZE, 
            CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect)
        
        # 绘制状态栏
        status_rect = pygame.Rect(0, WINDOW_SIZE[1]-40, WINDOW_SIZE[0], 40)
        pygame.draw.rect(self.screen, WHITE, status_rect)
        
        # 绘制生命值
        for i in range(self.lives):
            heart_pos = (10 + i * 30, WINDOW_SIZE[1]-30)
            self._draw_heart(heart_pos)
        
        # 绘制分数和地图信息
        score_text = self.font.render(f"分数: {self.score}", True, BLACK)
        map_text = self.font.render(f"{self.mode} - 地图 {self.map_index + 1}", True, BLACK)
        
        self.screen.blit(score_text, (WINDOW_SIZE[0]-150, WINDOW_SIZE[1]-35))
        self.screen.blit(map_text, (WINDOW_SIZE[0]//2-100, WINDOW_SIZE[1]-35))
        
        # 绘制道具状态
        if self.mode == MODE_LEVEL3:
            # 计算道具状态字符串的起始位置
            start_x = 10
            y_pos = 10
            
            # 绘制道具状态
            if self.has_shovel:
                shovel_text = self.font.render(f"铲子: {self.shovel_uses}次", True, BROWN)
                self.screen.blit(shovel_text, (start_x, y_pos))
                start_x += 120
            
            if self.has_sword:
                sword_text = self.font.render("剑: 已装备", True, SILVER)
                self.screen.blit(sword_text, (start_x, y_pos))
                start_x += 120
            
            if self.has_magnet:
                magnet_text = self.font.render("磁铁: 已装备", True, CYAN)
                self.screen.blit(magnet_text, (start_x, y_pos))
                start_x += 150
            
            if self.has_key:
                key_text = self.font.render("钥匙: 已获得", True, GOLD)
                self.screen.blit(key_text, (start_x, y_pos))
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def _draw_heart(self, pos):
        """绘制心形生命值"""
        x, y = pos
        size = 20
        points = [
            (x, y-size//4),
            (x-size//3, y-size//2),
            (x-size//2, y-size//3),
            (x-size//2, y),
            (x, y+size//3),
            (x+size//2, y),
            (x+size//2, y-size//3),
            (x+size//3, y-size//2),
        ]
        pygame.draw.polygon(self.screen, HEART_COLOR, points)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def load_map(self, map_data: Dict[str, Any], map_index: int = 0):
        """加载预生成的地图"""
        # 设置地图尺寸
        if "grid_size" in map_data:
            _apply_grid_config(self.mode, map_data["grid_size"])
        else:
            # 根据模式设置地图尺寸
            _apply_grid_config(self.mode)

        self.observation_space = spaces.Box(
            low=0, high=10, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int32
        )
        
        # 更新屏幕大小
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.shadow_surface = pygame.Surface(
            (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), 
            pygame.SRCALPHA
        )
        
        # 加载地图数据
        self.grid = np.array(map_data['grid'])
        self.obstacles = set(tuple(pos) for pos in map_data['obstacles'])
        self.coins = set(tuple(pos) for pos in map_data['coins'])
        self.monsters = [Monster(tuple(pos)) for pos in map_data['monsters']]
        self.map_index = map_index  # 保存地图编号
        
        # 加载道具位置
        self.items = {"shovel": None, "sword": None, "magnet": None, "key": None}
        if "items" in map_data:
            for item_name, pos in map_data["items"].items():
                if pos:
                    self.items[item_name] = tuple(pos)
        
        # 重置智能体位置和状态
        self.agent_pos = START_POS
        self.lives = INITIAL_LIVES
        self.score = 0
        self.steps_count = 0  # 重置步数计数
        
        # 重置道具状态
        self.has_shovel = False
        self.shovel_uses = 0
        self.has_sword = False
        self.has_magnet = False
        self.has_key = False
        
        # 重置动作和奖励历史
        self.action_history = []
        self.reward_history = []
        
        # 重置视野地图
        self.vision_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self._update_vision()
        
        # 记录初始已探索区域
        self.initial_discovered = np.sum(self.vision_map == DISCOVERED)
        
        return self.grid.copy(), {}
    
    def load_map_from_dict(self, map_data: Dict[str, Any]):
        """从字典加载地图，为了与train_agent.py兼容"""
        return self.load_map(map_data, map_data.get('map_index', 0))
    
    def get_action_meaning(self, action: int) -> str:
        """获取动作的含义描述"""
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        steps = [1, 2, 3]
        
        direction_idx = action // 3
        step_idx = action % 3
        
        if direction_idx < 4:
            return f"Move {directions[direction_idx]} {steps[step_idx]} step{'s' if steps[step_idx] > 1 else ''}"
        else:
            return "Unknown action"
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取当前状态的字典表示"""
        return {
            "grid": self.grid.copy(),
            "vision_map": self.vision_map.copy(),
            "agent_pos": self.agent_pos,
            "lives": self.lives,
            "score": self.score,
            "coins": self.coins.copy(),
            "monsters": [monster.pos for monster in self.monsters],
            "action_history": self.action_history.copy(),
            "reward_history": self.reward_history.copy(),
            "has_shovel": self.has_shovel,
            "shovel_uses": self.shovel_uses,
            "has_sword": self.has_sword,
            "has_magnet": self.has_magnet,
            "has_key": self.has_key,
        } 

    def end_session(self, env, success: bool, score: int) -> Dict[str, Any]:
        """
        结束游戏会话并更新指标
        
        Args:
            env: 游戏环境
            success: 是否成功通关
            score: 最终得分
            
        Returns:
            会话指标
        """
        # 正确计算探索率
        exploration_rate = self._compute_exploration_rate(env)
        
        # 更新最终指标
        self.session_metrics.update({
            "end_time": time.time(),
            "duration": time.time() - self.session_metrics["start_time"],
            "success": success,
            "score": score,
            "lives_remaining": env.lives,
            "collected_coins": len(env.coins),
            "exploration_rate": exploration_rate,  # 使用修正后的探索率计算
            "killed_monsters": sum(1 for log in self.session_logs if "killed_monsters" in log and log["killed_monsters"] > 0),
            "destroyed_obstacles": sum(1 for log in self.session_logs if "destroyed_obstacles" in log and log["destroyed_obstacles"] > 0)
        })
        
        print(f"结束关卡 {self.current_level_id} 的游戏会话")
        print(f"总步数: {self.session_metrics['steps']}")
        print(f"结果: {'成功' if success else '失败'}")
        print(f"得分: {score}")
        
        # 保存会话日志和指标
        self._save_session_data()
        
        return self.session_metrics 
