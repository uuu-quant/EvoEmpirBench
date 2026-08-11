# 游戏配置参数

# 地图大小
GRID_SIZES = {
    "Level 1": 7,
    "Level 2": 9,
    "Level 3": 11,
}
GRID_SIZE = 9  # 默认大小，会根据游戏模式动态设置
CELL_SIZE = 60  # 像素单位

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 215, 0)  # 金币颜色
HEART_COLOR = (255, 105, 180)  # 生命值颜色
PURPLE = (128, 0, 128)  # 怪物颜色
MENU_BG = (245, 245, 245)  # 菜单背景色
BUTTON_COLOR = (70, 130, 180)  # 按钮颜色
BUTTON_HOVER_COLOR = (100, 149, 237)  # 按钮悬停颜色
FOG_OF_WAR = (0, 0, 0)  # 战争迷雾（完全不可见）
BROWN = (139, 69, 19)  # 铲子颜色
SILVER = (192, 192, 192)  # 剑颜色
CYAN = (0, 255, 255)  # 磁铁颜色
GOLD = (255, 165, 0)  # 钥匙颜色

# 游戏元素
EMPTY = 0
OBSTACLE = 1
START = 2
GOAL = 3
AGENT = 4
COIN = 5
MONSTER = 6
SHOVEL = 7  # 铲子
SWORD = 8   # 剑
MAGNET = 9  # 磁铁
KEY = 10    # 钥匙

# 视野设置
VISION_RANGE = 1  # 视野范围（1表示3x3，因为是从中心向四周延伸1格）
NOT_DISCOVERED = 0  # 未探索
DISCOVERED = 1     # 已探索

# 游戏设置
INITIAL_LIVES = 3  # 初始生命值
COINS_COUNT = 5    # 每关金币数量
COIN_VALUE = 500   # 每个金币的分值
MIN_MONSTERS = 2   # 最少怪物数量
MAX_MONSTERS = 2   # 最多怪物数量

# 道具设置
SHOVEL_USES = 3    # 铲子可使用次数
MAGNET_RANGE = 2   # 磁铁吸取范围

# 游戏模式
MODE_LEVEL1 = "Level 1"
MODE_LEVEL2 = "Level 2"
MODE_LEVEL3 = "Level 3"

# 起点和终点位置 - 会根据地图大小动态调整
START_POS = None  # 左下角
GOAL_POS = None   # 右上角

# 移动配置
DIRECTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# 移动步数范围
MIN_STEPS = 1
MAX_STEPS = 3

# 窗口大小 - 会根据地图大小动态调整
WINDOW_SIZE = None  # 增加40像素显示状态栏
MENU_WINDOW_SIZE = (400, 350)  # 主菜单窗口大小

# 按钮配置
BUTTON_SIZE = (200, 50)
BUTTON_MARGIN = 20

# 初始化随模式变化的配置
def update_grid_config(mode, grid_size=None):
    """根据游戏模式更新网格配置"""
    global GRID_SIZE, START_POS, GOAL_POS, WINDOW_SIZE
    
    # 设置网格大小
    GRID_SIZE = grid_size or GRID_SIZES.get(mode, 9)
    
    # 更新起点和终点位置
    START_POS = (GRID_SIZE-1, 0)  # 左下角
    GOAL_POS = (0, GRID_SIZE-1)   # 右上角
    
    # 更新窗口大小
    WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 40)
    
    return GRID_SIZE, START_POS, GOAL_POS, WINDOW_SIZE

# 初始化配置
update_grid_config(MODE_LEVEL2)  # 默认为Level 2配置 
