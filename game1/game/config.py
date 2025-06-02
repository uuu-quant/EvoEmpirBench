"""
Configuration constants for the maze navigation game.
"""

# Game modes
MODE_LEVEL1 = "Level 1"
MODE_LEVEL2 = "Level 2"
MODE_LEVEL3 = "Level 3"

# Grid cell types
EMPTY = 0      # Empty cell
OBSTACLE = 1   # Wall/obstacle
COIN = 2       # Collectible coin
MONSTER = 3    # Enemy that moves randomly
GOAL = 4       # Goal position
SHOVEL = 7     # Item to break walls
SWORD = 8      # Item to defeat monsters
MAGNET = 9     # Item to attract coins
KEY = 10       # Item required to enter goal

# Grid size for all levels
GRID_SIZE = 9

# Game parameters
LIVES_DEFAULT = 3
COINS_COUNT = 5
MONSTERS_COUNT = {MODE_LEVEL1: 0, MODE_LEVEL2: 3, MODE_LEVEL3: 4}
MONSTER_MOVE_PROB = 0.7  # Probability of monster movement per step

# Vision map values
NOT_DISCOVERED = 0   # Cell not yet seen
DISCOVERED = 1       # Cell has been seen

# Standard positions
START_POS = (0, 0)   # Player starting position
GOAL_POS = (8, 8)    # Goal position (bottom right)

# Reward values
REWARD_STEP = -50           # Penalty for each step
REWARD_COIN = 500           # Reward for collecting a coin
REWARD_GOAL = 2000          # Reward for reaching the goal
REWARD_LIFE_LOST = -1000    # Penalty for losing a life
REWARD_EXPLORATION = 10     # Reward for discovering a new cell

# Actions
# 0-2: UP (1/2/3 steps)
# 3-5: DOWN (1/2/3 steps)
# 6-8: LEFT (1/2/3 steps)
# 9-11: RIGHT (1/2/3 steps)

# Map generation parameters
MIN_PATH_LENGTH = 12
MAX_OBSTACLES_PERCENT = 0.3
MIN_EMPTY_AROUND_COIN = 1 