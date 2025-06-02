"""
Configuration constants for the match-3 puzzle game.
"""

# Game parameters
GRID_SIZE = 8
COLOR_KEYS = ['A', 'B', 'C', 'D']
COSTS = {'row': 32, 'col': 32, 'bomb': 12, 'hammer': 4}

# File paths (relative)
STATE_FILE = "game_state_{}.json"
ACTION_LOG_FILE = "action_log_{}.jsonl"
CLEAR_LOG_FILE = "clear_log_{}.jsonl"
AGENT_API_RESPONSE_FILE = "agent_api_responses_{}.jsonl"
STATS_FILE = "stats_{}.json"
LEVELS_DIR = "levels"

# Score parameters
BASE_SCORE_MULTIPLIER = 5
BONUS_MULTIPLIER = 3
MIN_COUNT = 2
AVG_CLEAR_PER_STEP = 6

# Difficulty levels
DIFFICULTY_LEVELS = {
    'easy': {'steps_base': 15, 'steps_range': (0, 3), 'target_range': (8, 12)},
    'medium': {'steps_base': 12, 'steps_range': (0, 3), 'target_range': (12, 16)},
    'hard': {'steps_base': 10, 'steps_range': (0, 3), 'target_range': (16, 20)}
}
LEVELS_PER_DIFFICULTY = 30 