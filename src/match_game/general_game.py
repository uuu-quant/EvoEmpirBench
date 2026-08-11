import random
import json
import time
from copy import deepcopy
import threading
import os
import sys
import re
from datetime import datetime
from math import ceil
import argparse
from openai import OpenAI
import glob

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config.paths import MATCH_LEVELS_DIR, RESULTS_DIR

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 游戏参数
GRID_SIZE = 8
COLOR_KEYS = ['A', 'B', 'C', 'D']
COSTS = {'row': 32, 'col': 32, 'bomb': 12, 'hammer': 4}
STATE_FILE = "game_state_{}.json"
ACTION_LOG_FILE = "action_log_{}.jsonl"
CLEAR_LOG_FILE = "clear_log_{}.jsonl"
AGENT_API_RESPONSE_FILE = "agent_api_responses_{}.jsonl"
STATS_FILE = "stats_{}.json"
LEVELS_DIR = str(MATCH_LEVELS_DIR)
BASE_SCORE_MULTIPLIER = 5
BONUS_MULTIPLIER = 3
MIN_COUNT = 2
AVG_CLEAR_PER_STEP = 6

# 难度级别定义
DIFFICULTY_LEVELS = {
    'easy': {'steps_base': 15, 'steps_range': (0, 3), 'target_range': (8, 12)},
    'medium': {'steps_base': 12, 'steps_range': (0, 3), 'target_range': (12, 16)},
    'hard': {'steps_base': 10, 'steps_range': (0, 3), 'target_range': (16, 20)}
}
LEVELS_PER_DIFFICULTY = 30

class GPTClient:
    def __init__(self, key=None, url=None, model="gpt-4"):
        openai_key = os.getenv("OPENAI_API_KEY")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.APIKEY = key or openai_key or deepseek_key
        if url:
            self.baseURL = url
        elif openai_key:
            self.baseURL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or DEFAULT_OPENAI_BASE_URL
        elif deepseek_key:
            self.baseURL = os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        else:
            self.baseURL = DEFAULT_OPENAI_BASE_URL
        if not self.APIKEY:
            raise ValueError(
                "Missing API key. Set OPENAI_API_KEY/DEEPSEEK_API_KEY in the environment "
                "or pass --api_key explicitly."
            )
        self.client = self.init_connection()

    def init_connection(self):
        return OpenAI(
            api_key=self.APIKEY,
            base_url=self.baseURL
        )

    def get_response(self, messages: list, model=None):
        if model:
            self.model = model
            
        # 检查是否是 Qwen 系列模型
        is_qwen = 'qwen4' in self.model.lower()
        
        try:
            if is_qwen:
                # 对于 Qwen 系列模型使用流式输出
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    stream=True
                )
                
                # 收集完整的响应
                collected_content = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        collected_content += chunk.choices[0].delta.content
                
                if "该请求" in collected_content or "sorry" in collected_content:
                    return 'Failed'
                return collected_content
            else:
                # 对于其他模型使用普通输出
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3
                )
                if "该请求" in completion.choices[0].message.content or "sorry" in completion.choices[0].message.content:
                    return 'Failed'
                return completion.choices[0].message.content
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return 'Failed'

    def infer_data(self, data):
        return self.get_response(data)

class MatchGame:
    def __init__(self, model_name, auto_run=False, api_key=None, base_url=None):
        self.model_name = model_name
        self.auto_run = auto_run
        # 初始化 GPTClient，使用命令行参数或默认值
        self.api_client = GPTClient(
            key=api_key,
            url=base_url,
            model=model_name
        )
        self.board = []
        self.score = 0
        self.steps_remaining = 0
        self.max_steps = 0
        self.steps_used = 0
        self.current_level = 0
        self.current_difficulty = 'easy'
        self.is_human_mode = True
        self.is_processing = False
        self.game_over = False
        self.inventory = {'row': 1, 'col': 1, 'bomb': 1, 'hammer': 1}
        self.selecting_prop = None
        self.levels = {'easy': [], 'medium': [], 'hard': []}
        self.level_select = True
        self.message = ""
        self.message_timeout = 0
        self.action_log = []
        self.color_targets = {}
        self.color_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.total_cleared = 0
        self.clear_counts = []
        self.session_dir = ""
        self.session_timestamp = ""
        self.api_response_count = 0
        self.valid_api_response_count = 0
        self.load_or_generate_levels()

    def has_complete_data(self, model_name, difficulty, level):
        """检查指定关卡是否已有完整的结果文件"""
        # 构建结果目录路径
        level_dir = os.path.join(str(RESULTS_DIR), model_name, "game2", 
                                difficulty, f"level{level:02d}", "agent")
        
        # 检查目录是否存在
        if not os.path.exists(level_dir):
            return False
        
        # 检查是否有完整的统计文件
        stats_files = glob.glob(os.path.join(level_dir, "stats_*.json"))
        if not stats_files:
            return False
        
        # 检查最新的统计文件是否包含完整数据
        try:
            latest_stats_file = sorted(stats_files)[-1]
            with open(latest_stats_file, 'r') as f:
                stats_data = json.load(f)
                
                # 检查是否包含关键字段表明关卡已评估完毕
                required_fields = ["total_score", "cleared", "steps_remaining", 
                                "avg_score_per_step", "avg_clear_per_step"]
                
                if all(field in stats_data for field in required_fields):
                    print(f"检测到已完成评估: {difficulty} 关卡 {level}, 模型: {model_name}")
                    return True
        except Exception as e:
            print(f"读取统计文件失败: {str(e)}")
            
        return False

    def generate_levels(self):
        """生成关卡并存储在 data/levels/match_game/<difficulty>/levelXX.json。"""
        for difficulty, params in DIFFICULTY_LEVELS.items():
            levels = []
            difficulty_dir = os.path.join(LEVELS_DIR, difficulty)
            os.makedirs(difficulty_dir, exist_ok=True)
            for i in range(1, LEVELS_PER_DIFFICULTY + 1):
                base_steps = params['steps_base'] + random.randint(*params['steps_range'])
                target_range = params['target_range']
                color_targets = {
                    'A': random.randint(*target_range),
                    'B': random.randint(*target_range),
                    'C': random.randint(*target_range),
                    'D': random.randint(*target_range)
                }
                total_targets = sum(color_targets.values())
                steps = max(base_steps, ceil(total_targets / AVG_CLEAR_PER_STEP))
                board = [[random.choice(COLOR_KEYS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                has_valid_moves = False
                while not has_valid_moves:
                    for k in range(GRID_SIZE):
                        for l in range(GRID_SIZE):
                            region = self.find_connected_region(k, l, board[k][l], [[False]*GRID_SIZE for _ in range(GRID_SIZE)], board)
                            if len(region) >= 2:
                                has_valid_moves = True
                    if not has_valid_moves:
                        board = [[random.choice(COLOR_KEYS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                level = {
                    'board': board,
                    'max_steps': steps,
                    'inventory': {'row': 1, 'col': 1, 'bomb': 1, 'hammer': 1},
                    'color_targets': color_targets
                }
                levels.append(level)
                level_path = os.path.join(difficulty_dir, f"level{i:02d}.json")
                with open(level_path, 'w', encoding='utf-8') as f:
                    json.dump(level, f, indent=2)
                print(f"Generated and saved {difficulty} level {i}: max_steps={steps}, color_targets={color_targets}")
            self.levels[difficulty] = levels

    def load_or_generate_levels(self):
        """加载或生成关卡，存储在 data/levels/match_game 目录。"""
        os.makedirs(LEVELS_DIR, exist_ok=True)
        for difficulty in DIFFICULTY_LEVELS:
            difficulty_dir = os.path.join(LEVELS_DIR, difficulty)
            os.makedirs(difficulty_dir, exist_ok=True)
            levels = []
            for i in range(1, LEVELS_PER_DIFFICULTY + 1):
                level_path = os.path.join(difficulty_dir, f"level{i:02d}.json")
                if os.path.exists(level_path):
                    with open(level_path, 'r', encoding='utf-8') as f:
                        level = json.load(f)
                    levels.append(level)
                    print(f"Loaded {difficulty} level {i}: max_steps={level['max_steps']}, color_targets={level['color_targets']}")
                else:
                    print(f"No existing levels found for {difficulty}, generating new levels")
                    self.generate_levels()
                    return
            self.levels[difficulty] = levels

    def init_game(self, difficulty, level):
        """初始化游戏，结果存储在 outputs/results/{model}/game2/{difficulty}/levelXX/agent/。"""
        self.current_difficulty = difficulty
        self.current_level = level
        self.board = deepcopy(self.levels[difficulty][level-1]['board'])
        self.max_steps = self.levels[difficulty][level-1]['max_steps']
        self.inventory = deepcopy(self.levels[difficulty][level-1]['inventory'])
        self.color_targets = deepcopy(self.levels[difficulty][level-1]['color_targets'])
        self.color_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.score = 0
        self.steps_remaining = self.max_steps
        self.steps_used = 0
        self.game_over = False
        self.is_processing = False
        self.selecting_prop = None
        self.level_select = False
        self.message = ""
        self.message_timeout = 0
        self.action_log = []
        self.total_cleared = 0
        self.clear_counts = []
        self.api_response_count = 0
        self.valid_api_response_count = 0
        mode = 'human' if self.is_human_mode else 'agent'
        self.session_timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        # 统一使用 outputs/results/{model}/game2 目录
        self.session_dir = os.path.join(str(RESULTS_DIR), 
                                      self.model_name, "game2",
                                      difficulty, f"level{level:02d}", mode)
            
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"Initialized {difficulty} level {level}: max_steps={self.max_steps}, color_targets={self.color_targets}, model={self.model_name}")
        self.save_state()

    def set_human_mode(self):
        if not self.game_over:
            self.is_human_mode = True
            self.selecting_prop = None
            print("Switched to human mode")

    def set_agent_mode(self):
        if not self.game_over:
            self.is_human_mode = False
            self.selecting_prop = None
            threading.Thread(target=self.agent_play, daemon=True).start()
            print("Switched to agent mode")

    def reset_game(self):
        self.init_game(self.current_difficulty, self.current_level)
        print("Game reset")

    def show_level_select(self):
        self.level_select = True
        self.current_level = 0
        self.current_difficulty = 'easy'
        self.game_over = False
        self.is_processing = False
        self.message = ""
        print("Show level selection")

    def display_board(self):
        print("\nBoard state:")
        print("  " + " ".join(str(j) for j in range(GRID_SIZE)))
        for i in range(GRID_SIZE):
            row = [self.board[i][j] if self.board[i][j] else '.' for j in range(GRID_SIZE)]
            print(f"{i} {' '.join(row)}")
        print(f"Mode: {'human' if self.is_human_mode else 'agent'}")
        print(f"Difficulty: {self.current_difficulty}, Level: {self.current_level}, Model: {self.model_name}")
        print(f"Score: {self.score}, Steps remaining: {self.steps_remaining}, Props: {self.inventory}")
        print(f"Color targets: {self.color_targets}")
        print(f"Cleared: {self.color_counts}")
        if self.message and time.time() < self.message_timeout:
            print(f"Message: {self.message}")
        if self.game_over:
            print("Game over!")

    def find_connected_region(self, i, j, color, visited, board_ref):
        if i < 0 or i >= GRID_SIZE or j < 0 or j >= GRID_SIZE or visited[i][j] or board_ref[i][j] != color:
            return []
        visited[i][j] = True
        region = [[i, j]]
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for di, dj in directions:
            region.extend(self.find_connected_region(i + di, j + dj, color, visited, board_ref))
        return region

    def shift_columns(self):
        for j in range(GRID_SIZE):
            empty = -1
            for i in range(GRID_SIZE - 1, -1, -1):
                if self.board[i][j] is None:
                    empty = i
                    break
            if empty >= 0:
                for i in range(empty, -1, -1):
                    if self.board[i][j] is not None:
                        self.board[empty][j] = self.board[i][j]
                        self.board[i][j] = None
                        empty -= 1
                for i in range(empty, -1, -1):
                    self.board[i][j] = random.choice(COLOR_KEYS)

    def simulate_shift_columns(self, temp_board):
        for j in range(GRID_SIZE):
            empty = -1
            for i in range(GRID_SIZE - 1, -1, -1):
                if temp_board[i][j] is None:
                    empty = i
                    break
            if empty >= 0:
                for i in range(empty, -1, -1):
                    if temp_board[i][j] is not None:
                        temp_board[empty][j] = temp_board[i][j]
                        temp_board[i][j] = None
                        empty -= 1
                for i in range(empty, -1, -1):
                    temp_board[i][j] = random.choice(COLOR_KEYS)
        return temp_board

    def save_state(self):
        """保存游戏状态到 outputs/results/{model}/game2/{difficulty}/levelXX/{mode}/game_state_{timestamp}.json"""
        state = {
            'board': self.board,
            'score': self.score,
            'steps_remaining': self.steps_remaining,
            'max_steps': self.max_steps,
            'steps_used': self.steps_used,
            'current_difficulty': self.current_difficulty,
            'current_level': self.current_level,
            'is_human_mode': self.is_human_mode,
            'game_over': self.game_over,
            'inventory': self.inventory,
            'level_select': self.level_select,
            'color_targets': self.color_targets,
            'color_counts': self.color_counts,
            'total_cleared': self.total_cleared,
            'clear_counts': self.clear_counts,
            'model_name': self.model_name
        }
        state_path = os.path.join(self.session_dir, STATE_FILE.format(self.session_timestamp))
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        print(f"Game state saved to {state_path}")

    def save_clear_log(self):
        """保存清除日志到 outputs/results/{model}/game2/{difficulty}/levelXX/{mode}/clear_log_{timestamp}.jsonl"""
        clear_log = {
            'clear_counts': self.clear_counts,
            'avg_clear_per_step': self.total_cleared / self.steps_used if self.steps_used > 0 else 0,
            'model_name': self.model_name
        }
        clear_log_path = os.path.join(self.session_dir, CLEAR_LOG_FILE.format(self.session_timestamp))
        with open(clear_log_path, 'w', encoding='utf-8') as f:
            json.dump(clear_log, f, indent=2)
        print(f"Clear log saved to {clear_log_path}")

    def log_action(self, actor, action_type, pos=None, index=None):
        """记录操作到 outputs/results/{model}/game2/<difficulty>/levelXX/<mode>/action_log_<timestamp>.jsonl"""
        action = {
            'actor': actor,
            'type': action_type,
            'pos': pos,
            'index': index,
            'timestamp': time.time(),
            'difficulty': self.current_difficulty,
            'level': self.current_level,
            'score': self.score,
            'steps_remaining': self.steps_remaining,
            'color_counts': self.color_counts,
            'cleared': self.clear_counts[-1] if self.clear_counts else 0,
            'model_name': self.model_name
        }
        self.action_log.append(action)
        action_log_path = os.path.join(self.session_dir, ACTION_LOG_FILE.format(self.session_timestamp))
        with open(action_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(action, ensure_ascii=False) + '\n')
        print(f"Logged action: {actor} {action_type} {'pos=' + str(pos) if pos else 'index=' + str(index)}")

    def check_game_over(self):
        if self.steps_remaining <= 0:
            self.game_over = True
        targets_met = all(self.color_counts[color] >= self.color_targets[color] for color in COLOR_KEYS)
        if targets_met:
            self.game_over = True
            self.message = "Congratulations! All color targets achieved, level cleared!"
        elif self.steps_remaining <= 0 and not targets_met:
            self.message = "Steps exhausted, targets not met, game failed!"
        if self.game_over:
            self.is_processing = False
            avg_score_per_step = self.score / self.steps_used if self.steps_used > 0 else 0
            avg_clear_per_step = self.total_cleared / self.steps_used if self.steps_used > 0 else 0
            valid_api_response_ratio = (
                self.valid_api_response_count / self.api_response_count
                if self.api_response_count > 0 else 0.0
            )
            stats = {
                'total_score': self.score,
                'cleared': targets_met,
                'steps_remaining': self.steps_remaining,
                'avg_score_per_step': avg_score_per_step,
                'avg_clear_per_step': avg_clear_per_step,
                'api_response_count': self.api_response_count,
                'valid_api_response_count': self.valid_api_response_count,
                'valid_api_response_ratio': valid_api_response_ratio,
                'model_name': self.model_name
            }
            stats_path = os.path.join(self.session_dir, STATS_FILE.format(self.session_timestamp))
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats: Total score={self.score}, Cleared={targets_met}, "
                  f"Steps remaining={self.steps_remaining}, Avg score/step={avg_score_per_step:.2f}, "
                  f"Avg clear/step={avg_clear_per_step:.2f}, "
                  f"API responses={self.api_response_count}, Valid API responses={self.valid_api_response_count}, "
                  f"Valid API response ratio={valid_api_response_ratio:.2%}, Model={self.model_name}")
            print(f"Stats saved to {stats_path}")
            self.save_clear_log()
            self.save_state()

    def has_valid_moves(self, board_ref):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if board_ref[i][j]:
                    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
                    region = self.find_connected_region(i, j, board_ref[i][j], visited, board_ref)
                    if len(region) >= 2:
                        return True
        return False

    def can_props_create_moves(self):
        for prop, cost in COSTS.items():
            if self.score >= cost and self.inventory[prop] > 0:
                if prop == 'row':
                    for i in range(GRID_SIZE):
                        temp_board = deepcopy(self.board)
                        for j in range(GRID_SIZE):
                            temp_board[i][j] = None
                        temp_board = self.simulate_shift_columns(temp_board)
                        if self.has_valid_moves(temp_board):
                            return True
                elif prop == 'col':
                    for j in range(GRID_SIZE):
                        temp_board = deepcopy(self.board)
                        for i in range(GRID_SIZE):
                            temp_board[i][j] = None
                        temp_board = self.simulate_shift_columns(temp_board)
                        if self.has_valid_moves(temp_board):
                            return True
                elif prop == 'bomb':
                    for i in range(GRID_SIZE):
                        for j in range(GRID_SIZE):
                            temp_board = deepcopy(self.board)
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                                        temp_board[ni][nj] = None
                            temp_board = self.simulate_shift_columns(temp_board)
                            if self.has_valid_moves(temp_board):
                                return True
                elif prop == 'hammer':
                    for i in range(GRID_SIZE):
                        for j in range(GRID_SIZE):
                            if self.board[i][j] is not None:
                                temp_board = deepcopy(self.board)
                                temp_board[i][j] = None
                                temp_board = self.simulate_shift_columns(temp_board)
                                if self.has_valid_moves(temp_board):
                                    return True
        return False

    def select_prop(self, prop):
        if not self.is_human_mode or self.game_over or self.inventory[prop] == 0 or self.score < COSTS[prop]:
            self.message = "Insufficient score or no props available!"
            self.message_timeout = time.time() + 3
            print(f"Cannot select prop {prop}: insufficient score or no props")
            return False
        self.selecting_prop = prop
        print(f"Selected prop: {prop}")
        return True

    def clear_region(self, region):
        if len(region) < 2:
            print(f"Region size {len(region)} less than 2, cannot clear")
            return False
        print(f"Attempting to clear region: {region}")
        for i, j in region:
            color = self.board[i][j]
            if color:
                self.color_counts[color] += 1
            self.board[i][j] = None
        grid_count = len(region)
        self.score += grid_count * BASE_SCORE_MULTIPLIER + BONUS_MULTIPLIER * max(0, grid_count - MIN_COUNT)
        self.steps_remaining = max(0, self.steps_remaining - 1)
        self.steps_used += 1
        self.total_cleared += grid_count
        self.clear_counts.append(grid_count)
        self.shift_columns()
        self.check_game_over()
        print(f"Cleared region size {grid_count}, new score: {self.score}, color_counts: {self.color_counts}")
        self.save_state()
        return True

    def use_prop(self, prop, target):
        if self.score < COSTS[prop] or self.inventory[prop] == 0:
            self.message = "Insufficient score or no props available!"
            self.message_timeout = time.time() + 3
            print(f"Cannot use prop {prop}: insufficient score or no props")
            return False
        self.inventory[prop] -= 1
        self.score -= COSTS[prop]
        cleared_count = 0
        need_shift = False
        if prop == 'row':
            for j in range(GRID_SIZE):
                color = self.board[target][j]
                if color:
                    self.color_counts[color] += 1
                    cleared_count += 1
                self.board[target][j] = None
            self.steps_remaining = max(0, self.steps_remaining - 1)
            self.steps_used += 1
            need_shift = True
            print(f"Used row prop at row {target}")
        elif prop == 'col':
            for i in range(GRID_SIZE):
                color = self.board[i][target]
                if color:
                    self.color_counts[color] += 1
                    cleared_count += 1
                self.board[i][target] = None
            self.steps_remaining = max(0, self.steps_remaining - 1)
            self.steps_used += 1
            need_shift = True
            print(f"Used col prop at col {target}")
        elif prop == 'bomb':
            i, j = target
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                        color = self.board[ni][nj]
                        if color:
                            self.color_counts[color] += 1
                            cleared_count += 1
                        self.board[ni][nj] = None
            self.steps_remaining = max(0, self.steps_remaining - 1)
            self.steps_used += 1
            need_shift = True
            print(f"Used bomb prop at {target}")
        elif prop == 'hammer':
            i, j = target
            if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE and self.board[i][j] is not None:
                color = self.board[i][j]
                self.color_counts[color] += 1
                cleared_count += 1
                self.board[i][j] = None
                self.steps_remaining = max(0, self.steps_remaining - 1)
                self.steps_used += 1
                need_shift = True
                print(f"Used hammer prop at {target}")
            else:
                self.inventory[prop] += 1
                self.score += COSTS[prop]
                self.message = "Invalid grid!"
                self.message_timeout = time.time() + 3
                print(f"Cannot use hammer at {target}: invalid grid")
                return False
        if need_shift:
            self.total_cleared += cleared_count
            self.clear_counts.append(cleared_count)
            self.shift_columns()
        self.check_game_over()
        self.save_state()
        return True

    def call_agent_api(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                system_prompt = """
You are an AI assistant for an 8x8 match game (gridSize = 8). The board is an 8x8 grid with colors A, B, C, D, or null (empty). Rules:
- Eliminate ≥2 connected same-color tiles (horizontal/vertical), score = tiles * 5 + 3 * max(0, tiles - 2).
- Props (each usable once): row (clear row, 32 points), col (clear column, 32 points), bomb (clear 3x3 area, 12 points), hammer (clear 1 tile, 4 points).
- Each action costs 1 step. Goal: Clear the level by meeting color elimination targets (A, B, C, D) within limited steps while maximizing score and minimizing steps used.
- Primary objective: Ensure level completion by achieving all color targets.
- Secondary objectives: Maximize total score by prioritizing larger tile eliminations and efficient prop usage; minimize steps to preserve remaining steps.
- After elimination, new random tiles (A, B, C, D) fall from the top to fill empty spaces.
- Input: Board (8x8, A/B/C/D/null), score, steps remaining, inventory (row/col/bomb/hammer), color targets, current color counts.
- Output: Best action in JSON: {"action": {"type": "eliminate"|"row"|"col"|"bomb"|"hammer", "pos": [i,j] (for eliminate/bomb/hammer, 0≤i,j<8), "index": k (for row/col, 0≤k<8)}}.
- If no valid action, return {"action": null}.
"""
                user_prompt = f"""
Board:
{json.dumps(self.board, indent=2)}
Score: {self.score}
Steps remaining: {self.steps_remaining}
Inventory: {json.dumps(self.inventory)}
Color targets: {json.dumps(self.color_targets)}
Current color counts: {json.dumps(self.color_counts)}
Suggest the best action to achieve the level completion with maximum score and minimum steps.
"""
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
                content = self.api_client.get_response(messages)
                print(f"GPT API response (model={self.model_name}):", content)
                self.api_response_count += 1
                if content == 'Failed':
                    print(f"GPT API failed (model={self.model_name}, attempt={attempt+1})")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                if not self.is_human_mode:
                    response_log = {
                        'timestamp': time.time(),
                        'request': {
                            'model': self.model_name,
                            'messages': messages,
                            'stream': False
                        },
                        'response': {'content': content},
                        'model_name': self.model_name
                    }
                    response_path = os.path.join(self.session_dir, AGENT_API_RESPONSE_FILE.format(self.session_timestamp))
                    with open(response_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(response_log, ensure_ascii=False) + '\n')
                    print(f"API response saved to {response_path}")
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    action = json.loads(json_match.group(1).strip())
                    print(f"Agent selected (model={self.model_name}):", action)
                    return action
                else:
                    print(f"No valid JSON found in response (model={self.model_name})")
                    return None
            except Exception as e:
                print(f"GPT API error (model={self.model_name}, attempt={attempt+1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

    def agent_play(self):
        while not self.is_human_mode and not self.game_over and not self.is_processing:
            self.is_processing = True
            try:
                move = self.call_agent_api()
                if move and move.get('action'):
                    action = move['action']
                    type = action.get('type')
                    pos = action.get('pos')
                    index = action.get('index')
                    print(f"Agent action (model={self.model_name}): type={type}, pos={pos}, index={index}")
                    prev_steps = self.steps_remaining
                    success = False
                    if type == 'eliminate' and pos:
                        if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE and self.board[pos[0]][pos[1]]:
                            visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
                            region = self.find_connected_region(pos[0], pos[1], self.board[pos[0]][pos[1]], visited, self.board)
                            print(f"Found region: {region}")
                            if len(region) >= 2:
                                if self.clear_region(region):
                                    self.log_action("agent", type, pos)
                                    success = True
                                else:
                                    print(f"Agent action invalid (model={self.model_name}): {move}")
                            else:
                                print(f"Region size {len(region)} less than 2, cannot clear (model={self.model_name})")
                        else:
                            print(f"Invalid position (model={self.model_name}): pos={pos}")
                    elif type in ['row', 'col', 'bomb', 'hammer']:
                        if self.use_prop(type, index if type in ['row', 'col'] else pos):
                            self.log_action("agent", type, pos, index)
                            success = True
                        else:
                            print(f"Prop action invalid (model={self.model_name}): {move}")
                    else:
                        print(f"Unknown action type (model={self.model_name}): {type}")
                    if success and self.steps_remaining < prev_steps:
                        self.valid_api_response_count += 1
                else:
                    print(f"Agent returned no valid action (model={self.model_name})")
                time.sleep(1)
            except Exception as e:
                print(f"Agent execution error (model={self.model_name}): {str(e)}")
            finally:
                self.is_processing = False
                self.display_board()

    def run_all_levels(self):
        """自动运行所有难度的所有关卡（agent模式）"""
        total_levels = len(DIFFICULTY_LEVELS) * LEVELS_PER_DIFFICULTY
        completed_levels = 0
        
        for difficulty in DIFFICULTY_LEVELS:
            for level in range(1, LEVELS_PER_DIFFICULTY + 1):
                # 检查该关卡是否已有完整数据
                if self.has_complete_data(self.model_name, difficulty, level):
                    print(f"跳过已完成的 {difficulty} 关卡 {level}")
                    completed_levels += 1
                    continue
                
                print(f"\n运行 {difficulty} 关卡 {level} 的智能代理模式 ({completed_levels}/{total_levels} 已完成)")
                self.is_human_mode = False
                self.level_select = False
                self.init_game(difficulty, level)
                self.display_board()
                self.set_agent_mode()
                
                # 等待游戏结束
                try:
                    start_time = time.time()
                    while not self.game_over:
                        time.sleep(1)
                        # 增加超时保护，防止某个关卡卡住
                        if time.time() - start_time > 300:  # 5分钟超时
                            print(f"关卡 {difficulty} - {level} 执行超时，强制结束")
                            self.game_over = True
                            
                            # 保存当前状态
                            stats = {
                                'total_score': self.score,
                                'cleared': False,  # 因超时强制结束
                                'steps_remaining': self.steps_remaining,
                                'avg_score_per_step': self.score / self.steps_used if self.steps_used > 0 else 0,
                                'avg_clear_per_step': self.total_cleared / self.steps_used if self.steps_used > 0 else 0,
                                'api_response_count': self.api_response_count,
                                'valid_api_response_count': self.valid_api_response_count,
                                'valid_api_response_ratio': (
                                    self.valid_api_response_count / self.api_response_count
                                    if self.api_response_count > 0 else 0.0
                                ),
                                'model_name': self.model_name,
                                'timeout': True  # 标记为超时
                            }
                            stats_path = os.path.join(self.session_dir, STATS_FILE.format(self.session_timestamp))
                            with open(stats_path, 'w', encoding='utf-8') as f:
                                json.dump(stats, f, indent=2)
                            self.save_clear_log()
                            self.save_state()
                            break
                except KeyboardInterrupt:
                    print("\n检测到键盘中断，保存当前状态后退出")
                    if not self.game_over:
                        self.save_clear_log()
                        self.save_state()
                    return  # 退出整个评估过程
                
                completed_levels += 1
                print(f"完成 {difficulty} 关卡 {level} ({completed_levels}/{total_levels})")
        
        print(f"\n所有关卡评估完成！总共 {completed_levels}/{total_levels} 关卡。")

    def run(self):
        if self.auto_run:
            self.run_all_levels()
            return
        while True:
            if self.level_select:
                print("\nSelect difficulty (enter e for easy, m for medium, h for hard, q to quit):")
                for difficulty in DIFFICULTY_LEVELS:
                    print(f"Difficulty {difficulty}: {len(self.levels[difficulty])} levels available")
                difficulty_choice = input("> ").strip().lower()
                if difficulty_choice == 'q':
                    print("Exiting game")
                    break
                if difficulty_choice not in ['e', 'm', 'h']:
                    print("Invalid difficulty, please enter e, m, or h")
                    continue
                difficulty_map = {'e': 'easy', 'm': 'medium', 'h': 'hard'}
                self.current_difficulty = difficulty_map[difficulty_choice]
                print(f"\nSelect level in {self.current_difficulty} (enter 1-{LEVELS_PER_DIFFICULTY}):")
                for i in range(LEVELS_PER_DIFFICULTY):
                    print(f"Level {i+1}: max_steps={self.levels[self.current_difficulty][i]['max_steps']}, "
                          f"color_targets={self.levels[self.current_difficulty][i]['color_targets']}")
                level_choice = input("> ").strip()
                try:
                    level = int(level_choice)
                    if 1 <= level <= LEVELS_PER_DIFFICULTY:
                        print("\nSelect mode (enter h for human, a for agent):")
                        mode_choice = input("> ").strip().lower()
                        if mode_choice == 'h':
                            self.is_human_mode = True
                            self.init_game(self.current_difficulty, level)
                            self.display_board()
                        elif mode_choice == 'a':
                            self.is_human_mode = False
                            self.init_game(self.current_difficulty, level)
                            self.display_board()
                            threading.Thread(target=self.agent_play, daemon=True).start()
                        else:
                            print("Invalid mode, please enter h or a")
                            continue
                    else:
                        print(f"Invalid level, please enter 1-{LEVELS_PER_DIFFICULTY}")
                except ValueError:
                    print("Invalid input, please enter a number or q")
            else:
                if self.game_over:
                    print("Game over, exiting")
                    break
                if self.is_human_mode:
                    print("\nOperations (enter format below, q to quit):")
                    print("- Eliminate: e i j (e.g., e 0 0)")
                    print("- Prop: p <row|col|bomb|hammer> [i [j]] (e.g., p row 0, p bomb 0 0)")
                    print("- Mode: m <human|agent>")
                    print("- Reset: r")
                    print("- Select level: s")
                    choice = input("> ").strip().lower()
                    if choice == 'q':
                        print("Exiting game")
                        break
                    try:
                        parts = choice.split()
                        if not parts:
                            print("Invalid input")
                            continue
                        cmd = parts[0]
                        if cmd == 'e' and len(parts) == 3:
                            i, j = int(parts[1]), int(parts[2])
                            if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE and self.board[i][j]:
                                visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
                                region = self.find_connected_region(i, j, self.board[i][j], visited, self.board)
                                if self.clear_region(region):
                                    self.log_action("human", "eliminate", [i, j])
                                else:
                                    print("Invalid elimination, at least 2 same-color tiles required")
                            else:
                                print("Invalid coordinates or grid")
                        elif cmd == 'p' and len(parts) >= 2:
                            prop = parts[1]
                            if prop in ['row', 'col'] and len(parts) == 3:
                                index = int(parts[2])
                                if 0 <= index < GRID_SIZE:
                                    if self.select_prop(prop):
                                        if self.use_prop(prop, index):
                                            self.log_action("human", prop, None, index)
                                        self.selecting_prop = None
                                else:
                                    print("Invalid row/col index")
                            elif prop in ['bomb', 'hammer'] and len(parts) == 4:
                                i, j = int(parts[2]), int(parts[3])
                                if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
                                    if self.select_prop(prop):
                                        if self.use_prop(prop, [i, j]):
                                            self.log_action("human", prop, [i, j])
                                        self.selecting_prop = None
                                else:
                                    print("Invalid coordinates")
                            else:
                                print("Invalid prop command")
                        elif cmd == 'm' and len(parts) == 2:
                            mode = parts[1]
                            if mode == 'human':
                                self.set_human_mode()
                            elif mode == 'agent':
                                self.set_agent_mode()
                            else:
                                print("Invalid mode")
                        elif cmd == 'r':
                            self.reset_game()
                        elif cmd == 's':
                            self.show_level_select()
                        else:
                            print("Invalid command")
                    except (ValueError, IndexError):
                        print("Invalid input format")
                    self.display_board()
                else:
                    time.sleep(1)

# Example:
# OPENAI_API_KEY=... python src/match_game/general_game.py --model gpt-4 --auto-run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match Game with model-specific result storage")
    parser.add_argument('--model', type=str, default='gpt-4', help='Model name for result storage and API model (e.g., gpt-4, gpt-5)')
    parser.add_argument('--auto-run', action='store_true', help='Run all levels in agent mode automatically')
    parser.add_argument('--api_key', type=str, help='API key for the model')
    parser.add_argument('--base_url', type=str, help='Base URL for the API')
    args = parser.parse_args()
    
    game = MatchGame(
        model_name=args.model, 
        auto_run=args.auto_run,
        api_key=args.api_key,
        base_url=args.base_url
    )
    game.run()
