"""Centralized filesystem paths for the project.

Keep generated artifacts under ``outputs/`` and reproducible level assets under
``data/levels/`` so source code, inputs, and outputs stay visually separated.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
LEVELS_DIR = DATA_DIR / "levels"
MAZE_EVAL_MAPS_DIR = LEVELS_DIR / "maze_eval"
MAZE_TRAIN_MAPS_DIR = LEVELS_DIR / "maze_train"
MATCH_LEVELS_DIR = LEVELS_DIR / "match_game"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
MEMORY_DIR = OUTPUTS_DIR / "memory"
AGENT_MEMORY_DIR = MEMORY_DIR / "agent_memory"
MATCH_MEMORY_DIR = MEMORY_DIR / "match_game"
MEMORIES_DIR = MEMORY_DIR / "memories"
AGENT_SESSIONS_DIR = OUTPUTS_DIR / "agent_sessions"
MEMORY_VALIDATION_DIR = OUTPUTS_DIR / "memory_validation"
TRUTH_OPTIMIZATION_DIR = OUTPUTS_DIR / "truth_optimization"
MEMORY_PROMOTION_DIR = OUTPUTS_DIR / "memory_promotion"
COLLECTED_DATA_DIR = OUTPUTS_DIR / "collected_data"
PROCESSED_DATA_DIR = OUTPUTS_DIR / "processed_data"
