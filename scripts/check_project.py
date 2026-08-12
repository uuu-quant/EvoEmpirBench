#!/usr/bin/env python
"""Lightweight repository sanity checks for the cleaned release."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")

REQUIRED_DIRS = [
    "src/agent",
    "src/config",
    "src/game",
    "src/match_game",
    "scripts",
    "examples",
    "data/levels/maze_eval",
    "data/levels/maze_train",
    "data/levels/match_game",
]

FORBIDDEN_PATHS = [
    "results",
    "outputs",
    "data/agent_memory",
    "data/agent_sessions",
    "data/memory_validation",
    "data/truth_optimization",
    "data/memory_promotion",
    "data/collected_data",
    "data/processed_data",
    ".env",
    "venv",
    ".venv",
    ".idea",
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".sh",
    ".txt",
    ".toml",
    ".example",
}

EXPECTED_MAZE_SIZES = {
    "Level_1": 7,
    "Level_2": 9,
    "Level_3": 11,
}


def check_required_dirs() -> list[str]:
    errors = []
    for rel_path in REQUIRED_DIRS:
        if not (ROOT / rel_path).is_dir():
            errors.append(f"Missing required directory: {rel_path}")
    return errors


def check_forbidden_paths() -> list[str]:
    errors = []
    for rel_path in FORBIDDEN_PATHS:
        if (ROOT / rel_path).exists():
            errors.append(f"Generated/local output should be removed or ignored: {rel_path}")
    return errors


def check_json_assets() -> list[str]:
    errors = []
    json_roots = [
        ROOT / "data" / "levels" / "maze_eval",
        ROOT / "data" / "levels" / "maze_train",
        ROOT / "data" / "levels" / "match_game",
    ]
    for json_root in json_roots:
        if not json_root.exists():
            continue
        json_files = sorted(json_root.rglob("*.json"))
        if not json_files:
            errors.append(f"No JSON level files found under {json_root.relative_to(ROOT)}")
            continue
        for path in json_files[:10]:
            try:
                with path.open("r", encoding="utf-8") as file:
                    json.load(file)
            except Exception as exc:
                errors.append(f"Invalid JSON: {path.relative_to(ROOT)} ({exc})")
    return errors


def _maze_level_key(path: Path) -> str | None:
    match = re.search(r"Level_[123]", path.name)
    return match.group(0) if match else None


def _check_maze_map_shape(map_data: dict, expected_size: int, label: str) -> str | None:
    grid = map_data.get("grid")
    if not isinstance(grid, list) or len(grid) != expected_size:
        return f"{label}: expected {expected_size} grid rows"
    if any(not isinstance(row, list) or len(row) != expected_size for row in grid):
        return f"{label}: expected {expected_size} grid columns"
    if map_data.get("grid_size") != expected_size:
        return f"{label}: expected grid_size={expected_size}, got {map_data.get('grid_size')}"
    return None


def check_maze_dimensions() -> list[str]:
    errors = []
    for root_name in ["maze_eval", "maze_train"]:
        maze_root = ROOT / "data" / "levels" / root_name
        if not maze_root.exists():
            continue
        for path in sorted(maze_root.glob("Level_*.json")):
            level_key = _maze_level_key(path)
            if not level_key:
                continue
            expected_size = EXPECTED_MAZE_SIZES[level_key]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"Invalid JSON: {path.relative_to(ROOT)} ({exc})")
                continue

            if path.name.endswith("_collection.json"):
                if not isinstance(payload, list):
                    errors.append(f"{path.relative_to(ROOT)}: collection file should contain a list")
                    continue
                for index, map_data in enumerate(payload):
                    error = _check_maze_map_shape(
                        map_data,
                        expected_size,
                        f"{path.relative_to(ROOT)}[{index}]",
                    )
                    if error:
                        errors.append(error)
            else:
                error = _check_maze_map_shape(payload, expected_size, str(path.relative_to(ROOT)))
                if error:
                    errors.append(error)
    return errors


def iter_text_files():
    ignored_parts = {".git", "venv", ".venv", "__pycache__", ".idea", "results", "outputs"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".env.example"}:
            yield path


def check_secret_patterns() -> list[str]:
    errors = []
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_RE.search(text):
            errors.append(f"Possible API key pattern found in {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    checks = [
        check_required_dirs,
        check_forbidden_paths,
        check_json_assets,
        check_maze_dimensions,
        check_secret_patterns,
    ]

    errors = []
    for check in checks:
        errors.extend(check())

    if errors:
        print("Project check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Project check passed.")
    print(f"Root: {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
