"""Memory management utilities for the Match-2 environment.

This module keeps Match-2 specific subjective memories and validated truth
knowledge separate from the maze-navigation agent memory while using the
repository-wide ``outputs/`` convention for generated runtime state.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.config.paths import MATCH_MEMORY_DIR


class MatchGameMemoryManager:
    """Manage subjective memories and truth knowledge for Match-2 agents."""

    def __init__(self, memory_dir: str | None = None):
        self.memory_dir = Path(memory_dir) if memory_dir else MATCH_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.subjective_memories: Dict[str, Dict[str, Any]] = {}
        self.truth_knowledge: List[Dict[str, Any]] = []

        self._load_memory()

    def _convert_to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, set):
            return [self._convert_to_serializable(item) for item in obj]
        if hasattr(obj, "__dict__"):
            return {
                key: self._convert_to_serializable(value)
                for key, value in obj.__dict__.items()
                if not key.startswith("_")
            }
        return obj

    def _load_memory(self) -> None:
        truth_file = self.memory_dir / "truth_knowledge.json"
        if truth_file.exists():
            try:
                self.truth_knowledge = json.loads(truth_file.read_text(encoding="utf-8"))
                print(f"Loaded {len(self.truth_knowledge)} truth knowledge entries")
            except Exception as exc:
                print(f"Failed to load truth knowledge: {exc}")
                self.truth_knowledge = []

        subjective_file = self.memory_dir / "subjective_memories.json"
        if subjective_file.exists():
            try:
                self.subjective_memories = json.loads(subjective_file.read_text(encoding="utf-8"))
                print(f"Loaded subjective memories for {len(self.subjective_memories)} levels")
            except Exception as exc:
                print(f"Failed to load subjective memories: {exc}")
                self.subjective_memories = {}

    def save_memory(self) -> None:
        serializable_truth = self._convert_to_serializable(self.truth_knowledge)
        serializable_subjective = self._convert_to_serializable(self.subjective_memories)

        truth_file = self.memory_dir / "truth_knowledge.json"
        try:
            truth_file.write_text(
                json.dumps(serializable_truth, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Saved {len(self.truth_knowledge)} truth knowledge entries")
        except Exception as exc:
            print(f"Failed to save truth knowledge: {exc}")

        subjective_file = self.memory_dir / "subjective_memories.json"
        try:
            subjective_file.write_text(
                json.dumps(serializable_subjective, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Saved subjective memories for {len(self.subjective_memories)} levels")
        except Exception as exc:
            print(f"Failed to save subjective memories: {exc}")

    def get_level_id(self, difficulty: str, level: int) -> str:
        return f"{difficulty}_level{level:02d}"

    def get_subjective_memory(self, difficulty: str, level: int) -> Dict[str, Any]:
        level_id = self.get_level_id(difficulty, level)
        return self.subjective_memories.get(level_id, {})

    def get_truth_knowledge(self) -> List[Dict[str, Any]]:
        return self.truth_knowledge

    def add_subjective_memory(
        self,
        difficulty: str,
        level: int,
        experience_summary: str,
        strengths: List[str],
        weaknesses: List[str],
        game_metrics: Dict[str, Any],
    ) -> None:
        level_id = self.get_level_id(difficulty, level)
        self.subjective_memories[level_id] = {
            "level_id": level_id,
            "difficulty": difficulty,
            "level": level,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experience_summary": experience_summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "game_metrics": game_metrics,
        }
        self.save_memory()
        print(f"Added subjective memory for level {level_id}")

    def promote_to_truth(
        self,
        memory_items: List[str],
        difficulty: str | None = None,
        level: int | None = None,
        sources: Optional[List[str]] = None,
    ) -> List[bool]:
        promotion_results = []
        source_tags = sources if sources else [None] * len(memory_items)
        level_id = self.get_level_id(difficulty, level) if difficulty and level else "Unknown"

        for item, source_tag in zip(memory_items, source_tags):
            source_prefix = f"{source_tag} from " if source_tag else ""
            source_detail = f"{source_prefix}Level {level_id}" if level_id != "Unknown" else "Unknown"
            truth_entry = {
                "knowledge": item,
                "source": source_detail,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            if not any(entry.get("knowledge") == item for entry in self.truth_knowledge):
                self.truth_knowledge.append(truth_entry)
                print(f"Promoted '{item}' to truth knowledge")
                promotion_results.append(True)
            else:
                print(f"Truth knowledge '{item}' already exists, skipping")
                promotion_results.append(False)

        self.save_memory()
        return promotion_results

    def update_truth_knowledge(self, new_truths: List[Dict[str, Any]]) -> None:
        self.truth_knowledge = new_truths
        self.save_memory()
        print(f"Truth knowledge base updated; current entries: {len(self.truth_knowledge)}")

    def get_memory_prompt(
        self,
        difficulty: str | None = None,
        level: int | None = None,
        include_truth: bool = True,
        include_subjective: bool = True,
    ) -> str:
        prompt_parts = []

        if include_truth and self.truth_knowledge:
            prompt_parts.append("# Truth Knowledge (Validated Learning)")
            for entry in self.truth_knowledge:
                prompt_parts.append(f"- {entry['knowledge']}")
            prompt_parts.append("")

        if include_subjective and difficulty and level:
            level_id = self.get_level_id(difficulty, level)
            memory = self.subjective_memories.get(level_id)
            if memory:
                prompt_parts.append(f"# Subjective Memory for Level {level_id}")

                if memory.get("experience_summary"):
                    prompt_parts.append("## Experience Summary")
                    prompt_parts.append(memory["experience_summary"])
                    prompt_parts.append("")

                if memory.get("strengths"):
                    prompt_parts.append("## Strengths")
                    for strength in memory["strengths"]:
                        prompt_parts.append(f"- {strength}")
                    prompt_parts.append("")

                if memory.get("weaknesses"):
                    prompt_parts.append("## Weaknesses")
                    for weakness in memory["weaknesses"]:
                        prompt_parts.append(f"- {weakness}")
                    prompt_parts.append("")

        return "\n".join(prompt_parts) if prompt_parts else ""

    def clear_subjective_memory(self, difficulty: str, level: int) -> None:
        level_id = self.get_level_id(difficulty, level)
        if level_id in self.subjective_memories:
            del self.subjective_memories[level_id]
            print(f"Cleared subjective memory for level {level_id}")
            self.save_memory()

    def clear_all_memories(self) -> None:
        self.subjective_memories = {}
        self.truth_knowledge = []
        print("Cleared all memories and truth knowledge")
        self.save_memory()

    def validate_subjective_memory(
        self,
        prev_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any],
    ) -> bool:
        score_improved = new_metrics["total_score"] > prev_metrics["total_score"]
        success_improved = new_metrics["cleared"] and not prev_metrics["cleared"]
        efficiency_improved = new_metrics["avg_score_per_step"] > prev_metrics["avg_score_per_step"]
        clear_improved = new_metrics["avg_clear_per_step"] > prev_metrics["avg_clear_per_step"]

        improvement_score = (
            (1 if score_improved else 0) * 2
            + (1 if success_improved else 0) * 3
            + (1 if efficiency_improved else 0)
            + (1 if clear_improved else 0)
        )

        is_valid = score_improved and (
            new_metrics["cleared"] or (efficiency_improved and improvement_score >= 3)
        )

        print("Memory validation result:")
        print(f"- Score improved: {score_improved} ({prev_metrics['total_score']} -> {new_metrics['total_score']})")
        print(f"- Success improved: {success_improved} ({prev_metrics['cleared']} -> {new_metrics['cleared']})")
        print(
            "- Efficiency improved: "
            f"{efficiency_improved} ({prev_metrics['avg_score_per_step']:.2f} -> "
            f"{new_metrics['avg_score_per_step']:.2f})"
        )
        print(
            "- Clearance efficiency improved: "
            f"{clear_improved} ({prev_metrics['avg_clear_per_step']:.2f} -> "
            f"{new_metrics['avg_clear_per_step']:.2f})"
        )
        print(f"- Overall improvement score: {improvement_score}/7")
        print(f"- Memory valid: {is_valid}")

        return is_valid

    def optimize_truth_knowledge(self, reflection_func: Callable[[str], str]) -> None:
        all_truths = self.get_truth_knowledge()

        if not all_truths or len(all_truths) <= 1:
            print("Not enough truth entries for filtering and merging")
            return

        print(f"Starting truth knowledge filtering and merging; current entries: {len(all_truths)}")
        optimization_prompt = self._create_truth_optimization_prompt(all_truths)
        optimized_response = reflection_func(optimization_prompt)
        optimized_truths = self._parse_optimized_truth(optimized_response)

        if not optimized_truths:
            print("Could not parse optimized truth knowledge; keeping original knowledge base unchanged")
            return

        original_count = len(all_truths)
        optimized_count = len(optimized_truths)
        self.update_truth_knowledge(optimized_truths)
        self._log_truth_optimization(all_truths, optimized_truths)
        print(f"Truth knowledge optimization: {original_count} -> {optimized_count} entries")

    def _create_truth_optimization_prompt(self, truths: List[Dict[str, Any]]) -> str:
        prompt_parts = [
            "# Match-2 Game Truth Knowledge Organization Task",
            "",
            "Please review and organize the following Match-2 game truth knowledge entries. "
            "Identify duplicates and merge highly similar entries into clearer, reusable rules.",
            "",
            "## Current Knowledge Entries",
        ]

        for index, truth in enumerate(truths, 1):
            content = truth.get("knowledge", truth.get("content", truth.get("text", str(truth))))
            source = f"(Source: {truth.get('source', 'unknown')})" if "source" in truth else ""
            prompt_parts.append(f"{index}. {content} {source}")

        prompt_parts.extend(
            [
                "",
                "## Organization Requirements",
                "1. Remove exact duplicates.",
                "2. Merge entries only when their meanings are highly similar.",
                "3. Preserve concrete strategic details when merging.",
                "4. Keep independent knowledge points separate.",
                "5. Return only a numbered list inside a code block.",
                "",
                "```",
                "1. [Organized knowledge entry 1]",
                "2. [Organized knowledge entry 2]",
                "```",
            ]
        )
        return "\n".join(prompt_parts)

    def _parse_optimized_truth(self, optimization_response: str) -> List[Dict[str, Any]]:
        if not optimization_response:
            return []

        block_match = re.search(r"```(.*?)```", optimization_response, re.DOTALL)
        if block_match:
            candidate_text = block_match.group(1).strip()
        else:
            candidate_text = optimization_response.strip()

        content_lines = []
        for line in candidate_text.splitlines():
            if re.match(r"^\d+\.\s", line.strip()):
                content = re.sub(r"^\d+\.\s", "", line.strip()).strip()
                if content:
                    content_lines.append(content)

        return [
            {
                "knowledge": content,
                "source": "Multi-level Fusion",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for content in content_lines
        ]

    def _log_truth_optimization(
        self,
        original_truths: List[Dict[str, Any]],
        optimized_truths: List[Dict[str, Any]],
    ) -> None:
        optimization_dir = self.memory_dir / "truth_optimizations"
        optimization_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_data = {
            "timestamp": timestamp,
            "original_count": len(original_truths),
            "optimized_count": len(optimized_truths),
            "original_truths": original_truths,
            "optimized_truths": optimized_truths,
        }
        log_file = optimization_dir / f"optimization_{timestamp}.json"
        try:
            log_file.write_text(
                json.dumps(self._convert_to_serializable(log_data), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Truth knowledge optimization result saved to {log_file}")
        except Exception as exc:
            print(f"Failed to save truth knowledge optimization result: {exc}")


MemoryManager = MatchGameMemoryManager
