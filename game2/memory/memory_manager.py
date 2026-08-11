"""
Memory manager module for Game 2.
Manages agent memories, including subjective memories and truth knowledge.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable

from nips2025.common.utils import ensure_dir, to_serializable, save_json, load_json

class MemoryManager:
    """
    Memory manager for handling subjective memories and truth knowledge.
    
    This class manages two types of memories:
    - Subjective memories: Level-specific memories that are candidate for promotion
    - Truth knowledge: Validated memories that have been proven effective
    """
    
    def __init__(self, memory_dir: str = None):
        """
        Initialize memory manager.
        
        Args:
            memory_dir: Directory to store memory files
        """
        # Set memory directory
        if memory_dir is None:
            self.memory_dir = os.path.join(os.path.dirname(os.path.dirname(
                             os.path.dirname(os.path.abspath(__file__)))), "data", "memory")
        else:
            self.memory_dir = memory_dir
            
        # Store root directory for reference
        self.memory_root_dir = self.memory_dir
            
        # Create memory directory
        ensure_dir(self.memory_dir)
        
        # Initialize memory stores
        self.subjective_memories = {}  # Level-specific memories
        self.truth_knowledge = []      # Global validated memories
        
        # Load existing memories
        self._load_memory()
    
    def _load_memory(self):
        """Load existing memories from files."""
        # Load truth knowledge
        truth_file = os.path.join(self.memory_dir, 'truth_knowledge.json')
        if os.path.exists(truth_file):
            try:
                with open(truth_file, 'r', encoding='utf-8') as f:
                    self.truth_knowledge = json.load(f)
                print(f"Loaded {len(self.truth_knowledge)} truth knowledge entries")
            except Exception as e:
                print(f"Failed to load truth knowledge: {str(e)}")
                self.truth_knowledge = []
        
        # Load subjective memories from game directories
        game_dir = os.path.join(self.memory_dir, 'game2')
        if os.path.exists(game_dir):
            for difficulty in os.listdir(game_dir):
                difficulty_dir = os.path.join(game_dir, difficulty)
                if not os.path.isdir(difficulty_dir):
                    continue
                    
                for level_dir in os.listdir(difficulty_dir):
                    if not level_dir.startswith('level'):
                        continue
                        
                    try:
                        level_num = int(level_dir[5:])
                        memory_file = os.path.join(difficulty_dir, level_dir, 'subjective_memory.json')
                        
                        if os.path.exists(memory_file):
                            with open(memory_file, 'r', encoding='utf-8') as f:
                                memory_data = json.load(f)
                                level_id = self.get_level_id(difficulty, level_num)
                                self.subjective_memories[level_id] = memory_data
                                print(f"Loaded subjective memory for {difficulty} level {level_num}")
                    except Exception as e:
                        print(f"Failed to load memory: {str(e)}")
    
    def save_memory(self):
        """Save memories to files."""
        # Convert to serializable format
        serializable_truth = to_serializable(self.truth_knowledge)
        
        # Save truth knowledge
        truth_file = os.path.join(self.memory_dir, 'truth_knowledge.json')
        try:
            with open(truth_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_truth, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.truth_knowledge)} truth knowledge entries")
        except Exception as e:
            print(f"Failed to save truth knowledge: {str(e)}")
        
        # Save subjective memories to level directories
        for level_id, memory_data in self.subjective_memories.items():
            serializable_memory = to_serializable(memory_data)
            
            # Parse level ID to get difficulty and level
            parts = level_id.split('_')
            if len(parts) == 2:
                difficulty, level_str = parts
                if level_str.startswith('level'):
                    try:
                        level_num = int(level_str[5:])
                        
                        # Create directory path
                        memory_dir = os.path.join(
                            self.memory_dir, 
                            'game2',
                            difficulty, 
                            f"level{level_num:02d}"
                        )
                        ensure_dir(memory_dir)
                        
                        # Save memory file
                        memory_file = os.path.join(memory_dir, 'subjective_memory.json')
                        with open(memory_file, 'w', encoding='utf-8') as f:
                            json.dump(serializable_memory, f, indent=2, ensure_ascii=False)
                        print(f"Saved subjective memory for {difficulty} level {level_num}")
                    except Exception as e:
                        print(f"Failed to save memory: {str(e)}")
    
    def get_level_id(self, difficulty: str, level: int) -> str:
        """
        Generate level ID from difficulty and level number.
        
        Args:
            difficulty: Game difficulty
            level: Level number
            
        Returns:
            Level ID string
        """
        return f"{difficulty}_level{level:02d}"
    
    def get_subjective_memory(self, difficulty: str, level: int) -> Dict[str, Any]:
        """
        Get subjective memory for a level.
        
        Args:
            difficulty: Game difficulty
            level: Level number
            
        Returns:
            Subjective memory dictionary or empty dict if not found
        """
        level_id = self.get_level_id(difficulty, level)
        return self.subjective_memories.get(level_id, {})
    
    def get_truth_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get truth knowledge.
        
        Returns:
            List of truth knowledge entries
        """
        return self.truth_knowledge
    
    def add_subjective_memory(self, difficulty: str, level: int, experience_summary: str, 
                              strengths: List[str], weaknesses: List[str],
                              game_metrics: Dict[str, Any]):
        """
        Add subjective memory for a level.
        
        Args:
            difficulty: Game difficulty
            level: Level number
            experience_summary: Summary of agent experience
            strengths: List of identified strengths
            weaknesses: List of identified weaknesses
            game_metrics: Game performance metrics
        """
        level_id = self.get_level_id(difficulty, level)
        
        memory = {
            'level_id': level_id,
            'difficulty': difficulty,
            'level': level,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'experience_summary': experience_summary,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'game_metrics': game_metrics
        }
        
        self.subjective_memories[level_id] = memory
        self.save_memory()
        
        print(f"Added subjective memory for {difficulty} level {level}")
    
    def validate_subjective_memory(self, difficulty: str, level: int, 
                                  prev_metrics: Dict[str, Any], 
                                  new_metrics: Dict[str, Any]) -> bool:
        """
        Validate subjective memory by comparing performance metrics.
        
        Args:
            difficulty: Game difficulty
            level: Level number
            prev_metrics: Previous performance metrics
            new_metrics: New performance metrics
            
        Returns:
            Whether memory is valid
        """
        # Check key metrics
        score_improved = new_metrics["score"] > prev_metrics["score"]
        success_improved = new_metrics["success"] and not prev_metrics["success"]
        steps_improved = (new_metrics["steps"] < prev_metrics["steps"] and 
                         new_metrics["success"])
        
        # Match-3 specific metrics
        avg_clear_improved = (new_metrics.get("avg_clear_per_step", 0) > 
                            prev_metrics.get("avg_clear_per_step", 0))
        
        # Calculate improvement score
        improvement_score = (
            (1 if score_improved else 0) * 2 +        # Score weight: 2
            (1 if success_improved else 0) * 3 +       # Success weight: 3
            (1 if steps_improved else 0) * 1 +         # Steps weight: 1
            (1 if avg_clear_improved else 0) * 1       # Avg clear weight: 1
        )
        
        # Validation criteria:
        # 1. Score must improve
        # 2. Must successfully complete level
        is_valid = score_improved and new_metrics["success"]
        
        # Print validation results
        print(f"Memory Validation Result:")
        print(f"- Difficulty: {difficulty}, Level: {level}")
        print(f"- Success: {new_metrics['success']}")
        print(f"- Score improvement: {score_improved} ({prev_metrics['score']} -> {new_metrics['score']})")
        print(f"- Success improvement: {success_improved} ({prev_metrics['success']} -> {new_metrics['success']})")
        print(f"- Steps improvement: {steps_improved} ({prev_metrics['steps']} -> {new_metrics['steps']})")
        print(f"- Avg clear improvement: {avg_clear_improved} ({prev_metrics.get('avg_clear_per_step', 0):.2f} -> {new_metrics.get('avg_clear_per_step', 0):.2f})")
        print(f"- Overall improvement score: {improvement_score}/7 (for reference only)")
        print(f"- Memory is valid: {is_valid} (requires both score improvement AND successful completion)")
        
        # Log validation result
        self._log_validation_result(
            difficulty,
            level,
            prev_metrics,
            new_metrics,
            {
                "score_improved": score_improved,
                "success_improved": success_improved,
                "steps_improved": steps_improved,
                "avg_clear_improved": avg_clear_improved,
                "improvement_score": improvement_score,
                "is_valid": is_valid
            }
        )
        
        return is_valid
    
    def _log_validation_result(self, difficulty: str, level: int,
                              prev_metrics: Dict[str, Any], 
                              new_metrics: Dict[str, Any],
                              validation_result: Dict[str, Any]):
        """
        Log validation result.
        
        Args:
            difficulty: Game difficulty
            level: Level number
            prev_metrics: Previous metrics
            new_metrics: New metrics
            validation_result: Validation result
        """
        # Create level-specific directory
        validation_dir = os.path.join(
            self.memory_dir, 
            'game2',
            difficulty, 
            f"level{level:02d}",
            'validations'
        )
        ensure_dir(validation_dir)
        
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create log data
        log_data = {
            'timestamp': timestamp,
            'difficulty': difficulty,
            'level': level,
            'previous_metrics': prev_metrics,
            'new_metrics': new_metrics,
            'validation_result': validation_result,
            'subjective_memory': self.get_subjective_memory(difficulty, level)
        }
        
        # Save validation log
        log_file = os.path.join(validation_dir, f"validation_{timestamp}.json")
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(to_serializable(log_data), f, indent=2, ensure_ascii=False)
            print(f"Validation results saved to {log_file}")
        except Exception as e:
            print(f"Failed to save validation results: {str(e)}")
    
    def promote_to_truth(self, memory_items: List[str], difficulty: str = None, 
                       level: int = None, sources: Optional[List[str]] = None) -> List[bool]:
        """
        Promote memory items to truth knowledge.
        
        Args:
            memory_items: List of memory items to promote
            difficulty: Source difficulty
            level: Source level
            sources: List of source identifiers for each item
            
        Returns:
            List of booleans indicating promotion success for each item
        """
        promotion_results = []
        source_tags = sources if sources else [None] * len(memory_items)
        
        for i, (item, source_tag) in enumerate(zip(memory_items, source_tags)):
            # Create source description
            source_prefix = f"{source_tag} from " if source_tag else ""
            source_detail = f"{source_prefix}{difficulty} Level {level}" if difficulty and level else "Unknown"
            
            # Create truth entry
            truth_entry = {
                'knowledge': item,
                'source': source_detail,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Check if already exists
            if not any(entry['knowledge'] == item for entry in self.truth_knowledge):
                self.truth_knowledge.append(truth_entry)
                print(f"Promoted '{item}' to truth knowledge")
                promotion_results.append(True)
            else:
                print(f"Truth knowledge '{item}' already exists, skipping")
                promotion_results.append(False)
        
        self.save_memory()
        return promotion_results
    
    def optimize_truth_knowledge(self, reflect_func: Callable[[str], str]):
        """
        Filter and merge truth knowledge entries.
        
        Args:
            reflect_func: Function to use for reflection
        """
        # Get all truth knowledge
        all_truths = self.get_truth_knowledge()
        
        if not all_truths or len(all_truths) <= 1:
            print("Not enough entries in truth knowledge base for filtering and merging")
            return
        
        print(f"Starting filtering and merging of truth knowledge base, current entries: {len(all_truths)}")
        
        # Create optimization prompt
        optimization_prompt = self._create_truth_optimization_prompt(all_truths)
        
        # Get optimization response
        optimization_response = reflect_func(optimization_prompt)
        
        # Parse optimized truth knowledge
        optimized_truths = self._parse_optimized_truth(optimization_response)
        
        if not optimized_truths:
            print("Could not parse optimized truth knowledge, keeping original knowledge base unchanged")
            return
        
        # Record optimization results
        original_count = len(all_truths)
        optimized_count = len(optimized_truths)
        
        # Update truth knowledge if optimization was successful
        if optimized_count < original_count:
            print(f"Truth knowledge optimization successful: {original_count} -> {optimized_count} entries")
            self.truth_knowledge = optimized_truths
        else:
            print(f"Truth knowledge remains unchanged: {original_count} entries (no similar entries found to merge)")
            self.truth_knowledge = optimized_truths
        
        # Save updated knowledge
        self.save_memory()
        
        # Log optimization result
        self._log_truth_optimization(all_truths, optimized_truths)
    
    def _create_truth_optimization_prompt(self, truths: List[Dict[str, Any]]) -> str:
        """
        Create truth optimization prompt.
        
        Args:
            truths: Truth knowledge entries
            
        Returns:
            Optimization prompt
        """
        prompt_parts = [
            "# Match-3 Game Truth Knowledge Organization Task",
            "",
            "Please review and organize the following match-3 game truth knowledge entries. This is a progressive knowledge organization process to identify and remove duplicates while considering merging highly similar entries.",
            "",
            "## Current Knowledge Entries:"
        ]
        
        # Add current truth knowledge entries
        for i, truth in enumerate(truths, 1):
            # Get content from different possible keys
            content = truth.get('knowledge', truth.get('content', truth.get('text', str(truth))))
            source = f"(Source: {truth.get('source', 'unknown')})" if 'source' in truth else ""
            prompt_parts.append(f"{i}. {content} {source}")
        
        # Add organization instructions
        prompt_parts.append("")
        prompt_parts.append("## Organization Requirements:")
        prompt_parts.append("1. Identify and remove completely duplicate knowledge entries")
        prompt_parts.append("2. For knowledge entries that are highly similar in meaning but different in expression, merge them into a more comprehensive entry")
        prompt_parts.append("3. When merging, preserve the specificity and core meaning of the original knowledge, don't lose key details")
        prompt_parts.append("4. Merged entries should be concise but not at the expense of important information")
        prompt_parts.append("5. If two knowledge points only have minimal similarities, keep them as separate entries")
        prompt_parts.append("6. Knowledge entries without clear similarities should remain unchanged")
        prompt_parts.append("")
        prompt_parts.append("## Please return the organized knowledge base in the following format:")
        prompt_parts.append("```")
        prompt_parts.append("1. [Organized knowledge entry 1]")
        prompt_parts.append("2. [Organized knowledge entry 2]")
        prompt_parts.append("...")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("Note: This is a progressive knowledge organization process, you do not need to force a reduction in the number of entries. Only merge or remove entries when there is genuine high similarity or duplication.")
        
        return "\n".join(prompt_parts)
    
    def _parse_optimized_truth(self, optimization_response: str) -> List[Dict[str, Any]]:
        """
        Parse optimized truth knowledge.
        
        Args:
            optimization_response: Optimization response from agent
            
        Returns:
            Optimized truth knowledge entries
        """
        # Try to extract content from code block
        import re
        pattern = r"```(.*?)```"
        match = re.search(pattern, optimization_response, re.DOTALL)
        
        if not match:
            # Try to extract numbered list directly
            lines = optimization_response.strip().split("\n")
            content_lines = []
            for line in lines:
                # Check if line starts with number and dot
                if re.match(r"^\d+\.\s", line):
                    content = re.sub(r"^\d+\.\s", "", line).strip()
                    if content:
                        content_lines.append(content)
            
            if not content_lines:
                print("Could not parse optimized truth knowledge")
                return []
        else:
            # Extract content from code block
            content = match.group(1).strip()
            content_lines = [
                re.sub(r"^\d+\.\s", "", line).strip()
                for line in content.split("\n")
                if line.strip() and re.match(r"^\d+\.\s", line)
            ]
        
        # Create optimized truth entries
        optimized_truths = []
        for content in content_lines:
            if content:
                optimized_truths.append({
                    'knowledge': content,
                    'source': "Multi-level Fusion",
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        return optimized_truths
    
    def _log_truth_optimization(self, original_truths: List[Dict[str, Any]], 
                               optimized_truths: List[Dict[str, Any]]):
        """
        Log truth optimization result.
        
        Args:
            original_truths: Original truth knowledge entries
            optimized_truths: Optimized truth knowledge entries
        """
        # Create optimization directory
        optimization_dir = os.path.join(self.memory_dir, 'truth_optimizations')
        ensure_dir(optimization_dir)
        
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create log data
        log_data = {
            "timestamp": timestamp,
            "original_count": len(original_truths),
            "optimized_count": len(optimized_truths),
            "original_truths": original_truths,
            "optimized_truths": optimized_truths
        }
        
        # Save optimization log
        log_file = os.path.join(optimization_dir, f"optimization_{timestamp}.json")
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(to_serializable(log_data), f, indent=2, ensure_ascii=False)
            print(f"Truth knowledge optimization results saved to {log_file}")
        except Exception as e:
            print(f"Failed to save truth knowledge optimization results: {str(e)}")
    
    def get_memory_prompt(self, difficulty: str, level: int, 
                         include_truth: bool = True, 
                         include_subjective: bool = True) -> str:
        """
        Generate memory prompt for agent.
        
        Args:
            difficulty: Game difficulty
            level: Level number
            include_truth: Whether to include truth knowledge
            include_subjective: Whether to include subjective memory
            
        Returns:
            Formatted memory prompt string
        """
        prompt_parts = []
        
        # Add truth knowledge
        if include_truth and self.truth_knowledge:
            prompt_parts.append("# Truth Knowledge (Validated Learning)")
            for entry in self.truth_knowledge:
                prompt_parts.append(f"- {entry['knowledge']}")
            prompt_parts.append("")
        
        # Add subjective memory
        if include_subjective:
            level_id = self.get_level_id(difficulty, level)
            memory = self.subjective_memories.get(level_id, {})
            
            if memory:
                prompt_parts.append(f"# Subjective Memory for {difficulty} Level {level}")
                
                if memory.get('strengths'):
                    prompt_parts.append("## Strengths:")
                    for strength in memory['strengths']:
                        prompt_parts.append(f"- {strength}")
                    prompt_parts.append("")
                
                if memory.get('weaknesses'):
                    prompt_parts.append("## Weaknesses:")
                    for weakness in memory['weaknesses']:
                        prompt_parts.append(f"- {weakness}")
                    prompt_parts.append("")
        
        return "\n".join(prompt_parts) if prompt_parts else ""
    
    def clear_subjective_memory(self, difficulty: str, level: int):
        """
        Clear subjective memory for a level.
        
        Args:
            difficulty: Game difficulty
            level: Level number
        """
        level_id = self.get_level_id(difficulty, level)
        
        if level_id in self.subjective_memories:
            del self.subjective_memories[level_id]
            print(f"Cleared subjective memory for {difficulty} level {level}")
            
            # Delete file
            memory_dir = os.path.join(
                self.memory_dir, 
                'game2',
                difficulty, 
                f"level{level:02d}"
            )
            memory_file = os.path.join(memory_dir, 'subjective_memory.json')
            
            if os.path.exists(memory_file):
                try:
                    os.remove(memory_file)
                except Exception as e:
                    print(f"Failed to delete memory file: {str(e)}")
        
        self.save_memory()
    
    def clear_all_memories(self):
        """Clear all memories."""
        self.subjective_memories = {}
        self.truth_knowledge = []
        
        print("Cleared all memories and truth knowledge")
        self.save_memory() 