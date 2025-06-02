"""
Memory manager module for Game 1.
Manages agent memories, including subjective memories and truth knowledge.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional

from ...common.utils import ensure_dir, to_serializable, save_json, load_json

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
        
        # Load subjective memories from map directories
        map_dirs = [d for d in os.listdir(self.memory_dir) 
                  if os.path.isdir(os.path.join(self.memory_dir, d)) and d != 'truth_optimizations']
        
        for level_id in map_dirs:
            map_path = os.path.join(self.memory_dir, level_id)
            # Check for subjective memory file
            memory_file = os.path.join(map_path, 'subjective_memory.json')
            if os.path.exists(memory_file):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                        level_id = memory_data.get('level_id', level_id)
                        self.subjective_memories[level_id] = memory_data
                        print(f"Loaded subjective memory for level {level_id}")
                except Exception as e:
                    print(f"Failed to load memory file {memory_file}: {str(e)}")
    
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
        
        # Save subjective memories to map directories
        for level_id, memory_data in self.subjective_memories.items():
            serializable_memory = to_serializable(memory_data)
            
            # Create map directory
            map_dir = os.path.join(self.memory_dir, level_id)
            ensure_dir(map_dir)
            
            memory_file = os.path.join(map_dir, 'subjective_memory.json')
            try:
                with open(memory_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_memory, f, indent=2, ensure_ascii=False)
                print(f"Saved subjective memory for level {level_id}")
            except Exception as e:
                print(f"Failed to save subjective memory for {level_id}: {str(e)}")
    
    def get_subjective_memory(self, level_id: str) -> Dict[str, Any]:
        """
        Get subjective memory for a level.
        
        Args:
            level_id: Level identifier
            
        Returns:
            Subjective memory dictionary or empty dict if not found
        """
        return self.subjective_memories.get(level_id, {})
    
    def get_truth_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get truth knowledge.
        
        Returns:
            List of truth knowledge entries
        """
        return self.truth_knowledge
    
    def add_subjective_memory(self, level_id: str, experience_summary: str, 
                              strengths: List[str], weaknesses: List[str],
                              game_metrics: Dict[str, Any]):
        """
        Add subjective memory for a level.
        
        Args:
            level_id: Level identifier
            experience_summary: Summary of agent experience
            strengths: List of identified strengths
            weaknesses: List of identified weaknesses
            game_metrics: Game performance metrics
        """
        memory = {
            'level_id': level_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'experience_summary': experience_summary,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'game_metrics': game_metrics
        }
        
        self.subjective_memories[level_id] = memory
        self.save_memory()
        
        print(f"Added subjective memory for level {level_id}")
    
    def promote_to_truth(self, memory_items: List[str], level_id: Optional[str] = None, 
                       sources: Optional[List[str]] = None) -> List[bool]:
        """
        Promote subjective memories to truth knowledge.
        
        Args:
            memory_items: List of memory items to promote
            level_id: Source level ID
            sources: List of source identifiers for each item
            
        Returns:
            List of booleans indicating promotion success for each item
        """
        promotion_results = []
        source_tags = sources if sources else [None] * len(memory_items)
        
        for i, (item, source_tag) in enumerate(zip(memory_items, source_tags)):
            # Create truth entry with source and timestamp
            source_prefix = f"{source_tag} from " if source_tag else ""
            source_detail = f"{source_prefix}Level {level_id}" if level_id else "Unknown"
            
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
    
    def update_truth_knowledge(self, new_truths: List[Dict[str, Any]]):
        """
        Update truth knowledge with new entries.
        
        Args:
            new_truths: New truth knowledge entries
        """
        # Update truth knowledge
        self.truth_knowledge = new_truths
        
        # Save to file
        self.save_memory()
        
        print(f"Updated truth knowledge, now {len(self.truth_knowledge)} entries")
    
    def get_memory_prompt(self, level_id: str, include_truth: bool = True, 
                          include_subjective: bool = True) -> str:
        """
        Generate memory prompt for agent.
        
        Args:
            level_id: Level identifier
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
        if include_subjective and level_id in self.subjective_memories:
            memory = self.subjective_memories[level_id]
            prompt_parts.append(f"# Subjective Memory for Level {level_id}")
            
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
    
    def clear_subjective_memory(self, level_id: str):
        """
        Clear subjective memory for a level.
        
        Args:
            level_id: Level identifier
        """
        if level_id in self.subjective_memories:
            del self.subjective_memories[level_id]
            print(f"Cleared subjective memory for level {level_id}")
            
            # Delete file
            map_dir = os.path.join(self.memory_dir, level_id)
            memory_file = os.path.join(map_dir, 'subjective_memory.json')
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