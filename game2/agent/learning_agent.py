"""
Learning agent module for Game 2.
Implements an agent that can learn from match-3 puzzle game experiences.
"""

import os
import re
import json
import time
from typing import Dict, List, Any, Tuple, Optional

from nips2025.common.gpt_client import GPTClient
from nips2025.game2.memory.memory_manager import MemoryManager
from nips2025.common.utils import ensure_dir, to_serializable, save_json, get_timestamp
from nips2025.game2.game.config import *

class MatchLearningAgent:
    """
    Learning agent for the match-3 puzzle game.
    
    This agent:
    1. Plays games and records experiences
    2. Reflects on experiences to extract insights
    3. Validates insights through additional gameplay
    4. Promotes validated insights to a truth knowledge base
    """
    
    def __init__(self, api_client, memory_dir: str = None, results_dir: str = None):
        """
        Initialize match learning agent.
        
        Args:
            api_client: GPT client for API interactions
            memory_dir: Directory for storing memory
            results_dir: Directory for storing results
        """
        # Store API client
        self.api_client = api_client
        
        # Create memory manager
        self.memory_manager = MemoryManager(memory_dir)
        
        # Set results directory
        if results_dir is None:
            self.results_dir = self.memory_manager.memory_dir
        else:
            self.results_dir = results_dir
            ensure_dir(self.results_dir)
        
        # Current session information
        self.difficulty = None
        self.level = None
        self.session_logs = []
        self.session_metrics = {}
        self.session_timestamp = get_timestamp()
        self.session_dir = ""
    
    def start_session(self, difficulty: str, level: int):
        """
        Start new game session.
        
        Args:
            difficulty: Game difficulty
            level: Level number
        """
        # Set current session info
        self.difficulty = difficulty
        self.level = level
        
        # Get level ID
        level_id = self.memory_manager.get_level_id(difficulty, level)
        
        # Reset session logs and metrics
        self.session_logs = []
        self.session_metrics = {
            "level_id": level_id,
            "difficulty": difficulty,
            "level": level,
            "start_time": time.time(),
            "steps": 0,
            "success": False,
            "score": 0,
            "total_cleared": 0,
            "color_counts": {color: 0 for color in COLOR_KEYS},
            "api_calls": 0,
            "valid_api_calls": 0
        }
        
        # Generate new timestamp
        self.session_timestamp = get_timestamp()
        
        # Create session directory
        self.session_dir = os.path.join(
            self.results_dir,
            "game2",
            difficulty,
            f"level{level:02d}",
            "agent"
        )
        ensure_dir(self.session_dir)
        
        print(f"Starting game session for {difficulty} level {level}")
    
    def log_interaction(self, state: Dict[str, Any], action: Dict[str, Any], 
                       response: str, reward: int, cleared_count: int):
        """
        Log agent-environment interaction.
        
        Args:
            state: Game state
            action: Action taken
            response: API response text
            reward: Reward received
            cleared_count: Number of tiles cleared
        """
        # Record interaction
        interaction = {
            "step": len(self.session_logs) + 1,
            "state": {
                "board": state.get("board"),
                "score": state.get("score"),
                "steps_remaining": state.get("steps_remaining"),
                "inventory": state.get("inventory"),
                "color_targets": state.get("color_targets"),
                "color_counts": state.get("color_counts")
            },
            "action": action,
            "reward": reward,
            "cleared_count": cleared_count,
            "api_response": response,
            "timestamp": time.time()
        }
        
        self.session_logs.append(interaction)
        
        # Update session metrics
        self.session_metrics["steps"] += 1
        self.session_metrics["api_calls"] += 1
        if action is not None:
            self.session_metrics["valid_api_calls"] += 1
        
        # Update color counts if provided
        if state.get("color_counts"):
            self.session_metrics["color_counts"] = state["color_counts"]
        
        # Update total cleared
        self.session_metrics["total_cleared"] += cleared_count 
    
    def end_session(self, score: int, color_counts: Dict[str, int], 
                  cleared: bool, steps_remaining: int) -> Dict[str, Any]:
        """
        End game session and update metrics.
        
        Args:
            score: Final score
            color_counts: Final color counts
            cleared: Whether level was cleared
            steps_remaining: Remaining steps
            
        Returns:
            Session metrics
        """
        # Update final metrics
        self.session_metrics.update({
            "end_time": time.time(),
            "duration": time.time() - self.session_metrics["start_time"],
            "success": cleared,
            "score": score,
            "color_counts": color_counts,
            "steps_remaining": steps_remaining,
            "avg_clear_per_step": (
                self.session_metrics["total_cleared"] / self.session_metrics["steps"]
                if self.session_metrics["steps"] > 0 else 0
            ),
            "avg_score_per_step": (
                score / self.session_metrics["steps"]
                if self.session_metrics["steps"] > 0 else 0
            ),
            "valid_api_ratio": (
                self.session_metrics["valid_api_calls"] / self.session_metrics["api_calls"]
                if self.session_metrics["api_calls"] > 0 else 0
            )
        })
        
        # Log results
        print(f"Ending game session for {self.difficulty} level {self.level}")
        print(f"Result: {'Success' if cleared else 'Failure'}")
        print(f"Score: {score}, Steps remaining: {steps_remaining}")
        print(f"Color counts: {color_counts}")
        print(f"Color targets: {self.session_logs[0]['state']['color_targets'] if self.session_logs else 'Unknown'}")
        
        # Save session data
        self._save_session_data()
        
        return self.session_metrics
    
    def _save_session_data(self):
        """Save session data to files."""
        # Convert to serializable format
        serializable_logs = to_serializable(self.session_logs)
        serializable_metrics = to_serializable(self.session_metrics)
        
        try:
            # Save session logs
            log_file = os.path.join(self.session_dir, f"session_log_{self.session_timestamp}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_logs, f, indent=2, ensure_ascii=False)
            
            # Save session metrics
            metrics_file = os.path.join(self.session_dir, f"session_metrics_{self.session_timestamp}.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            
            # Save action log
            action_log_file = os.path.join(self.session_dir, ACTION_LOG_FILE.format(self.session_timestamp))
            with open(action_log_file, 'w', encoding='utf-8') as f:
                for log in self.session_logs:
                    f.write(json.dumps({
                        "step": log["step"],
                        "action": log["action"],
                        "cleared_count": log["cleared_count"],
                        "score": log["state"]["score"],
                        "steps_remaining": log["state"]["steps_remaining"]
                    }, ensure_ascii=False) + '\n')
            
            print(f"Session data saved to {self.session_dir}")
        except Exception as e:
            print(f"Error saving session data: {str(e)}")
    
    def get_action(self, state, with_memory=True):
        """
        Get agent's next action.
        
        Args:
            state: Game state
            with_memory: Whether to use memory enhancement
            
        Returns:
            Action to take
        """
        # Check if memory enhancement should be used
        if with_memory:
            # Build memory-enhanced system prompt
            memory_prompt = self.memory_manager.get_memory_prompt(
                self.difficulty, self.level,
                include_truth=True,
                include_subjective=True
            )
            
            if memory_prompt:
                # Create system prompt with memory
                system_prompt = f"""
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

{memory_prompt}
"""
                # Create user prompt with game state
                user_prompt = self._create_game_state_prompt(state)
                
                # Get response with memory-enhanced prompt
                response = self.api_client.get_response([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                
                # Parse action from response
                action = self._parse_action_from_response(response)
                
                return action, response
        
        # Default case - use standard prompt
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
        user_prompt = self._create_game_state_prompt(state)
        
        # Get response with standard prompt
        response = self.api_client.get_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Parse action from response
        action = self._parse_action_from_response(response)
        
        return action, response
    
    def _parse_action_from_response(self, response):
        """
        Parse action from API response.
        
        Args:
            response: API response text
            
        Returns:
            Parsed action or None if invalid
        """
        if response == 'Failed':
            return None
        
        # Try to extract JSON from response
        try:
            # Look for JSON code block
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                action = json.loads(json_str)
                return action
            
            # Try to extract standalone JSON object
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                action = json.loads(json_str)
                return action
            
            # If no JSON found, return None
            return None
        except Exception as e:
            print(f"Error parsing action from response: {str(e)}")
            return None 
    
    def reflect_on_session(self) -> Tuple[str, List[str], List[str]]:
        """
        Reflect on game session and extract insights.
        
        Returns:
            (experience_summary, strengths, weaknesses): Extracted insights
        """
        # Create reflection prompt
        reflection_prompt = self._create_reflection_prompt()
        
        # Get reflection response
        reflection_response = self.api_client.get_response([
            {"role": "system", "content": "You are an AI assistant that helps analyze game playing sessions and extract meaningful insights."},
            {"role": "user", "content": reflection_prompt}
        ])
        
        # Parse reflection
        experience_summary, strengths, weaknesses = self._parse_reflection(reflection_response)
        
        print("\n===== Game Reflection =====")
        print("Strengths:")
        for s in strengths:
            print(f"- {s}")
        print("Weaknesses:")
        for w in weaknesses:
            print(f"- {w}")
        print("====================\n")
        
        return experience_summary, strengths, weaknesses
    
    def _create_reflection_prompt(self) -> str:
        """
        Create reflection prompt.
        
        Returns:
            Reflection prompt
        """
        prompt_parts = [
            "# Match Game Session Analysis",
            "",
            "Please analyze the following match game session and provide insights about the agent's performance.",
            "",
            "## Game Metrics:"
        ]
        
        # Add session metrics
        for key, value in self.session_metrics.items():
            if key not in ["start_time", "end_time"] and not isinstance(value, dict):
                prompt_parts.append(f"- {key}: {value}")
        
        # Add color targets and counts
        if self.session_logs:
            first_log = self.session_logs[0]['state']
            if 'color_targets' in first_log:
                prompt_parts.append("- color_targets: " + str(first_log['color_targets']))
            
            last_log = self.session_logs[-1]['state']
            if 'color_counts' in last_log:
                prompt_parts.append("- final_color_counts: " + str(last_log['color_counts']))
        
        # Add session highlights
        prompt_parts.append("\n## Session Highlights:")
        
        # Add a sample of actions for context
        sample_indices = []
        if len(self.session_logs) <= 5:
            # If few actions, include all
            sample_indices = list(range(len(self.session_logs)))
        else:
            # Otherwise, include first, last, and some in between
            sample_indices = [0, len(self.session_logs) // 4, len(self.session_logs) // 2, 
                            3 * len(self.session_logs) // 4, len(self.session_logs) - 1]
        
        for i in sample_indices:
            log = self.session_logs[i]
            prompt_parts.append(f"\nStep {log['step']}:")
            prompt_parts.append(f"- Score: {log['state']['score']}")
            prompt_parts.append(f"- Steps remaining: {log['state']['steps_remaining']}")
            prompt_parts.append(f"- Action: {str(log['action'])}")
            prompt_parts.append(f"- Cleared: {log['cleared_count']}")
        
        # Add analysis instructions
        prompt_parts.append("\n## Analysis Tasks:")
        prompt_parts.append("""
1. Provide a concise summary of the agent's experience and performance in this match game (2-3 sentences).

2. List the strengths demonstrated in this session. Focus on effective strategies, good decision-making, and successful techniques. Provide as many as you can identify.

3. List the weaknesses or areas for improvement from this session. Focus on missed opportunities, inefficient moves, and strategic errors. Provide as many as you can identify.

Please format your response as follows:

Experience Summary:
[Your 2-3 sentence summary]

Strengths:
- [Strength 1]
- [Strength 2]
- [Strength 3]
...

Weaknesses:
- [Weakness 1]
- [Weakness 2]
- [Weakness 3]
...
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_reflection(self, reflection_text: str) -> Tuple[str, List[str], List[str]]:
        """
        Parse reflection response.
        
        Args:
            reflection_text: Reflection text from agent
            
        Returns:
            (experience_summary, strengths, weaknesses): Extracted insights
        """
        # Extract experience summary
        experience_summary = ""
        summary_section = re.search(r"Experience Summary:(.*?)(?:\n\n|\nStrengths:)", reflection_text, re.DOTALL)
        if summary_section:
            experience_summary = summary_section.group(1).strip()
        
        # Extract strengths
        strengths = []
        strengths_section = re.search(r"Strengths:(.*?)(?:\n\n|\nWeaknesses:)", reflection_text, re.DOTALL)
        if strengths_section:
            strengths_text = strengths_section.group(1)
            strengths = [s.strip()[2:].strip() for s in strengths_text.split("\n") if s.strip().startswith("-")]
        
        # Extract weaknesses
        weaknesses = []
        weaknesses_section = re.search(r"Weaknesses:(.*?)(?:\n\n|$)", reflection_text, re.DOTALL)
        if weaknesses_section:
            weaknesses_text = weaknesses_section.group(1)
            weaknesses = [w.strip()[2:].strip() for w in weaknesses_text.split("\n") if w.strip().startswith("-")]
        
        return experience_summary, strengths, weaknesses
    
    def record_subjective_memory(self, experience_summary: str, strengths: List[str], weaknesses: List[str]):
        """
        Record subjective memory.
        
        Args:
            experience_summary: Experience summary
            strengths: List of strengths
            weaknesses: List of weaknesses
        """
        self.memory_manager.add_subjective_memory(
            self.difficulty,
            self.level,
            experience_summary,
            strengths,
            weaknesses,
            self.session_metrics
        )
    
    def validate_subjective_memory(self, prev_metrics: Dict[str, Any], new_metrics: Dict[str, Any]) -> bool:
        """
        Validate subjective memory.
        
        Args:
            prev_metrics: Previous metrics
            new_metrics: New metrics
            
        Returns:
            Whether memory is valid
        """
        return self.memory_manager.validate_subjective_memory(
            self.difficulty,
            self.level,
            prev_metrics,
            new_metrics
        )
    
    def promote_memory_to_truth(self):
        """Promote subjective memory to truth knowledge."""
        memory = self.memory_manager.get_subjective_memory(self.difficulty, self.level)
        
        if not memory:
            print(f"No subjective memory found for {self.difficulty} level {self.level}")
            return
        
        # Extract memory items
        memory_items = []
        
        # Add experience summary if it exists
        if memory.get("experience_summary"):
            memory_items.append(memory["experience_summary"])
        
        # Add strengths
        if "strengths" in memory:
            memory_items.extend(memory["strengths"])
        
        # Add weaknesses
        if "weaknesses" in memory:
            memory_items.extend(memory["weaknesses"])
        
        # Promote to truth knowledge
        self.memory_manager.promote_to_truth(memory_items, self.difficulty, self.level)
        
        print(f"Promoted memory from {self.difficulty} level {self.level} to truth knowledge")
        
        # Optimize truth knowledge
        self.optimize_truth_knowledge()
    
    def optimize_truth_knowledge(self):
        """Optimize truth knowledge base."""
        # Define reflection function to pass to memory manager
        def reflect_func(prompt):
            """Helper function to pass to memory manager for reflection."""
            response = self.api_client.get_response([
                {"role": "system", "content": "You are an AI assistant specializing in knowledge management, optimizing entries in knowledge bases by removing duplicates and merging similar items."},
                {"role": "user", "content": prompt}
            ])
            return response
        
        # Call memory manager's optimize function
        self.memory_manager.optimize_truth_knowledge(reflect_func)
    
    def clear_current_subjective_memory(self):
        """Clear current level's subjective memory."""
        self.memory_manager.clear_subjective_memory(self.difficulty, self.level)
    
    def _create_game_state_prompt(self, state):
        """
        Create user prompt with game state.
        
        Args:
            state: Game state
            
        Returns:
            User prompt string
        """
        return f"""
Board:
{json.dumps(state.get('board'), indent=2)}
Score: {state.get('score')}
Steps remaining: {state.get('steps_remaining')}
Inventory: {json.dumps(state.get('inventory'))}
Color targets: {json.dumps(state.get('color_targets'))}
Current color counts: {json.dumps(state.get('color_counts'))}
Suggest the best action to achieve the level completion with maximum score and minimum steps.
""" 