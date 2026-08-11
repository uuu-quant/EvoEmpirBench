#!/usr/bin/env python
"""
Learning agent for the Maze Navigation environment.
Uses GPT for action selection and reflection.
"""

import os
import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ...common.gpt_client import GPTClient
from ..game.config import MODE_LEVEL1, MODE_LEVEL2, MODE_LEVEL3

class MazeNavigationAgent:
    """
    Learning agent for the Maze Navigation environment.
    Uses GPT for decision making with optional truth knowledge.
    """
    
    def __init__(self, api_client: GPTClient, memory_dir: str = None, 
                results_dir: str = None, truth_knowledge: List[Dict] = None):
        """
        Initialize the learning agent.
        
        Args:
            api_client: GPT client for API calls
            memory_dir: Directory to store memory
            results_dir: Directory to store results
            truth_knowledge: List of truth knowledge items
        """
        self.api_client = api_client
        self.memory_dir = memory_dir
        self.results_dir = results_dir
        self.truth_knowledge = truth_knowledge
        
        # Session variables
        self.current_mode = None
        self.current_map = None
        self.session_start_time = None
        self.session_id = None
        self.interactions = []
        
        # Create memory directory if needed
        if memory_dir:
            os.makedirs(memory_dir, exist_ok=True)
    
    def start_session(self, mode: str, map_index: int):
        """
        Start a new agent session.
        
        Args:
            mode: Game mode/difficulty
            map_index: Map index
        """
        self.current_mode = mode
        self.current_map = map_index
        self.session_start_time = time.time()
        self.session_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{mode}_{map_index}"
        self.interactions = []
        
        print(f"Starting agent session for {mode} map {map_index}")
    
    def observe(self, state: np.ndarray, info: Dict[str, Any]):
        """
        Store initial observation.
        
        Args:
            state: Environment state
            info: Additional information
        """
        # Store initial observation (we don't take an action yet)
        self.initial_state = state.copy()
        self.initial_info = info.copy()
    
    def get_action(self, state: np.ndarray, info: Dict[str, Any], with_memory: bool = False) -> Tuple[int, str]:
        """
        Get action from the agent.
        
        Args:
            state: Environment state
            info: Additional information
            with_memory: Whether to use memory for action selection
        
        Returns:
            Tuple of (action, response)
        """
        # Create prompt for the API
        prompt = self._create_prompt(state, info, with_memory)
        
        # Get response from API
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.api_client.get_response(messages)
            
            # Parse action from response
            action = self._parse_action(response)
            
            if action is None:
                print("Failed to parse action from response")
                # Fallback to random action
                action = random.randint(0, 3)
            
            return action, response
        except Exception as e:
            print(f"Error getting action: {str(e)}")
            return None, str(e)
    
    def _create_prompt(self, state: np.ndarray, info: Dict[str, Any], with_memory: bool = False) -> str:
        """
        Create prompt for the API.
        
        Args:
            state: Environment state
            info: Additional information
            with_memory: Whether to use memory
        
        Returns:
            Prompt string
        """
        # Convert state to readable format
        state_desc = self._format_state(state)
        
        # Extract key information
        player_pos = info.get("agent_position", (0, 0))
        visible_grid = info.get("visible_grid", [])
        inventory = info.get("inventory", {})
        goal_pos = info.get("goal_position", None)
        coins = info.get("coins", [])
        visible_coins = info.get("visible_coins", [])
        monsters = info.get("monsters", [])
        visible_monsters = info.get("visible_monsters", [])
        obstacles = info.get("obstacles", [])
        visible_obstacles = info.get("visible_obstacles", [])
        lives = info.get("lives", 3)
        
        prompt = f"""
I am playing a maze navigation game where I control an agent in a grid-based environment.

Current situation:
- Mode: {self.current_mode}
- Player position: {player_pos}
- Goal position: {goal_pos if goal_pos is not None and info.get("goal_visible", False) else "Unknown"}
- Lives remaining: {lives}
- Coins collected: {info.get("collected_coins", 0)}
- Inventory: {inventory}
- Visible coins: {visible_coins}
- Visible monsters: {visible_monsters}
- Visible obstacles: {visible_obstacles}

Visible area:
{state_desc}

I need to navigate to the goal while collecting coins and avoiding/dealing with obstacles and monsters.

Possible actions:
0: Move UP
1: Move RIGHT
2: Move DOWN
3: Move LEFT
4: Use SHOVEL (if available)
5: Use SWORD (if available)
6: Use MAGNET (if available)
7: Use KEY (if available)

Please give me the best action to take next. Respond with a single number from 0-7 representing the action.
"""
        
        # Add additional context for different modes
        if self.current_mode == MODE_LEVEL3:
            prompt += """
This is a difficult level with monsters that can kill me and obstacles that block my path.
I should use the SWORD to kill monsters and the SHOVEL to remove obstacles when necessary.
"""
        
        # Add memory if requested
        if with_memory and self.interactions:
            # Add recent interaction history
            recent_interactions = self.interactions[-5:] if len(self.interactions) > 5 else self.interactions
            
            prompt += "\nRecent interactions:\n"
            for i, interaction in enumerate(recent_interactions):
                prompt += f"Step {i+1}:\n"
                prompt += f"- State: {interaction['info']}\n"
                prompt += f"- Action taken: {interaction['action']}\n"
                prompt += f"- Result: Reward {interaction['reward']}\n"
        
        return prompt
    
    def _format_state(self, state: np.ndarray) -> str:
        """
        Format state matrix into a readable string.
        
        Args:
            state: Environment state
        
        Returns:
            Formatted string
        """
        if len(state.shape) == 3:  # Multi-channel state
            # Extract the main channel (usually the first one contains the walls/obstacles)
            visible_area = state[:, :, 0]
        else:
            visible_area = state
        
        # Convert to string representation
        state_str = ""
        for row in visible_area:
            state_str += "".join([str(int(cell)) if cell != 0 else "." for cell in row]) + "\n"
        
        return state_str
    
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for the API.
        
        Returns:
            System prompt string
        """
        system_prompt = """
You are an AI assistant helping navigate a maze-like environment. Your goal is to guide the agent to the goal position while collecting coins and avoiding dangers.

In this environment:
- The agent can move in four directions: UP (0), RIGHT (1), DOWN (2), LEFT (3)
- The agent can use items if they're in the inventory: SHOVEL (4), SWORD (5), MAGNET (6), KEY (7)
- Walls and obstacles block movement
- Coins can be collected for points
- Monsters should be avoided or defeated with a SWORD
- Some obstacles can be removed with a SHOVEL
- MAGNET can attract coins from a distance
- KEY can unlock special doors

Your response should be a single number from 0-7 representing the best action to take.
"""
        
        # Add truth knowledge if available
        if self.truth_knowledge:
            truth_prompt = "\n\nThe following are established truths that you should use to guide your decisions:\n"
            for i, truth in enumerate(self.truth_knowledge):
                knowledge = truth.get("knowledge", "")
                if knowledge:
                    truth_prompt += f"{i+1}. {knowledge}\n"
            
            system_prompt += truth_prompt
        
        return system_prompt
    
    def _parse_action(self, response: str) -> Optional[int]:
        """
        Parse action from API response.
        
        Args:
            response: API response
        
        Returns:
            Action as an integer, or None if parsing fails
        """
        try:
            # First, try to extract just the number
            import re
            numbers = re.findall(r'\b[0-7]\b', response)
            if numbers:
                return int(numbers[0])
            
            # If that fails, try to find any integer in the response
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.isdigit() and 0 <= int(line) <= 7:
                    return int(line)
            
            # If still no action found, check for action keywords
            action_keywords = {
                "up": 0, "north": 0,
                "right": 1, "east": 1,
                "down": 2, "south": 2,
                "left": 3, "west": 3,
                "shovel": 4,
                "sword": 5,
                "magnet": 6,
                "key": 7
            }
            
            for keyword, action in action_keywords.items():
                if keyword in response.lower():
                    return action
            
            return None
        except Exception as e:
            print(f"Error parsing action: {str(e)}")
            return None
    
    def log_interaction(self, state: np.ndarray, action: int, response: str, reward: float, info: Dict[str, Any]):
        """
        Log interaction for learning.
        
        Args:
            state: Environment state
            action: Action taken
            response: API response
            reward: Reward received
            info: Additional information
        """
        interaction = {
            "timestamp": time.time(),
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action": int(action),
            "response": response,
            "reward": float(reward),
            "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list, dict))}
        }
        
        self.interactions.append(interaction)
    
    def end_session(self, success: bool, score: float, exploration_rate: float, steps_used: int):
        """
        End the agent session.
        
        Args:
            success: Whether the session was successful
            score: Final score
            exploration_rate: Exploration rate
            steps_used: Number of steps used
        """
        # Calculate session duration
        session_duration = time.time() - self.session_start_time
        
        # Create session summary
        session_summary = {
            "session_id": self.session_id,
            "mode": self.current_mode,
            "map_index": self.current_map,
            "success": success,
            "score": score,
            "exploration_rate": exploration_rate,
            "steps_used": steps_used,
            "duration": session_duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save session data if memory directory is available
        if self.memory_dir:
            # Create mode-specific directory
            mode_dir = os.path.join(self.memory_dir, self.current_mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            # Save session summary
            summary_file = os.path.join(mode_dir, f"session_{self.session_id}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(session_summary, f, indent=2)
            
            # Save interactions
            interactions_file = os.path.join(mode_dir, f"session_{self.session_id}_interactions.json")
            with open(interactions_file, 'w') as f:
                json.dump(self.interactions, f, indent=2)
            
            print(f"Session data saved to {mode_dir}")
        
        print(f"Session ended: success={success}, score={score}, steps={steps_used}")
        
        # Clear session data
        self.interactions = []
    
    def reflect_on_session(self) -> Tuple[str, List[str], List[str]]:
        """
        Generate a reflection on the session.
        
        Returns:
            Tuple of (experience_summary, strengths, weaknesses)
        """
        if not self.interactions:
            return "No interactions to reflect on.", [], []
        
        # Create a summary of the session
        total_reward = sum(interaction["reward"] for interaction in self.interactions)
        num_steps = len(self.interactions)
        
        # Create prompt for reflection
        reflection_prompt = f"""
I need to reflect on my performance in a maze navigation game.

Session summary:
- Mode: {self.current_mode}
- Map: {self.current_map}
- Steps taken: {num_steps}
- Total reward: {total_reward}

Please analyze my performance and provide:
1. A brief summary of what happened during this session (2-3 sentences)
2. Three strengths I demonstrated 
3. Three areas where I could improve

Format your response as:
Summary: [your summary]

Strengths:
- [strength 1]
- [strength 2]
- [strength 3]

Weaknesses:
- [weakness 1]
- [weakness 2]
- [weakness 3]
"""
        
        # Get response from API
        messages = [
            {"role": "system", "content": "You are an AI coach that helps analyze game performance."},
            {"role": "user", "content": reflection_prompt}
        ]
        
        try:
            response = self.api_client.get_response(messages)
            
            # Parse the reflection
            experience_summary, strengths, weaknesses = self._parse_reflection(response)
            
            return experience_summary, strengths, weaknesses
        except Exception as e:
            print(f"Error generating reflection: {str(e)}")
            return "Failed to generate reflection.", [], []
    
    def _parse_reflection(self, response: str) -> Tuple[str, List[str], List[str]]:
        """
        Parse reflection from API response.
        
        Args:
            response: API response
        
        Returns:
            Tuple of (experience_summary, strengths, weaknesses)
        """
        experience_summary = ""
        strengths = []
        weaknesses = []
        
        try:
            # Split response into sections
            sections = response.split("\n\n")
            
            # Extract summary
            for section in sections:
                if section.lower().startswith("summary:"):
                    experience_summary = section.replace("Summary:", "").strip()
                    break
            
            # Extract strengths
            strengths_section = ""
            for section in sections:
                if "strengths" in section.lower():
                    strengths_section = section
                    break
            
            if strengths_section:
                for line in strengths_section.split("\n"):
                    if line.strip().startswith("-") or line.strip().startswith("*"):
                        strength = line.strip().replace("-", "").replace("*", "").strip()
                        if strength:
                            strengths.append(strength)
            
            # Extract weaknesses
            weaknesses_section = ""
            for section in sections:
                if "weaknesses" in section.lower():
                    weaknesses_section = section
                    break
            
            if weaknesses_section:
                for line in weaknesses_section.split("\n"):
                    if line.strip().startswith("-") or line.strip().startswith("*"):
                        weakness = line.strip().replace("-", "").replace("*", "").strip()
                        if weakness:
                            weaknesses.append(weakness)
            
            # Ensure we have at least some data
            if not experience_summary:
                experience_summary = "Completed a maze navigation session."
            
            if not strengths:
                strengths = ["Completed the session."]
            
            if not weaknesses:
                weaknesses = ["Could improve overall strategy."]
            
            return experience_summary, strengths[:3], weaknesses[:3]
        except Exception as e:
            print(f"Error parsing reflection: {str(e)}")
            return "Completed a maze navigation session.", ["Completed the session."], ["Could improve overall strategy."]
    
    def record_subjective_memory(self, experience_summary: str, strengths: List[str], weaknesses: List[str]):
        """
        Record subjective memory from reflection.
        
        Args:
            experience_summary: Summary of the experience
            strengths: List of strengths
            weaknesses: List of weaknesses
        """
        if not self.memory_dir:
            return
        
        # Create memory entry
        memory = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
            "mode": self.current_mode,
            "map_index": self.current_map,
            "experience_summary": experience_summary,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
        # Save to memory file
        memory_file = os.path.join(self.memory_dir, "subjective_memories.json")
        
        try:
            # Load existing memories if file exists
            existing_memories = []
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    existing_memories = json.load(f)
            
            # Add new memory
            existing_memories.append(memory)
            
            # Save updated memories
            with open(memory_file, 'w') as f:
                json.dump(existing_memories, f, indent=2)
            
            print(f"Subjective memory recorded to {memory_file}")
        except Exception as e:
            print(f"Error recording subjective memory: {str(e)}") 