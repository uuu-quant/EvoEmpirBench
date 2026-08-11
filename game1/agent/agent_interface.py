"""
Agent interface module for Game 1.
Provides standardized interface for AI agents to interact with the game.
"""

import re
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from ...common.gpt_client import GPTClient
from .map_processor import MapProcessor
from ..game.config import *

class Agent:
    """Base agent interface for maze navigation game."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4", api_type: str = "openai", 
                 base_url: str = None, truth_knowledge: List[Dict[str, str]] = None):
        """
        Initialize agent.
        
        Args:
            api_key: API key for LLM
            model: Model name
            api_type: API type (openai or compatible)
            base_url: Base URL for API
            truth_knowledge: List of truth knowledge entries
        """
        # Initialize system prompts
        self.base_system_prompt = """You are an intelligent agent solving a maze problem. Your task is to navigate through the maze efficiently while collecting rewards and avoiding dangers.

Core Game Elements:
- A: Your current position
- G: Goal (always visible)
- C: Coin (+500 points)
- #: Wall (costs life if hit)
- ?: Unexplored area
- .: Empty space

Your Priorities (in order):
1. Stay alive (avoid walls/monsters)
2. Reach the goal
3. Collect coins when safe
4. Explore efficiently
5. Minimize steps

You will receive the current game state and must choose an action (0-11) based on careful analysis of the situation.
Always explain your reasoning before making a decision."""

        self.level1_prompt = """
Level 1 Characteristics:
- 9x9 grid size
- No monsters
- 5 coins to collect
- Focus on basic navigation and coin collection"""

        self.level2_prompt = """
Level 2 Characteristics:
- 9x9 grid size
- Contains monsters (M) that move randomly
- 5 coins to collect
- Requires careful planning to avoid monsters"""

        self.level3_prompt = """
Level 3 Characteristics:
- 9x9 grid size
- Contains monsters and special items
- Items available:
  * T: Shovel (break walls, 3 uses)
  * W: Sword (defeat monsters)
  * N: Magnet (attract nearby coins)
  * K: Key (required for goal)
- Most complex navigation"""

        # Initialize API client
        self.api_key = api_key
        self.model = model
        self.api_type = api_type.lower()
        self.base_url = base_url
        
        # Store truth knowledge
        self.truth_knowledge = truth_knowledge
        
        # Initialize GPT client
        self.gpt_client = GPTClient(
            key=self.api_key,
            url=self.base_url,
            model=self.model
        )
        
        # Set default mode and update system prompt
        self.mode = MODE_LEVEL2
        self._update_system_prompt(self.mode)
        self._initialize_conversation()
        
        # Debug flag
        self.show_prompt = False
    
    def _update_system_prompt(self, mode):
        """
        Update system prompt based on game mode.
        
        Args:
            mode: Game mode to use for prompt
        """
        self.mode = mode
        
        # Choose appropriate level-specific prompt
        if mode == MODE_LEVEL1:
            level_prompt = self.level1_prompt
        elif mode == MODE_LEVEL2:
            level_prompt = self.level2_prompt
        elif mode == MODE_LEVEL3:
            level_prompt = self.level3_prompt
        else:
            level_prompt = self.level2_prompt  # Default
        
        # Combine base prompt and level-specific prompt
        self.system_prompt = self.base_system_prompt + level_prompt
        
        # Add truth knowledge to system prompt if available
        if self.truth_knowledge:
            truth_prompt = "\n\nTraining Knowledge (use these insights to make better decisions):\n"
            
            # Add truth knowledge entries
            for i, knowledge in enumerate(self.truth_knowledge):
                if 'knowledge' in knowledge:
                    # Limit length to avoid exceeding context window
                    if len(truth_prompt) < 5000:
                        truth_prompt += f"{i+1}. {knowledge['knowledge']}\n"
            
            # Add truth knowledge to system prompt
            self.system_prompt += truth_prompt
    
    def _initialize_conversation(self):
        """Initialize conversation history."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def get_action(self, game_state: Dict[str, Any]) -> Tuple[int, Dict]:
        """
        Get agent's next action.
        
        Args:
            game_state: Current game state
            
        Returns:
            (action, response_dict): Action number and complete response information
        """
        try:
            # Check if full vision mode is active
            vision_map = np.array(game_state['vision_map'])
            is_full_vision = np.all(vision_map == DISCOVERED)
            
            # Format monster positions
            monsters = game_state.get('monsters', [])
            if monsters and isinstance(monsters[0], list):
                # Convert list format [[x1, y1], [x2, y2]] to tuple format [(x1, y1), (x2, y2)]
                monsters = [tuple(pos) for pos in monsters]
            
            # Format map for agent
            map_info = MapProcessor.format_map_for_agent(
                grid=np.array(game_state['grid']),
                vision_map=vision_map,
                agent_pos=tuple(game_state['agent_pos']),
                coins=set(tuple(pos) for pos in game_state.get('coins', [])),
                lives=game_state.get('lives', 0),
                score=game_state.get('score', 0),
                mode=self.mode,
                monsters=monsters,
                obstacles=set(tuple(pos) for pos in game_state.get('obstacles', []))
            )
            
            # Get action descriptions
            actions_info = MapProcessor.get_actions_description()
            
            # Add item status information for Level 3
            items_info = ""
            if self.mode == MODE_LEVEL3:
                items_info = "\nCurrent items status:\n"
                
                if game_state.get('has_shovel', False):
                    items_info += f"- Shovel: Equipped (Uses remaining: {game_state.get('shovel_uses', 0)})\n"
                else:
                    items_info += "- Shovel: Not equipped\n"
                    
                if game_state.get('has_sword', False):
                    items_info += "- Sword: Equipped\n"
                else:
                    items_info += "- Sword: Not equipped\n"
                    
                if game_state.get('has_magnet', False):
                    items_info += "- Magnet: Equipped\n"
                else:
                    items_info += "- Magnet: Not equipped\n"
                    
                if game_state.get('has_key', False):
                    items_info += "- Key: Collected\n"
                else:
                    items_info += "- Key: Not collected (required to finish)\n"
            
            # Add full vision notice
            vision_info = ""
            if is_full_vision:
                vision_info = "\nNOTE: You have full vision of the entire map. You can see all obstacles, coins, monsters, and items without the need to explore.\n"
            
            # Construct message
            message = f"""Current Game State:
{map_info}
{vision_info}
Game Status:
- Score: {game_state.get('score', 0)}
- Lives: {game_state.get('lives', 3)}
- Current Position (row,col): ({game_state['agent_pos'][0]},{game_state['agent_pos'][1]})

Movement System:
- Actions 0-2: Move UP (row-1) [1/2/3 steps]
- Actions 3-5: Move DOWN (row+1) [1/2/3 steps]
- Actions 6-8: Move LEFT (col-1) [1/2/3 steps]
- Actions 9-11: Move RIGHT (col+1) [1/2/3 steps]

Available Actions:
{actions_info}
{items_info}

Scoring System:
- New cell explored: +10 points
- Coin collected: +500 points
- Step taken: -50 points
- Life lost: -1000 points
- Goal reached: +2000 points

Please analyze the current situation and choose your next action:
1. Analyze visible area and potential risks
2. Consider exploration value and rewards
3. Choose action number (0-11)
"""

            # Add game mode specific hints
            if self.mode == MODE_LEVEL2 or self.mode == MODE_LEVEL3:
                message += "- You cannot touch monsters (M), or you'll lose a life and return to start\n"
            
            if self.mode == MODE_LEVEL3:
                message += """
Special rules for Level 3:
- You must collect the key (K) before you can enter the goal
- With a shovel (T), you can break through walls without losing lives (3 uses)
- With a sword (W), you can defeat monsters without losing lives
- With a magnet (N), you can collect coins in a 5x5 area around you
"""

            message += """
The ultimate goal is to explore the map, collect coins, reach the goal, while maintaining a high score

Response format:
1. First analyze the current situation, including explored areas, coin positions, and potential risks
2. Consider possible movement options and their consequences, especially focusing on exploration value
3. Finally, provide your choice using the format "Action: X" where X is a number between 0-11

For example:
After analysis, you should write "Action: 9" to indicate moving right by 1 step
"""
            
            # Show prompt if debug flag is set
            if self.show_prompt:
                print("\n" + "="*80)
                print(f"Prompt sent to model: {self.model}")
                print("="*80)
                print(message)
                print("="*80 + "\n")
            
            # Create messages list
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ]
            
            # Get response from LLM
            content = self.gpt_client.get_response(messages)
            
            # Handle API failure
            if content == 'Failed':
                print("API call failed, using random action")
                return random.randint(0, 11), {
                    "prompt": message,
                    "choices": [{"message": {"content": "API call failed, using random action"}}],
                    "system_prompt": self.system_prompt
                }
            
            # Extract action from response
            action = self._extract_action(content)
            
            # Create response dictionary
            response_dict = {
                "prompt": message,
                "choices": [{"message": {"content": content}}],
                "system_prompt": self.system_prompt
            }
            
            return action, response_dict
            
        except Exception as e:
            print(f"Error getting action: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return default action on error
            return 0, {
                "prompt": "Error occurred",
                "choices": [{"message": {"content": "Error occurred, defaulting to UP 1 step."}}],
                "usage": {}
            }
    
    def _extract_action(self, content: str) -> int:
        """
        Extract action number from model response.
        
        Args:
            content: Response content from LLM
            
        Returns:
            Action number (0-11), defaults to 0 if extraction fails
        """
        # Look for "Action:" pattern
        action_pattern = r"Action\s*[:：]\s*(\d+)"
        matches = re.search(action_pattern, content)
        if matches:
            action = int(matches.group(1))
            # Validate action range
            if 0 <= action <= 11:
                return action
        
        # Try alternative patterns
        patterns = [
            r"action\s*(\d+)",
            r"action number[：:]\s*(\d+)",
            r"choose action[：:]\s*(\d+)",
            r"I choose action[：:]\s*(\d+)",
            r"choice[：:]\s*(\d+)",
            r"(\d+)\s*[。，,.]?\s*$"  # Number at end of sentence
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, content)
            if matches:
                action = int(matches.group(1))
                # Validate action range
                if 0 <= action <= 11:
                    return action
        
        # Look for action descriptions
        action_descriptions = {
            "move up 1 step": 0, "move up one step": 0,
            "move up 2 steps": 1, "move up two steps": 1,
            "move up 3 steps": 2, "move up three steps": 2,
            "move down 1 step": 3, "move down one step": 3,
            "move down 2 steps": 4, "move down two steps": 4,
            "move down 3 steps": 5, "move down three steps": 5,
            "move left 1 step": 6, "move left one step": 6,
            "move left 2 steps": 7, "move left two steps": 7,
            "move left 3 steps": 8, "move left three steps": 8,
            "move right 1 step": 9, "move right one step": 9,
            "move right 2 steps": 10, "move right two steps": 10,
            "move right 3 steps": 11, "move right three steps": 11
        }
        
        for desc, action in action_descriptions.items():
            if desc.lower() in content.lower():
                return action
        
        # Default to action 0 if no match found
        return 0
    
    def reset(self):
        """Reset conversation history."""
        self._initialize_conversation()
    
    def set_mode(self, mode):
        """
        Set game mode and update prompts.
        
        Args:
            mode: Game mode to set
        """
        if mode != self.mode:
            self._update_system_prompt(mode)
            self._initialize_conversation()
    
    def set_show_prompt(self, show: bool):
        """
        Set whether to show prompts for debugging.
        
        Args:
            show: Whether to show prompts
        """
        self.show_prompt = show 