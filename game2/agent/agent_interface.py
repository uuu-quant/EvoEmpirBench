"""
Agent interface for Match-3 game.
Defines the interface for interacting with the match-3 game environment.
"""

import json
import time
import re
from typing import Dict, List, Any, Tuple, Optional

class MatchGameAgentInterface:
    """
    Interface for agents interacting with match-3 game.
    This class provides a standard API for different agent implementations to interact with the game.
    """
    
    def __init__(self, api_client=None):
        """
        Initialize the agent interface.
        
        Args:
            api_client: Optional API client for LLM-based agents
        """
        self.api_client = api_client
        self.system_prompt = """
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
    
    def get_action(self, state: Dict[str, Any], with_memory: bool = True) -> Tuple[Dict[str, Any], str]:
        """
        Get next action from the agent.
        
        Args:
            state: Current game state
            with_memory: Whether to use memory
            
        Returns:
            (action, raw_response): The selected action and raw API response
        """
        if not self.api_client:
            return None, "No API client available"
        
        # Build user prompt with game state
        user_prompt = f"""
Board:
{json.dumps(state['board'], indent=2)}
Score: {state['score']}
Steps remaining: {state['steps_remaining']}
Inventory: {json.dumps(state['inventory'])}
Color targets: {json.dumps(state['color_targets'])}
Current color counts: {json.dumps(state['color_counts'])}
Suggest the best action to achieve the level completion with maximum score and minimum steps.
"""
        
        # Add memory if requested (implemented by subclasses)
        if with_memory and hasattr(self, 'get_memory_prompt'):
            memory_prompt = self.get_memory_prompt()
            if memory_prompt:
                user_prompt += "\n\nAdditional game knowledge based on previous experience:\n" + memory_prompt
        
        # Prepare messages for API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call API
        response = self.api_client.get_response(messages)
        
        # Parse action from response
        action = self._parse_action_from_response(response)
        
        return action, response
    
    def _parse_action_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract action from model response.
        
        Args:
            response: Raw response string from the model
            
        Returns:
            Parsed action dictionary or None if parsing fails
        """
        if not response or response == 'Failed':
            return None
        
        try:
            # Try to extract JSON block from response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # Try to find JSON structure in the entire response
            action_match = re.search(r'(\{[\s\S]*"action"[\s\S]*\})', response, re.DOTALL)
            if action_match:
                potential_json = action_match.group(1)
                return json.loads(potential_json)
            
            return None
                
        except Exception as e:
            print(f"Error parsing action from response: {str(e)}")
            return None
    
    def reset(self):
        """Reset the agent state."""
        pass  # Implemented by subclasses
    
    def update_system_prompt(self, new_prompt: str):
        """
        Update the system prompt.
        
        Args:
            new_prompt: New system prompt
        """
        self.system_prompt = new_prompt 