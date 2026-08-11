import requests
import json
import os
from typing import Dict, Any, List, Tuple, Optional
import random

from src.agent.map_processor import MapProcessor
from src.config.game_config import *
from src.config.api_config import (
    DEFAULT_API_TYPE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from src.agent.gpt_client import GPTClient

class DeepSeekAgent:
    """使用DeepSeek V3 API或OpenAI兼容API的智能代理接口"""
    
    def __init__(self, api_key: str = None, model: str = None, api_type: str = DEFAULT_API_TYPE, base_url: str = None):
        """
        初始化AI代理
        
        Args:
            api_key: API密钥
            model: 使用的模型，默认为gpt-4
            api_type: API类型，可选"deepseek"或"openai"
            base_url: 自定义API基础URL，仅当api_type="openai"时使用
        """
        # 初始化系统提示
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
- 7x7 grid size
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
- 11x11 grid size
- Contains monsters and special items
- Items available:
  * T: Shovel (break walls, 3 uses)
  * W: Sword (defeat monsters)
  * N: Magnet (attract nearby coins)
  * K: Key (required for goal)
- Most complex navigation"""

        self.api_type = (api_type or DEFAULT_API_TYPE).lower()
        if self.api_type == "deepseek":
            self.api_key = api_key or DEEPSEEK_API_KEY
            self.model = model or DEEPSEEK_MODEL
            self.base_url = base_url or DEEPSEEK_BASE_URL
        elif self.api_type == "openai":
            self.api_key = api_key or OPENAI_API_KEY
            self.model = model or OPENAI_MODEL
            self.base_url = base_url or OPENAI_BASE_URL
        else:
            raise ValueError(f"Unsupported api_type: {self.api_type}. Use 'openai' or 'deepseek'.")

        if not self.api_key:
            env_name = "DEEPSEEK_API_KEY" if self.api_type == "deepseek" else "OPENAI_API_KEY"
            raise ValueError(f"Missing API key. Set {env_name} or pass api_key explicitly.")
        
        # 初始化 GPT 客户端
        self.gpt_client = GPTClient(
            key=self.api_key,
            url=self.base_url,
            model=self.model
        )
        
        self.show_prompt = True
        self.mode = MODE_LEVEL2
        
        # 初始化系统提示
        self._update_system_prompt(self.mode)
        self._initialize_conversation()
        
        # 验证API连接
        print(f"正在验证与API的连接...")
        try:
            test_response = self.gpt_client.get_response([
                {"role": "system", "content": "这是一个测试消息。"},
                {"role": "user", "content": "测试连接"}
            ])
            if test_response != 'Failed':
                print(f"✓ API连接成功！使用: {self.api_type.upper()} API, 模型: {self.model}")
            else:
                print("× API连接测试失败：收到Failed响应")
        except Exception as e:
            print(f"× API连接测试失败: {str(e)}")
            print(f"请检查API密钥和网络连接。")
    
    def _update_system_prompt(self, mode):
        """根据游戏模式更新系统提示"""
        self.mode = mode
        
        # 选择适当的提示
        if mode == MODE_LEVEL1:
            level_prompt = self.level1_prompt
        elif mode == MODE_LEVEL2:
            level_prompt = self.level2_prompt
        elif mode == MODE_LEVEL3:
            level_prompt = self.level3_prompt
        else:
            level_prompt = self.level2_prompt  # 默认
        
        # 组合基础提示和特定级别提示
        self.system_prompt = self.base_system_prompt + level_prompt
    
    def _initialize_conversation(self):
        """初始化对话历史"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        调用API
        
        Args:
            messages: 消息列表
            
        Returns:
            API响应
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            )
            response.raise_for_status()
            
            # 解析响应
            response_data = response.json()
            
            # 处理不同API格式的响应
            if self.api_type == "openai":
                # 对OpenAI格式的响应进行转换，使其与DeepSeek格式兼容
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    choice = response_data['choices'][0]
                    if 'message' in choice:
                        # 标准OpenAI格式
                        pass
                    elif 'delta' in choice and 'content' in choice['delta']:
                        # 处理流式响应格式
                        choice['message'] = {'content': choice['delta']['content']}
            
            return response_data
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return {"choices": [{"message": {"content": "API调用失败，无法获取响应。"}}]}
    
    def get_action(self, game_state: Dict[str, Any]) -> Tuple[int, Dict]:
        """
        获取代理的下一个动作
        
        Args:
            game_state: 包含游戏当前状态的字典
            
        Returns:
            (action, response_dict): 动作编号和包含完整响应信息的字典
        """
        try:
            # 格式化地图信息
            map_info = MapProcessor.format_map_for_agent(
                grid=game_state['grid'],
                vision_map=game_state['vision_map'],
                agent_pos=game_state['agent_pos'],
                coins=game_state.get('coins', set()),
                lives=game_state.get('lives', 0),
                score=game_state.get('score', 0),
                mode=self.mode,
                monsters=game_state.get('monsters', []),
                obstacles=game_state.get('obstacles', set())
            )
            
            # 添加动作说明
            actions_info = MapProcessor.get_actions_description()
            
            # 添加道具状态信息
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
            
            # 构建消息
            message = f"""Current Game State:
{map_info}

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

            # 根据游戏模式添加特定提示
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
            
            # 显示发送给模型的提示（如果开启了显示选项）
            if self.show_prompt:
                print("\n" + "="*80)
                print(f"发送给{self.api_type.upper()}模型的提示:")
                print("="*80)
                print(message)
                print("="*80 + "\n")
            
            # 创建消息列表
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ]
            
            # 使用 GPTClient 获取响应
            content = self.gpt_client.get_response(messages)
            
            # 如果API调用失败，返回随机动作
            if content == 'Failed':
                print("API调用失败，使用随机动作")
                return random.randint(0, 11), {
                    "prompt": message,
                    "choices": [{"message": {"content": "API调用失败，使用随机动作"}}],
                    "system_prompt": self.system_prompt
                }
            
            # 从回复中提取动作编号
            action = self._extract_action(content)
            
            # 创建包含完整信息的响应字典
            response_dict = {
                "prompt": message,
                "choices": [{"message": {"content": content}}],
                "system_prompt": self.system_prompt
            }
            
            return action, response_dict
            
        except Exception as e:
            print(f"获取动作时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 如果出错，返回默认动作和空响应
            return 0, {
                "prompt": "Error occurred",
                "choices": [{"message": {"content": "发生错误，默认向上移动1步。"}}],
                "usage": {}
            }
    
    def _extract_action(self, content: str) -> int:
        """
        从API响应中提取动作编号
        
        Args:
            content: API响应内容
            
        Returns:
            动作编号，如果无法提取则返回0（向上移动1步）
        """
        # 寻找可能包含动作编号的文本模式
        import re
        
        # 首先，尝试找"Action:"标记后的数字（最精确的方式）
        action_choice_pattern = r"Action\s*[:：]\s*(\d+)"
        matches = re.search(action_choice_pattern, content)
        if matches:
            action = int(matches.group(1))
            # 确保动作在有效范围内
            if 0 <= action <= 11:
                return action
        
        # 尝试其他模式
        patterns = [
            r"action\s*(\d+)",
            r"action number[：:]\s*(\d+)",
            r"choose action[：:]\s*(\d+)",
            r"I choose action[：:]\s*(\d+)",
            r"choice[：:]\s*(\d+)",
            r"(\d+)\s*[。，,.]?\s*$"  # 句子结尾的数字
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, content)
            if matches:
                action = int(matches.group(1))
                # 确保动作在有效范围内
                if 0 <= action <= 11:
                    return action
        
        # 如果未找到匹配，尝试查找动作描述
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
        
        # 如果仍未找到，默认返回0
        return 0
    
    def reset(self):
        """重置对话历史"""
        self._initialize_conversation()
    
    def set_mode(self, mode):
        """设置游戏模式并更新系统提示"""
        if mode != self.mode:
            self._update_system_prompt(mode)
            self._initialize_conversation()
    
    def _test_api_connection(self):
        """测试API连接"""
        test_message = "测试连接"
        test_messages = [
            {"role": "system", "content": "这是一个测试消息。"},
            {"role": "user", "content": test_message}
        ]
        
        payload = {
            "model": self.model,
            "messages": test_messages,
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def set_show_prompt(self, show: bool):
        """设置是否显示发送给模型的提示"""
        self.show_prompt = show 
