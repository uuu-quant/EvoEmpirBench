import os
import sys
import json
import time
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from src.agent.agent_interface import DeepSeekAgent
from src.agent.memory_manager import MemoryManager
from src.game.environment import PathFindingEnv
from src.config.game_config import COINS_COUNT, DISCOVERED
from src.config.paths import (
    AGENT_SESSIONS_DIR,
    MEMORY_PROMOTION_DIR,
    MEMORY_VALIDATION_DIR,
    TRUTH_OPTIMIZATION_DIR,
)

class LearningAgent:
    """
    学习型代理，能够从游戏经验中学习并优化其行为
    """
    
    def __init__(self, api_key: str = None, model: str = None, api_type: str = "openai",
                 base_url: str = None, memory_dir: str = None):
        """
        初始化学习代理
        
        Args:
            api_key: API密钥
            model: 使用的模型名称
            api_type: API类型，"deepseek"或"openai"
            base_url: API基础URL，仅当api_type="openai"时使用
            memory_dir: 记忆保存目录
        """
        # 创建记忆管理器
        self.memory_manager = MemoryManager(memory_dir)
        
        # 创建智能体接口
        self.agent = DeepSeekAgent(
            api_key=api_key,
            model=model,
            api_type=api_type,
            base_url=base_url
        )
        
        # 创建反思智能体接口（用于分析游戏和总结经验）
        self.reflection_agent = DeepSeekAgent(
            api_key=api_key,
            model=model,
            api_type=api_type,
            base_url=base_url
        )
        
        # 当前关卡ID
        self.current_level_id = None
        
        # 当前游戏会话的日志
        self.session_logs = []
        
        # 当前游戏会话的指标
        self.session_metrics = {}
    
    def _get_level_id(self, env: PathFindingEnv) -> str:
        """
        生成关卡ID
        
        Args:
            env: 游戏环境
            
        Returns:
            关卡ID字符串
        """
        return f"{env.mode}_map{env.map_index + 1}"
    
    def start_session(self, env: PathFindingEnv):
        """
        开始新的游戏会话
        
        Args:
            env: 游戏环境
        """
        # 设置当前关卡ID
        self.current_level_id = self._get_level_id(env)
        
        # 更新代理模式
        self.agent.set_mode(env.mode)
        
        # 清空会话日志和指标
        self.session_logs = []
        self.session_metrics = {
            "level_id": self.current_level_id,
            "mode": env.mode,
            "map_index": env.map_index,
            "start_time": time.time(),
            "steps": 0,
            "success": False,
            "score": 0,
            "lives_remaining": env.lives,
            "collected_coins": 0,
            "exploration_rate": 0.0,
            "killed_monsters": 0,
            "destroyed_obstacles": 0,
            "api_calls": 0
        }
        
        print(f"开始关卡 {self.current_level_id} 的游戏会话")
    
    def _convert_to_serializable(self, obj):
        """
        将对象转换为可JSON序列化的格式
        
        Args:
            obj: 待转换的对象
            
        Returns:
            可JSON序列化的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象，转换为字典
            return {k: self._convert_to_serializable(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
        else:
            return obj
    
    def get_action(self, game_state: Dict[str, Any], with_memory: bool = True) -> Tuple[int, Dict]:
        """
        获取代理的下一个动作
        
        Args:
            game_state: 游戏状态
            with_memory: 是否使用记忆增强
            
        Returns:
            (action, response_dict): 动作编号和响应信息
        """
        # 检查是否需要更新提示词
        if with_memory:
            # 构建包含记忆的系统提示
            memory_prompt = self.memory_manager.get_memory_prompt(
                self.current_level_id,
                include_truth=True,                   # 总是包含真理知识
                include_subjective=True               # 包含当前关卡的主观记忆
            )
            
            if memory_prompt:
                # 合并基础提示和记忆提示
                original_prompt = self.agent.system_prompt
                enhanced_prompt = original_prompt + "\n\n" + memory_prompt
                
                # 临时更新系统提示
                self.agent.system_prompt = enhanced_prompt
                
                # 在行动后恢复原始提示
                action, response = self.agent.get_action(game_state)
                self.agent.system_prompt = original_prompt
                
                return action, response
        
        # 不使用记忆增强或没有可用记忆时，直接使用原始提示
        return self.agent.get_action(game_state)
    
    def log_interaction(self, state: Dict[str, Any], action: int, response: Dict[str, Any], reward: float):
        """
        记录智能体与环境的交互
        
        Args:
            state: 游戏状态
            action: 执行的动作
            response: API响应
            reward: 获得的奖励
        """
        # 记录交互日志
        interaction = {
            "step": len(self.session_logs) + 1,
            "state": state,
            "action": action,
            "reward": reward,
            "api_response": {
                "prompt": response.get("prompt"),
                "completion": response["choices"][0]["message"]["content"] if "choices" in response else None
            },
            "timestamp": time.time()
        }
        
        self.session_logs.append(interaction)
        
        # 更新会话指标
        self.session_metrics["steps"] += 1
        self.session_metrics["api_calls"] += 1
    
    def end_session(self, env: PathFindingEnv, success: bool, score: int) -> Dict[str, Any]:
        """
        结束游戏会话并更新指标
        
        Args:
            env: 游戏环境
            success: 是否成功通关
            score: 最终得分
            
        Returns:
            会话指标
        """
        # 更新最终指标
        self.session_metrics.update({
            "end_time": time.time(),
            "duration": time.time() - self.session_metrics["start_time"],
            "success": success,
            "score": score,
            "lives_remaining": env.lives,
            "collected_coins": COINS_COUNT - len(env.coins),
            "exploration_rate": self._compute_exploration_rate(env),
            "killed_monsters": sum(1 for log in self.session_logs if "killed_monsters" in log and log["killed_monsters"] > 0),
            "destroyed_obstacles": sum(1 for log in self.session_logs if "destroyed_obstacles" in log and log["destroyed_obstacles"] > 0)
        })
        
        print(f"结束关卡 {self.current_level_id} 的游戏会话")
        print(f"总步数: {self.session_metrics['steps']}")
        print(f"结果: {'成功' if success else '失败'}")
        print(f"得分: {score}")
        
        # 保存会话日志和指标
        self._save_session_data()
        
        return self.session_metrics
    
    def _compute_exploration_rate(self, env: PathFindingEnv) -> float:
        """计算探索率"""
        grid_size = env.grid.shape[0]
        total_cells = grid_size * grid_size
        obstacles_count = len(env.obstacles)
        explorable_cells = total_cells - obstacles_count
        
        # 统计已探索的单元格数量
        discovered_count = np.sum(env.vision_map == DISCOVERED)
        
        # 确保不会超过总可探索单元格数
        discovered_count = min(discovered_count, explorable_cells)
        
        # 计算探索率
        exploration_rate = discovered_count / explorable_cells if explorable_cells > 0 else 0
        
        # 确保探索率不超过1
        exploration_rate = min(exploration_rate, 1.0)
        
        return exploration_rate
    
    def _save_session_data(self):
        """保存会话数据到文件"""
        # 创建会话数据目录
        session_dir = os.path.join(str(AGENT_SESSIONS_DIR), self.current_level_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 转换会话日志为可序列化格式
        serializable_logs = self._convert_to_serializable(self.session_logs)
        
        # 转换会话指标为可序列化格式
        serializable_metrics = self._convert_to_serializable(self.session_metrics)
        
        try:
            # 保存会话日志
            log_file = os.path.join(session_dir, f"session_log_{timestamp}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_logs, f, indent=2, ensure_ascii=False)
            
            # 保存会话指标
            metrics_file = os.path.join(session_dir, f"session_metrics_{timestamp}.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            
            print(f"会话数据已保存到 {session_dir}")
        except Exception as e:
            print(f"保存会话数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def reflect_on_session(self) -> Tuple[str, List[str], List[str]]:
        """
        反思游戏会话，总结经验、优点和缺点
        
        Returns:
            (经验总结, 优点列表, 缺点列表)
        """
        # 准备反思提示
        reflection_prompt = self._create_reflection_prompt()
        
        # 重置反思代理
        self.reflection_agent.reset()
        
        # 发送反思提示给代理
        reflection_messages = [
            {"role": "system", "content": "You are an AI assistant that helps analyze game playing sessions and extract meaningful insights."},
            {"role": "user", "content": reflection_prompt}
        ]
        
        # 使用反思代理获取反思结果
        reflection_response = self.reflection_agent.gpt_client.get_response(reflection_messages)
        
        # 解析反思结果
        experience_summary, strengths, weaknesses = self._parse_reflection(reflection_response)
        
        print("\n===== 游戏反思 =====")
        print(f"经验总结: {experience_summary}")
        print("优点:")
        for s in strengths:
            print(f"- {s}")
        print("缺点:")
        for w in weaknesses:
            print(f"- {w}")
        print("====================\n")
        
        return experience_summary, strengths, weaknesses
    
    def _create_reflection_prompt(self) -> str:
        """创建反思提示"""
        prompt_parts = [
            "# Game Session Analysis",
            "",
            "Please analyze the following game session and provide insights about the agent's performance.",
            "",
            "## Game Metrics:"
        ]
        
        # 添加会话指标
        for key, value in self.session_metrics.items():
            if key not in ["start_time", "end_time"]:
                prompt_parts.append(f"- {key}: {value}")
        
        # 添加游戏日志摘要
        prompt_parts.append("\n## Session Highlights:")
        
        # 只添加关键的交互记录
        highlights_count = min(5, len(self.session_logs))
        for i in range(highlights_count):
            log = self.session_logs[i]
            completion = log["api_response"]["completion"]
            prompt_parts.append(f"\nStep {log['step']} (Beginning):")
            prompt_parts.append(f"Agent's reasoning: {completion[:200]}...")
        
        # 添加中间部分的一些关键交互
        if len(self.session_logs) > 10:
            mid_index = len(self.session_logs) // 2
            log = self.session_logs[mid_index]
            completion = log["api_response"]["completion"]
            prompt_parts.append(f"\nStep {log['step']} (Middle):")
            prompt_parts.append(f"Agent's reasoning: {completion[:200]}...")
        
        # 添加最后部分的交互
        if len(self.session_logs) > 5:
            for i in range(max(0, len(self.session_logs) - 3), len(self.session_logs)):
                log = self.session_logs[i]
                completion = log["api_response"]["completion"]
                prompt_parts.append(f"\nStep {log['step']} (End):")
                prompt_parts.append(f"Agent's reasoning: {completion[:200]}...")
        
        # 添加分析指导
        prompt_parts.append("\n## Analysis Tasks:")
        prompt_parts.append("""
1. Provide a one-sentence experience summary in this format: "Due to [cause], [consequence] happened. Next time, I should [improvement]."

2. List the strengths demonstrated in this session. Provide as many as you can identify.

3. List the weaknesses or areas for improvement from this session. Provide as many as you can identify.

Please format your response as follows:

Experience Summary:
[Your one-sentence summary here]

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
        """解析反思结果，提取经验总结、优点和缺点"""
        # 提取经验总结
        experience_summary = ""
        summary_match = re.search(r"Experience Summary:\s*(.*?)(?:\n\n|\n[A-Za-z]+:)", reflection_text, re.DOTALL)
        if summary_match:
            experience_summary = summary_match.group(1).strip()
        
        # 提取优点
        strengths = []
        strengths_section = re.search(r"Strengths:(.*?)(?:\n\n|\nWeaknesses:)", reflection_text, re.DOTALL)
        if strengths_section:
            strengths_text = strengths_section.group(1)
            strengths = [s.strip()[2:].strip() for s in strengths_text.split("\n") if s.strip().startswith("-")]
        
        # 提取缺点
        weaknesses = []
        weaknesses_section = re.search(r"Weaknesses:(.*?)(?:\n\n|$)", reflection_text, re.DOTALL)
        if weaknesses_section:
            weaknesses_text = weaknesses_section.group(1)
            weaknesses = [w.strip()[2:].strip() for w in weaknesses_text.split("\n") if w.strip().startswith("-")]
        
        return experience_summary, strengths, weaknesses
    
    def record_subjective_memory(self, experience_summary: str, strengths: List[str], weaknesses: List[str]):
        """
        记录主观记忆
        
        Args:
            experience_summary: 经验总结
            strengths: 优点列表
            weaknesses: 缺点列表
        """
        self.memory_manager.add_subjective_memory(
            self.current_level_id,
            experience_summary,
            strengths,
            weaknesses,
            self.session_metrics
        )
    
    def validate_subjective_memory(self, prev_metrics: Dict[str, Any], new_metrics: Dict[str, Any]) -> bool:
        """
        验证主观记忆的有效性
        
        Args:
            prev_metrics: 之前的游戏指标
            new_metrics: 新的游戏指标
            
        Returns:
            是否有效
        """
        # 检查关键指标是否有改善
        score_improved = new_metrics["score"] > prev_metrics["score"]
        success_improved = new_metrics["success"] and not prev_metrics["success"]
        steps_improved = new_metrics["steps"] < prev_metrics["steps"] and new_metrics["success"]
        exploration_improved = new_metrics["exploration_rate"] > prev_metrics["exploration_rate"]
        
        # 计算总体改进分数 (但现在仅用于记录)
        improvement_score = (
            (1 if score_improved else 0) * 2 +  # 分数改进权重为2
            (1 if success_improved else 0) * 3 +  # 成功率改进权重为3
            (1 if steps_improved else 0) * 1 +  # 步数改进权重为1
            (1 if exploration_improved else 0) * 1  # 探索率改进权重为1
        )
        
        # 严格的验证条件：
        # 1. 得分必须有提高 (必须条件)
        # 2. 必须成功通关 (必须条件)
        is_valid = score_improved and new_metrics["success"]
        
        print(f"Memory Validation Result:")
        print(f"- Success: {new_metrics['success']}")
        print(f"- Score improvement: {score_improved} ({prev_metrics['score']} -> {new_metrics['score']})")
        print(f"- Success improvement: {success_improved} ({prev_metrics['success']} -> {new_metrics['success']})")
        print(f"- Steps improvement: {steps_improved} ({prev_metrics['steps']} -> {new_metrics['steps']})")
        print(f"- Exploration improvement: {exploration_improved} ({prev_metrics['exploration_rate']:.2f} -> {new_metrics['exploration_rate']:.2f})")
        print(f"- Overall improvement score: {improvement_score}/7 (for reference only)")
        print(f"- Memory is valid: {is_valid} (requires both score improvement AND successful completion)")
        
        # 记录验证结果到文件中
        self._log_validation_result(
            prev_metrics,
            new_metrics,
            {
                "score_improved": score_improved,
                "success_improved": success_improved,
                "steps_improved": steps_improved,
                "exploration_improved": exploration_improved,
                "improvement_score": improvement_score,
                "is_valid": is_valid
            }
        )
        
        return is_valid
        
    def _log_validation_result(self, prev_metrics: Dict[str, Any], new_metrics: Dict[str, Any], 
                              validation_result: Dict[str, Any]):
        """
        记录验证结果到文件
        
        Args:
            prev_metrics: 之前的游戏指标
            new_metrics: 新的游戏指标
            validation_result: 验证结果
        """
        # 创建验证日志目录
        validation_dir = str(MEMORY_VALIDATION_DIR)
        os.makedirs(validation_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 构建记录数据
        log_data = {
            "level_id": self.current_level_id,
            "timestamp": timestamp,
            "previous_metrics": prev_metrics,
            "new_metrics": new_metrics,
            "validation_result": validation_result,
            "subjective_memory": self.memory_manager.get_subjective_memory(self.current_level_id)
        }
        
        # 转换为可序列化格式
        serializable_log = self._convert_to_serializable(log_data)
        
        # 保存验证日志
        log_file = os.path.join(validation_dir, f"validation_{self.current_level_id}_{timestamp}.json")
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_log, f, indent=2, ensure_ascii=False)
            print(f"Validation result saved to {log_file}")
        except Exception as e:
            print(f"Error saving validation result: {str(e)}")
    
    def promote_memory_to_truth(self):
        """将当前关卡的主观记忆提升为真理知识"""
        # 获取当前关卡的主观记忆
        memory = self.memory_manager.get_subjective_memory(self.current_level_id)
        
        if not memory:
            print(f"没有找到关卡 {self.current_level_id} 的主观记忆，无法提升为真理")
            return
        
        # 提取要提升的知识条目
        memory_items = []
        memory_sources = []
        
        # 添加经验总结
        if "experience_summary" in memory and memory["experience_summary"]:
            memory_items.append(memory["experience_summary"])
            memory_sources.append("experience_summary")
        
        # 添加所有优点和缺点（不限制数量）
        if "strengths" in memory and memory["strengths"]:
            for i, strength in enumerate(memory["strengths"]):
                memory_items.append(strength)
                memory_sources.append(f"strength_{i+1}")
        
        if "weaknesses" in memory and memory["weaknesses"]:
            for i, weakness in enumerate(memory["weaknesses"]):
                memory_items.append(weakness)
                memory_sources.append(f"weakness_{i+1}")
        
        # 提升为真理知识，并记录提升来源
        promoted_items = self.memory_manager.promote_to_truth(memory_items, self.current_level_id, memory_sources)
        
        # 记录提升结果
        self._log_promotion_result(memory, memory_items, memory_sources, promoted_items)
        
        print(f"已将关卡 {self.current_level_id} 的主观记忆提升为真理知识")
        
        # 提升完成后，进行真理库筛选和融合
        self.optimize_truth_knowledge()
    
    def optimize_truth_knowledge(self):
        """
        筛选和融合真理知识库中的知识，去除重复或相似的真理
        """
        # 获取当前所有真理知识
        all_truths = self.memory_manager.get_truth_knowledge()
        
        if not all_truths or len(all_truths) <= 1:
            print("真理知识库中没有足够的条目进行筛选和融合")
            return
        
        print(f"开始筛选和融合真理知识库，当前条目数量: {len(all_truths)}")
        
        # 构建提示词，要求模型筛选和融合知识
        optimization_prompt = self._create_truth_optimization_prompt(all_truths)
        
        # 发送给反思代理进行处理
        self.reflection_agent.reset()
        
        optimization_messages = [
            {"role": "system", "content": "You are a knowledge management AI assistant, specialized in optimizing knowledge items in the knowledge base, removing duplicates and merging similar knowledge items into more concise and comprehensive expressions."},
            {"role": "user", "content": optimization_prompt}
        ]
        
        optimization_response = self.reflection_agent.gpt_client.get_response(optimization_messages)
        
        # 解析优化结果
        optimized_truths = self._parse_optimized_truth(optimization_response)
        
        if not optimized_truths:
            print("无法解析优化后的真理知识，保持原有知识库不变")
            return
        
        # 更新真理知识库
        self.memory_manager.update_truth_knowledge(optimized_truths)
        
        print(f"真理知识库优化完成，优化后条目数量: {len(optimized_truths)}")
        
        # 记录优化结果
        self._log_truth_optimization(all_truths, optimized_truths)
    
    def _create_truth_optimization_prompt(self, truths: List[Dict[str, Any]]) -> str:
        """
        创建真理知识优化提示
        
        Args:
            truths: 当前所有真理知识
            
        Returns:
            优化提示
        """
        prompt_parts = [
            "# Truth Knowledge Base Optimization Task",
            "",
            "Please review the following game truth knowledge items, identify and remove duplicates, and merge similar knowledge items into more concise and comprehensive expressions.",
            "",
            "## Current Knowledge Items:"
        ]
        
        # 添加当前所有真理知识，确保使用正确的键名
        for i, truth in enumerate(truths, 1):
            # 处理可能的不同键名（'content'或原始文本直接存储）
            content = truth.get('content', truth.get('text', str(truth)))
            source = f"(Source: {truth.get('source_level', 'unknown')})" if 'source_level' in truth else ""
            prompt_parts.append(f"{i}. {content} {source}")
        
        prompt_parts.append("")
        prompt_parts.append("## Optimization Requirements:")
        prompt_parts.append("1. Identify and remove completely duplicate knowledge items")
        prompt_parts.append("2. Identify knowledge items with similar meanings but different expressions, and merge them into a single more comprehensive item")
        prompt_parts.append("3. Maintain the original meaning of the knowledge items, do not add new information")
        prompt_parts.append("4. The result should be more concise and clear, but without losing important information")
        prompt_parts.append("5. If there are no obvious duplicates or similarities between knowledge items, they can remain unchanged")
        prompt_parts.append("")
        prompt_parts.append("## Please return the optimized knowledge base in the following format:")
        prompt_parts.append("```")
        prompt_parts.append("1. [Optimized knowledge item 1]")
        prompt_parts.append("2. [Optimized knowledge item 2]")
        prompt_parts.append("...")
        prompt_parts.append("```")
        
        return "\n".join(prompt_parts)
    
    def _parse_optimized_truth(self, optimization_response: str) -> List[Dict[str, Any]]:
        """
        解析优化后的真理知识
        
        Args:
            optimization_response: 优化响应文本
            
        Returns:
            优化后的真理知识列表
        """
        # 提取优化后的知识列表
        pattern = r"```(.*?)```"
        match = re.search(pattern, optimization_response, re.DOTALL)
        
        if not match:
            # 尝试直接提取格式化的项目列表
            lines = optimization_response.strip().split("\n")
            content_lines = []
            for line in lines:
                # 检查行是否以数字和句点开头，表示列表项
                if re.match(r"^\d+\.\s", line):
                    content = re.sub(r"^\d+\.\s", "", line).strip()
                    if content:
                        content_lines.append(content)
            
            if not content_lines:
                print("无法解析优化后的真理知识")
                return []
        else:
            # 处理从代码块中提取的内容
            content = match.group(1).strip()
            content_lines = [
                re.sub(r"^\d+\.\s", "", line).strip()
                for line in content.split("\n")
                if line.strip() and re.match(r"^\d+\.\s", line)
            ]
        
        # 创建优化后的真理知识列表
        optimized_truths = []
        for content in content_lines:
            if content:
                optimized_truths.append({
                    "content": content,
                    "source_level": "多关卡融合",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        return optimized_truths
    
    def _log_truth_optimization(self, original_truths: List[Dict[str, Any]], 
                               optimized_truths: List[Dict[str, Any]]):
        """
        记录真理知识优化结果
        
        Args:
            original_truths: 原始真理知识
            optimized_truths: 优化后的真理知识
        """
        # 创建优化日志目录
        optimization_dir = str(TRUTH_OPTIMIZATION_DIR)
        os.makedirs(optimization_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 构建记录数据
        log_data = {
            "timestamp": timestamp,
            "original_count": len(original_truths),
            "optimized_count": len(optimized_truths),
            "original_truths": original_truths,
            "optimized_truths": optimized_truths
        }
        
        # 转换为可序列化格式
        serializable_log = self._convert_to_serializable(log_data)
        
        # 保存优化日志
        log_file = os.path.join(optimization_dir, f"truth_optimization_{timestamp}.json")
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_log, f, indent=2, ensure_ascii=False)
            print(f"Truth knowledge optimization result saved to {log_file}")
        except Exception as e:
            print(f"Error saving truth knowledge optimization result: {str(e)}")
    
    def _log_promotion_result(self, memory: Dict[str, Any], items: List[str], 
                             sources: List[str], promoted: List[bool]):
        """
        记录提升结果到文件
        
        Args:
            memory: 主观记忆
            items: 提升的知识条目
            sources: 知识来源标识
            promoted: 提升结果
        """
        # 创建提升日志目录
        promotion_dir = str(MEMORY_PROMOTION_DIR)
        os.makedirs(promotion_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 构建记录数据
        log_data = {
            "level_id": self.current_level_id,
            "timestamp": timestamp,
            "memory": memory,
            "promotion_details": [
                {
                    "item": item,
                    "source": source,
                    "promoted": promoted
                }
                for item, source, promoted in zip(items, sources, promoted)
            ]
        }
        
        # 转换为可序列化格式
        serializable_log = self._convert_to_serializable(log_data)
        
        # 保存提升日志
        log_file = os.path.join(promotion_dir, f"promotion_{self.current_level_id}_{timestamp}.json")
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_log, f, indent=2, ensure_ascii=False)
            print(f"Promotion result saved to {log_file}")
        except Exception as e:
            print(f"Error saving promotion result: {str(e)}")
    
    def clear_current_subjective_memory(self):
        """清除当前关卡的主观记忆"""
        self.memory_manager.clear_subjective_memory(self.current_level_id) 
