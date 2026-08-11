import os
import json
import time
from typing import List, Dict, Any

from src.config.paths import MEMORIES_DIR

class MemoryManager:
    def __init__(self):
        # 初始化记忆存储
        self.subjective_memories = {}  # 按关卡ID存储主观记忆
        self.truth_knowledge = []  # 所有关卡共享的真理知识
        
        # 创建记忆存储目录
        self.memory_dir = self._create_memory_directory()
        
        # 加载已有的记忆
        self._load_memories()
    
    def _create_memory_directory(self) -> str:
        """创建记忆存储目录"""
        memory_dir = str(MEMORIES_DIR)
        os.makedirs(memory_dir, exist_ok=True)
        return memory_dir
    
    def _load_memories(self):
        """加载已有的记忆"""
        # 加载主观记忆
        subjective_path = os.path.join(self.memory_dir, 'subjective_memories.json')
        if os.path.exists(subjective_path):
            try:
                with open(subjective_path, 'r', encoding='utf-8') as f:
                    self.subjective_memories = json.load(f)
                print(f"已加载主观记忆，关卡数量: {len(self.subjective_memories)}")
            except Exception as e:
                print(f"加载主观记忆失败: {str(e)}")
                self.subjective_memories = {}
        
        # 加载真理知识
        truth_path = os.path.join(self.memory_dir, 'truth_knowledge.json')
        if os.path.exists(truth_path):
            try:
                with open(truth_path, 'r', encoding='utf-8') as f:
                    self.truth_knowledge = json.load(f)
                print(f"已加载真理知识，条目数量: {len(self.truth_knowledge)}")
            except Exception as e:
                print(f"加载真理知识失败: {str(e)}")
                self.truth_knowledge = []
    
    def _save_memories(self):
        """保存记忆到文件"""
        # 保存主观记忆
        subjective_path = os.path.join(self.memory_dir, 'subjective_memories.json')
        try:
            with open(subjective_path, 'w', encoding='utf-8') as f:
                json.dump(self.subjective_memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存主观记忆失败: {str(e)}")
        
        # 保存真理知识
        truth_path = os.path.join(self.memory_dir, 'truth_knowledge.json')
        try:
            with open(truth_path, 'w', encoding='utf-8') as f:
                json.dump(self.truth_knowledge, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存真理知识失败: {str(e)}")
    
    def add_subjective_memory(self, level_id: str, experience_summary: str, 
                            strengths: List[str], weaknesses: List[str], 
                            metrics: Dict[str, Any] = None):
        """
        添加主观记忆
        
        Args:
            level_id: 关卡ID
            experience_summary: 经验总结
            strengths: 优点列表
            weaknesses: 缺点列表
            metrics: 游戏指标
        """
        # 创建记忆条目
        memory = {
            "experience_summary": experience_summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }
        
        # 添加到主观记忆
        self.subjective_memories[level_id] = memory
        
        # 保存到文件
        self._save_memories()
        
        print(f"已为关卡 {level_id} 添加主观记忆")
    
    def get_subjective_memory(self, level_id: str) -> Dict[str, Any]:
        """
        获取指定关卡的主观记忆
        
        Args:
            level_id: 关卡ID
            
        Returns:
            主观记忆，如果不存在则返回空字典
        """
        return self.subjective_memories.get(level_id, {})
    
    def clear_subjective_memory(self, level_id: str):
        """
        清除指定关卡的主观记忆
        
        Args:
            level_id: 关卡ID
        """
        if level_id in self.subjective_memories:
            del self.subjective_memories[level_id]
            self._save_memories()
            print(f"已清除关卡 {level_id} 的主观记忆")
        else:
            print(f"关卡 {level_id} 没有主观记忆，无需清除")
    
    def promote_to_truth(self, memory_items: List[str], source_level: str, 
                        sources: List[str] = None) -> List[bool]:
        """
        将主观记忆提升为真理知识
        
        Args:
            memory_items: 记忆条目列表
            source_level: 来源关卡ID
            sources: 每个条目的来源标识（如 'experience_summary', 'strength_1'）
            
        Returns:
            提升结果列表，表示每个条目是否被成功提升
        """
        if not memory_items:
            return []
        
        results = []
        
        for i, item in enumerate(memory_items):
            source = sources[i] if sources and i < len(sources) else f"memory_{i+1}"
            
            # 检查是否已存在相同内容的真理知识
            duplicate = False
            for truth in self.truth_knowledge:
                if truth["content"].lower() == item.lower():
                    duplicate = True
                    break
            
            if not duplicate:
                # 添加为新的真理知识
                self.truth_knowledge.append({
                    "content": item,
                    "source_level": source_level,
                    "source_type": source,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                results.append(True)
            else:
                # 跳过重复项
                print(f"已跳过重复的真理知识: {item[:50]}...")
                results.append(False)
        
        # 保存更新后的真理知识
        self._save_memories()
        
        print(f"已将 {sum(results)} 条记忆提升为真理知识")
        
        return results
    
    def get_truth_knowledge(self) -> List[Dict[str, Any]]:
        """
        获取所有真理知识
        
        Returns:
            真理知识列表
        """
        return self.truth_knowledge.copy()
    
    def update_truth_knowledge(self, optimized_truths: List[Dict[str, Any]]):
        """
        更新真理知识库
        
        Args:
            optimized_truths: 优化后的真理知识
        """
        self.truth_knowledge = optimized_truths
        self._save_memories()
        print(f"已更新真理知识库，当前条目数量: {len(self.truth_knowledge)}")
    
    def get_memory_prompt(self, level_id: str = None, 
                        include_truth: bool = True, 
                        include_subjective: bool = True) -> str:
        """
        获取记忆提示，包括真理知识和主观记忆
        
        Args:
            level_id: 关卡ID，用于获取特定关卡的主观记忆
            include_truth: 是否包含真理知识
            include_subjective: 是否包含主观记忆
            
        Returns:
            记忆提示
        """
        prompt_parts = []
        
        # 添加真理知识
        if include_truth and self.truth_knowledge:
            prompt_parts.append("# Truth Knowledge")
            prompt_parts.append("The following are verified universal truths learned from past games:")
            for i, truth in enumerate(self.truth_knowledge, 1):
                prompt_parts.append(f"{i}. {truth['content']}")
            prompt_parts.append("")
        
        # 添加特定关卡的主观记忆
        if include_subjective and level_id and level_id in self.subjective_memories:
            memory = self.subjective_memories[level_id]
            
            prompt_parts.append(f"# Subjective Memory for Level {level_id}")
            
            if memory.get("experience_summary"):
                prompt_parts.append("## Experience Summary")
                prompt_parts.append(memory["experience_summary"])
                prompt_parts.append("")
            
            if memory.get("strengths"):
                prompt_parts.append("## Strengths")
                for i, strength in enumerate(memory["strengths"], 1):
                    prompt_parts.append(f"{i}. {strength}")
                prompt_parts.append("")
            
            if memory.get("weaknesses"):
                prompt_parts.append("## Weaknesses")
                for i, weakness in enumerate(memory["weaknesses"], 1):
                    prompt_parts.append(f"{i}. {weakness}")
                prompt_parts.append("")
        
        # 如果没有记忆可用
        if not prompt_parts:
            return ""
        
        return "\n".join(prompt_parts) 
