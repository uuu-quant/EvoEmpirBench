import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional

from src.config.paths import AGENT_MEMORY_DIR

class MemoryManager:
    """
    记忆管理器类，负责管理代理的主观记忆和真理知识
    """
    
    def __init__(self, memory_dir: str = None):
        """
        初始化记忆管理器
        
        Args:
            memory_dir: 记忆文件保存目录
        """
        # 设置记忆保存目录
        if memory_dir is None:
            self.memory_dir = str(AGENT_MEMORY_DIR)
        else:
            self.memory_dir = memory_dir
            
        # 创建记忆保存目录
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # 主观记忆模块 (每个关卡一个)
        self.subjective_memories = {}
        
        # 真理模块 (全局共享)
        self.truth_knowledge = []
        
        # 加载已有的记忆和真理
        self._load_memory()
    
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
    
    def _load_memory(self):
        """加载已有的记忆和真理知识"""
        # 加载真理知识
        truth_file = os.path.join(self.memory_dir, 'truth_knowledge.json')
        if os.path.exists(truth_file):
            try:
                with open(truth_file, 'r', encoding='utf-8') as f:
                    self.truth_knowledge = json.load(f)
                print(f"已加载 {len(self.truth_knowledge)} 条真理知识")
            except Exception as e:
                print(f"加载真理知识失败: {str(e)}")
                self.truth_knowledge = []
        
        # 加载主观记忆 (遍历记忆目录下的所有关卡记忆文件)
        memory_pattern = os.path.join(self.memory_dir, 'subjective_memory_*.json')
        import glob
        memory_files = glob.glob(memory_pattern)
        
        for memory_file in memory_files:
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    level_id = memory_data.get('level_id')
                    if level_id:
                        self.subjective_memories[level_id] = memory_data
                        print(f"已加载关卡 {level_id} 的主观记忆")
            except Exception as e:
                print(f"加载记忆文件 {memory_file} 失败: {str(e)}")
    
    def save_memory(self):
        """保存记忆和真理知识到文件"""
        # 转换为可序列化格式
        serializable_truth = self._convert_to_serializable(self.truth_knowledge)
        
        # 保存真理知识
        truth_file = os.path.join(self.memory_dir, 'truth_knowledge.json')
        try:
            with open(truth_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_truth, f, indent=2, ensure_ascii=False)
            print(f"已保存 {len(self.truth_knowledge)} 条真理知识")
        except Exception as e:
            print(f"保存真理知识失败: {str(e)}")
        
        # 保存主观记忆
        for level_id, memory_data in self.subjective_memories.items():
            serializable_memory = self._convert_to_serializable(memory_data)
            memory_file = os.path.join(self.memory_dir, f'subjective_memory_{level_id}.json')
            try:
                with open(memory_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_memory, f, indent=2, ensure_ascii=False)
                print(f"已保存关卡 {level_id} 的主观记忆")
            except Exception as e:
                print(f"保存主观记忆 {level_id} 失败: {str(e)}")
    
    def get_subjective_memory(self, level_id: str) -> Dict[str, Any]:
        """
        获取指定关卡的主观记忆
        
        Args:
            level_id: 关卡ID
            
        Returns:
            关卡的主观记忆，如果不存在则返回空字典
        """
        return self.subjective_memories.get(level_id, {})
    
    def get_truth_knowledge(self) -> List[str]:
        """
        获取真理知识列表
        
        Returns:
            真理知识列表
        """
        return self.truth_knowledge
    
    def add_subjective_memory(self, level_id: str, experience_summary: str, 
                              strengths: List[str], weaknesses: List[str],
                              game_metrics: Dict[str, Any]):
        """
        添加主观记忆
        
        Args:
            level_id: 关卡ID
            experience_summary: 经验总结
            strengths: 优点列表
            weaknesses: 缺点列表
            game_metrics: 游戏指标
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
        
        print(f"已添加关卡 {level_id} 的主观记忆")
    
    def promote_to_truth(self, memory_items: List[str], level_id: Optional[str] = None, 
                       sources: Optional[List[str]] = None) -> List[bool]:
        """
        将主观记忆提升为真理知识
        
        Args:
            memory_items: 要提升为真理的记忆条目
            level_id: 关卡ID，如果提供，将在日志中注明来源
            sources: 知识来源标识列表，如果提供，将记录在日志中
            
        Returns:
            每个记忆项是否成功提升的布尔值列表
        """
        promotion_results = []
        source_tags = sources if sources else [None] * len(memory_items)
        
        for i, (item, source_tag) in enumerate(zip(memory_items, source_tags)):
            # 构建真理条目，包括来源和时间戳
            source_prefix = f"{source_tag} from " if source_tag else ""
            source_detail = f"{source_prefix}Level {level_id}" if level_id else "Unknown"
            
            truth_entry = {
                'knowledge': item,
                'source': source_detail,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 检查是否已存在相同内容的真理
            if not any(entry['knowledge'] == item for entry in self.truth_knowledge):
                self.truth_knowledge.append(truth_entry)
                print(f"已将 '{item}' 提升为真理知识")
                promotion_results.append(True)
            else:
                print(f"真理知识 '{item}' 已存在，跳过添加")
                promotion_results.append(False)
        
        self.save_memory()
        return promotion_results
    
    def get_memory_prompt(self, level_id: str, include_truth: bool = True, 
                          include_subjective: bool = True) -> str:
        """
        生成包含记忆的提示词
        
        Args:
            level_id: 关卡ID
            include_truth: 是否包含真理知识
            include_subjective: 是否包含主观记忆
            
        Returns:
            包含记忆的提示词
        """
        prompt_parts = []
        
        # 添加真理知识
        if include_truth and self.truth_knowledge:
            prompt_parts.append("# Truth Knowledge (Validated Learning)")
            for entry in self.truth_knowledge:
                prompt_parts.append(f"- {entry['knowledge']}")
            prompt_parts.append("")
        
        # 添加主观记忆
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
            
            if memory.get('experience_summary'):
                prompt_parts.append(f"## Experience Summary:")
                prompt_parts.append(memory['experience_summary'])
                prompt_parts.append("")
        
        return "\n".join(prompt_parts) if prompt_parts else ""
    
    def clear_subjective_memory(self, level_id: str):
        """
        清除指定关卡的主观记忆
        
        Args:
            level_id: 关卡ID
        """
        if level_id in self.subjective_memories:
            del self.subjective_memories[level_id]
            print(f"已清除关卡 {level_id} 的主观记忆")
            
            # 删除对应的文件
            memory_file = os.path.join(self.memory_dir, f'subjective_memory_{level_id}.json')
            if os.path.exists(memory_file):
                try:
                    os.remove(memory_file)
                except Exception as e:
                    print(f"删除记忆文件失败: {str(e)}")
        
        self.save_memory()
    
    def clear_all_memories(self):
        """清除所有记忆和真理知识"""
        self.subjective_memories = {}
        self.truth_knowledge = []
        
        print("已清除所有记忆和真理知识")
        self.save_memory() 
