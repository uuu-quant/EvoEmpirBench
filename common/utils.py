"""
Common utility functions shared between both game environments.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

def ensure_dir(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Path to check/create
        
    Returns:
        The directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_timestamp() -> str:
    """
    Get current timestamp string in format suitable for filenames.
    
    Returns:
        Formatted timestamp string
    """
    return time.strftime("%Y%m%d_%H%M%S")

def save_json(data: Any, filepath: str, pretty: bool = True) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Target file path
        pretty: Whether to format with indentation
        
    Returns:
        Success status
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        return False

def load_json(filepath: str) -> Optional[Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Source file path
        
    Returns:
        Loaded data or None on error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return None

def to_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable version of the object
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
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return {k: to_serializable(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    else:
        return obj
        
def create_results_dir(base_dir: str, model_name: str, game_id: str, mode: str) -> str:
    """
    Create standardized results directory structure.
    
    Args:
        base_dir: Base directory
        model_name: Name of model being evaluated
        game_id: Game identifier (game1 or game2)
        mode: Mode/difficulty being evaluated
        
    Returns:
        Path to created results directory
    """
    path = os.path.join(base_dir, "results", model_name, game_id, mode)
    os.makedirs(path, exist_ok=True)
    return path 