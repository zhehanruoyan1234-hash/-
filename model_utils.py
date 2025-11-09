"""
模型工具模块
提供统一的模型缓存配置、环境变量设置等功能
"""
import os
from pathlib import Path
from typing import Optional, Tuple

# 全局缓存目录（单例）
_cache_dir: Optional[str] = None
_cache_dir_lock = False

def get_model_cache_dir() -> str:
    """获取模型缓存目录（单例模式）
    
    Returns:
        模型缓存目录路径
    """
    global _cache_dir
    
    if _cache_dir is None:
        # 使用项目根目录下的 models_cache
        base_dir = Path(__file__).parent
        _cache_dir = str(base_dir / 'models_cache')
        os.makedirs(_cache_dir, exist_ok=True)
    
    return _cache_dir

def setup_transformers_cache(cache_dir: Optional[str] = None) -> str:
    """设置 Transformers 缓存目录和环境变量
    
    Args:
        cache_dir: 缓存目录路径，如果为None则使用默认目录
        
    Returns:
        实际使用的缓存目录路径
    """
    if cache_dir is None:
        cache_dir = get_model_cache_dir()
    
    # 设置环境变量
    os.environ.setdefault('TRANSFORMERS_CACHE', cache_dir)
    os.environ.setdefault('HF_HOME', cache_dir)
    
    return cache_dir

def get_device() -> str:
    """获取计算设备（CPU或CUDA）
    
    Returns:
        'cuda' 或 'cpu'
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_device_id() -> int:
    """获取设备ID（用于pipeline）
    
    Returns:
        0 (CUDA) 或 -1 (CPU)
    """
    import torch
    return 0 if torch.cuda.is_available() else -1

