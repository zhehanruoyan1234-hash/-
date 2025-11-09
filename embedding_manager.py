"""
多维度向量嵌入支持
为不同类型的记忆使用不同的嵌入模型和维度
"""
from typing import Dict, Optional, List
from sentence_transformers import SentenceTransformer
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)

# 从环境变量读取配置（兼容config.py）
def get_setting(key: str, default: str) -> str:
    """获取配置值"""
    return os.getenv(key, default)

class MemoryType(Enum):
    """记忆类型"""
    TEXT = "text"           # 文本记忆（384维）
    EMOTION = "emotion"     # 情绪记忆（128维）
    SCHEDULE = "schedule"   # 行程记忆（384维）
    PREFERENCE = "preference"  # 偏好记忆（384维）

class EmbeddingConfig:
    """嵌入配置"""
    def __init__(self, model_name: str, dimension: int, description: str = ""):
        self.model_name = model_name
        self.dimension = dimension
        self.description = description

# 默认嵌入配置（从环境变量读取，避免硬编码）
EMBEDDING_CONFIGS = {
    MemoryType.TEXT: EmbeddingConfig(
        model_name=get_setting("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        dimension=int(get_setting("TEXT_EMBEDDING_DIM", "384")),
        description="文本记忆嵌入模型"
    ),
    MemoryType.EMOTION: EmbeddingConfig(
        model_name=get_setting("EMOTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        dimension=int(get_setting("EMOTION_EMBEDDING_DIM", "128")),
        description="情绪记忆嵌入模型"
    ),
    MemoryType.SCHEDULE: EmbeddingConfig(
        model_name=get_setting("SCHEDULE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        dimension=int(get_setting("SCHEDULE_EMBEDDING_DIM", "384")),
        description="行程记忆嵌入模型"
    ),
    MemoryType.PREFERENCE: EmbeddingConfig(
        model_name=get_setting("PREFERENCE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        dimension=int(get_setting("PREFERENCE_EMBEDDING_DIM", "384")),
        description="偏好记忆嵌入模型"
    ),
}

# 全局模型缓存
_embedding_models: Dict[MemoryType, Optional[SentenceTransformer]] = {}
_embedding_dimensions: Dict[MemoryType, int] = {}

def get_memory_type(memory_info: Dict) -> MemoryType:
    """根据记忆信息确定记忆类型"""
    memory_type_str = memory_info.get("memory_type", "event")
    is_schedule = memory_info.get("is_schedule", False)
    
    if is_schedule or memory_type_str == "schedule":
        return MemoryType.SCHEDULE
    elif memory_type_str == "preference":
        return MemoryType.PREFERENCE
    elif memory_type_str == "emotion":
        return MemoryType.EMOTION
    else:
        return MemoryType.TEXT

def get_embedding_model_for_type(memory_type: MemoryType) -> Optional[SentenceTransformer]:
    """获取指定类型的嵌入模型（延迟加载）
    优化：使用本地缓存目录，避免每次重启都重新下载
    """
    global _embedding_models
    
    if memory_type not in _embedding_models:
        config = EMBEDDING_CONFIGS.get(memory_type)
        if not config:
            logger.warning(f"未找到记忆类型 {memory_type} 的配置，使用默认配置")
            config = EMBEDDING_CONFIGS[MemoryType.TEXT]
        
        try:
            # 优化1：使用公共工具模块设置缓存
            from model_utils import get_model_cache_dir
            cache_dir = get_model_cache_dir()
            
            model = SentenceTransformer(
                config.model_name,
                cache_folder=cache_dir  # 指定缓存目录
            )
            _embedding_models[memory_type] = model
            _embedding_dimensions[memory_type] = config.dimension
            logger.info(f"[OK] {config.description}加载成功: {config.model_name} ({config.dimension}维, 缓存目录: {cache_dir})")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            _embedding_models[memory_type] = None
    
    return _embedding_models.get(memory_type)

def get_embedding_dimension(memory_type: MemoryType) -> int:
    """获取指定类型的向量维度"""
    if memory_type in _embedding_dimensions:
        return _embedding_dimensions[memory_type]
    
    config = EMBEDDING_CONFIGS.get(memory_type)
    if config:
        return config.dimension
    
    return 384  # 默认维度

def encode_text(text: str, memory_type: MemoryType) -> Optional[List[float]]:
    """编码文本为向量"""
    model = get_embedding_model_for_type(memory_type)
    if not model:
        return None
    
    try:
        return model.encode(text).tolist()
    except Exception as e:
        logger.error(f"编码文本失败: {e}")
        return None

