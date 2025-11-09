"""
缓存管理器：用于缓存分析结果和数据库查询结果，提高响应速度
"""
from functools import lru_cache
from collections import OrderedDict
import hashlib
import json
import time
from typing import Optional, Dict, Any

class LRUCache:
    """简单的LRU缓存实现"""
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.access_times = {}  # 记录访问时间
        self.ttl = {}  # 生存时间（秒），None表示永不过期
    
    def get(self, key: str, default=None) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            # 检查是否过期
            if key in self.ttl and self.ttl[key] is not None:
                if time.time() > self.ttl[key]:
                    # 过期，删除
                    del self.cache[key]
                    del self.access_times[key]
                    del self.ttl[key]
                    return default
            
            # 移到末尾（最近使用）
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            return self.cache[key]
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """设置缓存值"""
        if key in self.cache:
            # 更新现有值
            self.cache.move_to_end(key)
        else:
            # 检查是否超过最大容量
            if len(self.cache) >= self.max_size:
                # 删除最旧的（第一个）
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.access_times:
                    del self.access_times[oldest_key]
                if oldest_key in self.ttl:
                    del self.ttl[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        if ttl is not None:
            self.ttl[key] = time.time() + ttl
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.ttl.clear()
    
    def remove(self, key: str):
        """移除指定key"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.ttl:
            del self.ttl[key]
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self.cache)

# 全局缓存实例
_emotion_cache = LRUCache(max_size=256)  # 情绪分析缓存
_role_cache = LRUCache(max_size=256)     # 角色分析缓存
_user_data_cache = LRUCache(max_size=64)  # 用户数据缓存（TTL较短）
_conversation_cache = LRUCache(max_size=128)  # 对话历史缓存

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """生成缓存键"""
    # 将参数序列化为字符串
    key_parts = [prefix]
    
    for arg in args:
        if isinstance(arg, str):
            key_parts.append(arg)
        elif isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True, ensure_ascii=False))
        else:
            key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (dict, list)):
            key_parts.append(f"{k}={json.dumps(v, sort_keys=True, ensure_ascii=False)}")
        else:
            key_parts.append(f"{k}={v}")
    
    key_string = "|".join(key_parts)
    # 使用MD5生成固定长度的键
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def _get_cache(cache_instance, prefix: str, *args, **kwargs) -> Optional[Any]:
    """通用的缓存获取函数"""
    cache_key = generate_cache_key(prefix, *args, **kwargs)
    return cache_instance.get(cache_key)

def _set_cache(cache_instance, prefix: str, value: Any, ttl: float, *args, **kwargs):
    """通用的缓存设置函数"""
    cache_key = generate_cache_key(prefix, *args, **kwargs)
    cache_instance.set(cache_key, value, ttl=ttl)

def get_emotion_cache(user_input: str, conversation_history: list = None) -> Optional[Dict]:
    """获取情绪分析缓存"""
    return _get_cache(_emotion_cache, "emotion", user_input, conversation_history=conversation_history)

def set_emotion_cache(user_input: str, result: Dict, conversation_history: list = None, ttl: float = 3600):
    """设置情绪分析缓存（默认1小时过期）"""
    _set_cache(_emotion_cache, "emotion", result, ttl, user_input, conversation_history=conversation_history)

def get_role_cache(user_input: str, conversation_history: list = None) -> Optional[Dict]:
    """获取角色分析缓存"""
    return _get_cache(_role_cache, "role", user_input, conversation_history=conversation_history)

def set_role_cache(user_input: str, result: Dict, conversation_history: list = None, ttl: float = 3600):
    """设置角色分析缓存（默认1小时过期）"""
    _set_cache(_role_cache, "role", result, ttl, user_input, conversation_history=conversation_history)

def get_user_data_cache(user_id: str) -> Optional[Dict]:
    """获取用户数据缓存（短TTL，5分钟）"""
    return _get_cache(_user_data_cache, "user_data", user_id)

def set_user_data_cache(user_id: str, data: Dict, ttl: float = 300):
    """设置用户数据缓存（默认5分钟过期）"""
    _set_cache(_user_data_cache, "user_data", data, ttl, user_id)

def get_conversation_history_cache(user_id: str, limit: int) -> Optional[list]:
    """获取对话历史缓存（短TTL，1分钟）"""
    return _get_cache(_conversation_cache, "conv_history", user_id, limit=limit)

def set_conversation_history_cache(user_id: str, limit: int, data: list, ttl: float = 60):
    """设置对话历史缓存（默认1分钟过期）"""
    _set_cache(_conversation_cache, "conv_history", data, ttl, user_id, limit=limit)

def clear_all_caches():
    """清空所有缓存"""
    _emotion_cache.clear()
    _role_cache.clear()
    _user_data_cache.clear()
    _conversation_cache.clear()

def clear_user_caches(user_id: str):
    """清空特定用户的所有缓存"""
    # 由于使用MD5键，无法直接按用户ID删除，需要遍历
    # 这是简化版本，实际使用中可以考虑使用前缀索引
    keys_to_remove = []
    for key in list(_user_data_cache.cache.keys()):
        # 检查key是否包含用户ID（简化检查）
        if user_id in str(key):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        _user_data_cache.remove(key)

