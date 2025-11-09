"""
高级缓存系统
支持结构化键、多维度失效策略、统计监控和持久化
"""
import hashlib
import json
import time
import pickle
import threading
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict, defaultdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

# Redis已移除，使用SQLite和内存缓存
REDIS_AVAILABLE = False

# DiskCache导入（可选）
try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

class CachePriority(Enum):
    """缓存优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class CacheType(Enum):
    """缓存类型"""
    EMOTION = "emotion"
    ROLE = "role"
    USER_DATA = "user_data"
    CONVERSATION = "conversation"
    VECTOR = "vector"
    MEMORY = "memory"

class CacheKeyBuilder:
    """结构化缓存键生成器"""
    
    @staticmethod
    def build_key(cache_type: CacheType, *parts: str, **kwargs) -> str:
        """
        构建结构化缓存键
        
        Args:
            cache_type: 缓存类型
            *parts: 键的组成部分（如user_id, session_id等）
            **kwargs: 额外的键值对
        
        Returns:
            结构化键，格式：{type}:{part1}:{part2}:{hash}
        """
        key_parts = [cache_type.value]
        
        # 添加主要部分
        for part in parts:
            if part:
                # 清理特殊字符，避免冲突
                clean_part = str(part).replace(":", "_").replace("|", "_")
                key_parts.append(clean_part)
        
        # 添加额外参数（按字母顺序排序以保持一致性）
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            param_parts = [f"{k}={v}" for k, v in sorted_kwargs if v is not None]
            if param_parts:
                # 对于复杂参数，使用SHA256哈希（避免键过长）
                params_str = "|".join(param_parts)
                if len(params_str) > 100:
                    params_hash = hashlib.sha256(params_str.encode('utf-8')).hexdigest()[:16]
                    key_parts.append(f"params:{params_hash}")
                else:
                    key_parts.append(f"params:{params_str.replace(':', '_')}")
        
        return ":".join(key_parts)
    
    @staticmethod
    def build_user_key(cache_type: CacheType, user_id: str, *args, **kwargs) -> str:
        """构建用户相关的缓存键"""
        return CacheKeyBuilder.build_key(cache_type, "user", user_id, *args, **kwargs)
    
    @staticmethod
    def build_vector_key(text: str, memory_type: str = "text") -> str:
        """构建向量缓存键"""
        # 对长文本使用SHA256哈希
        if len(text) > 200:
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
            return CacheKeyBuilder.build_key(CacheType.VECTOR, memory_type, text_hash)
        else:
            clean_text = text.replace(":", "_").replace("|", "_")
            return CacheKeyBuilder.build_key(CacheType.VECTOR, memory_type, clean_text)

class CacheStatistics:
    """缓存统计信息"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.expires = 0
        self.evictions = 0
        self.sets = 0
        self.errors = 0
        self._lock = threading.Lock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_expire(self):
        with self._lock:
            self.expires += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    def record_set(self):
        with self._lock:
            self.sets += 1
    
    def record_error(self):
        with self._lock:
            self.errors += 1
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "expires": self.expires,
            "evictions": self.evictions,
            "sets": self.sets,
            "errors": self.errors,
            "hit_rate": self.get_hit_rate(),
            "total_requests": self.hits + self.misses
        }
    
    def reset(self):
        """重置统计"""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.expires = 0
            self.evictions = 0
            self.sets = 0
            self.errors = 0

class LFUCacheEntry:
    """LFU缓存条目（带访问频率）"""
    def __init__(self, key: str, value: Any, priority: CachePriority = CachePriority.MEDIUM):
        self.key = key
        self.value = value
        self.access_count = 1
        self.last_access = time.time()
        self.created_at = time.time()
        self.priority = priority
        self.ttl = None

class AdvancedCache:
    """高级缓存实现（支持LRU/LFU，多维度失效，统计，持久化）"""
    
    def __init__(self, name: str, max_size: int = 128, 
                 eviction_policy: str = "LRU",  # LRU or LFU
                 enable_persistence: bool = False,
                 persistence_path: Optional[Path] = None):
        self.name = name
        self.max_size = max_size
        self.eviction_policy = eviction_policy.upper()
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or Path(f"cache_snapshots/{name}.cache")
        
        # LRU缓存结构
        self.cache: OrderedDict = OrderedDict()
        
        # LFU缓存结构（如果使用LFU）
        self.lfu_cache: Dict[str, LFUCacheEntry] = {}
        
        # TTL管理
        self.ttl_map: Dict[str, float] = {}
        
        # 访问时间记录
        self.access_times: Dict[str, float] = {}
        
        # 优先级配置（不同优先级有不同的TTL）
        self.priority_ttl = {
            CachePriority.LOW: 300,      # 5分钟
            CachePriority.MEDIUM: 3600,   # 1小时
            CachePriority.HIGH: 86400,   # 24小时
            CachePriority.CRITICAL: None  # 永不过期
        }
        
        # 统计信息
        self.stats = CacheStatistics()
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 加载持久化缓存（如果启用）
        if self.enable_persistence:
            self._load_from_disk()
    
    def get(self, key: str, default=None) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            # 检查LFU缓存
            if self.eviction_policy == "LFU":
                if key in self.lfu_cache:
                    entry = self.lfu_cache[key]
                    
                    # 检查TTL
                    if entry.ttl and time.time() > entry.ttl:
                        del self.lfu_cache[key]
                        if key in self.ttl_map:
                            del self.ttl_map[key]
                        self.stats.record_expire()
                        self.stats.record_miss()
                        return default
                    
                    # 更新访问频率
                    entry.access_count += 1
                    entry.last_access = time.time()
                    self.stats.record_hit()
                    return entry.value
                else:
                    self.stats.record_miss()
                    return default
            
            # LRU缓存
            if key in self.cache:
                # 检查TTL
                if key in self.ttl_map:
                    if time.time() > self.ttl_map[key]:
                        del self.cache[key]
                        del self.ttl_map[key]
                        if key in self.access_times:
                            del self.access_times[key]
                        self.stats.record_expire()
                        self.stats.record_miss()
                        return default
                
                # 移到末尾（最近使用）
                self.cache.move_to_end(key)
                self.access_times[key] = time.time()
                self.stats.record_hit()
                return self.cache[key]
            
            self.stats.record_miss()
            return default
    
    def set(self, key: str, value: Any, 
            ttl: Optional[float] = None,
            priority: CachePriority = CachePriority.MEDIUM):
        """设置缓存值"""
        with self._lock:
            # 如果指定了优先级，使用优先级的默认TTL
            if ttl is None:
                ttl = self.priority_ttl.get(priority)
            
            # LFU缓存
            if self.eviction_policy == "LFU":
                if key in self.lfu_cache:
                    entry = self.lfu_cache[key]
                    entry.value = value
                    entry.access_count += 1
                    entry.last_access = time.time()
                    entry.priority = priority
                    if ttl:
                        entry.ttl = time.time() + ttl
                else:
                    # 检查容量
                    if len(self.lfu_cache) >= self.max_size:
                        self._evict_lfu()
                    
                    entry = LFUCacheEntry(key, value, priority)
                    if ttl:
                        entry.ttl = time.time() + ttl
                    self.lfu_cache[key] = entry
                    if ttl:
                        self.ttl_map[key] = time.time() + ttl
                
                self.stats.record_set()
                return
            
            # LRU缓存
            if key in self.cache:
                # 更新现有值
                self.cache.move_to_end(key)
            else:
                # 检查容量
                if len(self.cache) >= self.max_size:
                    self._evict_lru()
                
                # 添加到末尾
                self.cache[key] = value
            
            self.access_times[key] = time.time()
            if ttl:
                self.ttl_map[key] = time.time() + ttl
            else:
                # 如果设置了priority但没有TTL，使用优先级默认TTL
                if priority in self.priority_ttl:
                    priority_ttl = self.priority_ttl[priority]
                    if priority_ttl:
                        self.ttl_map[key] = time.time() + priority_ttl
            
            self.stats.record_set()
    
    def _evict_lru(self):
        """LRU淘汰：删除最久未使用的项"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.ttl_map:
                del self.ttl_map[oldest_key]
            if oldest_key in self.access_times:
                del self.access_times[oldest_key]
            self.stats.record_eviction()
    
    def _evict_lfu(self):
        """LFU淘汰：删除访问频率最低的项"""
        if not self.lfu_cache:
            return
        
        # 找到访问频率最低的项
        min_entry = min(self.lfu_cache.values(), key=lambda e: e.access_count)
        del self.lfu_cache[min_entry.key]
        if min_entry.key in self.ttl_map:
            del self.ttl_map[min_entry.key]
        self.stats.record_eviction()
    
    def invalidate(self, key: str):
        """主动失效：删除指定键"""
        with self._lock:
            if self.eviction_policy == "LFU":
                if key in self.lfu_cache:
                    del self.lfu_cache[key]
            else:
                if key in self.cache:
                    del self.cache[key]
            
            if key in self.ttl_map:
                del self.ttl_map[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def invalidate_pattern(self, pattern: str):
        """按模式失效：删除匹配模式的键"""
        with self._lock:
            keys_to_remove = []
            
            if self.eviction_policy == "LFU":
                for key in self.lfu_cache.keys():
                    if pattern in key:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del self.lfu_cache[key]
            else:
                for key in self.cache.keys():
                    if pattern in key:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del self.cache[key]
            
            for key in keys_to_remove:
                if key in self.ttl_map:
                    del self.ttl_map[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.lfu_cache.clear()
            self.ttl_map.clear()
            self.access_times.clear()
    
    def size(self) -> int:
        """返回缓存大小"""
        if self.eviction_policy == "LFU":
            return len(self.lfu_cache)
        return len(self.cache)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.get_stats()
    
    def save_to_disk(self):
        """保存缓存到磁盘"""
        if not self.enable_persistence:
            return
        
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                "cache": dict(self.cache) if self.eviction_policy == "LRU" else None,
                "lfu_cache": {k: {
                    "key": v.key,
                    "value": v.value,
                    "access_count": v.access_count,
                    "last_access": v.last_access,
                    "created_at": v.created_at,
                    "priority": v.priority.value,
                    "ttl": v.ttl
                } for k, v in self.lfu_cache.items()} if self.eviction_policy == "LFU" else None,
                "ttl_map": self.ttl_map,
                "access_times": self.access_times,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"[OK] 缓存 {self.name} 已保存到磁盘: {self.persistence_path}")
        except Exception as e:
            print(f"⚠ 保存缓存失败: {e}")
    
    def _load_from_disk(self):
        """从磁盘加载缓存"""
        if not self.persistence_path.exists():
            return
        
        try:
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查数据是否过期（快照超过24小时则忽略）
            if "timestamp" in cache_data:
                snapshot_time = datetime.fromisoformat(cache_data["timestamp"])
                if (datetime.now() - snapshot_time).total_seconds() > 86400:
                    print(f"⚠ 缓存快照已过期（超过24小时），跳过加载")
                    return
            
            if self.eviction_policy == "LFU" and cache_data.get("lfu_cache"):
                for k, v in cache_data["lfu_cache"].items():
                    entry = LFUCacheEntry(v["key"], v["value"], CachePriority(v["priority"]))
                    entry.access_count = v["access_count"]
                    entry.last_access = v["last_access"]
                    entry.created_at = v["created_at"]
                    entry.ttl = v["ttl"]
                    self.lfu_cache[k] = entry
            
            if self.eviction_policy == "LRU" and cache_data.get("cache"):
                self.cache = OrderedDict(cache_data["cache"])
            
            if cache_data.get("ttl_map"):
                self.ttl_map = cache_data["ttl_map"]
            
            if cache_data.get("access_times"):
                self.access_times = cache_data["access_times"]
            
            print(f"[OK] 缓存 {self.name} 已从磁盘加载: {len(self.cache) if self.eviction_policy == 'LRU' else len(self.lfu_cache)} 项")
        except Exception as e:
            print(f"⚠ 加载缓存失败: {e}")

class UnifiedCacheManager:
    """统一缓存管理器（支持多级缓存、统计、持久化）"""
    
    def __init__(self, redis_client=None, enable_persistence: bool = True):
        # redis_client参数保留以兼容旧代码，但不再使用
        self.enable_persistence = enable_persistence
        
        # 创建不同类型的缓存实例
        self.emotion_cache = AdvancedCache(
            "emotion", max_size=256, eviction_policy="LRU",
            enable_persistence=enable_persistence
        )
        self.role_cache = AdvancedCache(
            "role", max_size=256, eviction_policy="LRU",
            enable_persistence=enable_persistence
        )
        self.user_data_cache = AdvancedCache(
            "user_data", max_size=64, eviction_policy="LFU",
            enable_persistence=enable_persistence
        )
        self.conversation_cache = AdvancedCache(
            "conversation", max_size=128, eviction_policy="LRU",
            enable_persistence=enable_persistence
        )
        self.vector_cache = AdvancedCache(
            "vector", max_size=1000, eviction_policy="LRU",
            enable_persistence=enable_persistence
        )
    
    def get_emotion(self, user_input: str, conversation_history: list = None) -> Optional[Dict]:
        """获取情绪分析缓存"""
        key = CacheKeyBuilder.build_key(CacheType.EMOTION, user_input, 
                                        conversation_history=conversation_history)
        return self.emotion_cache.get(key)
    
    def set_emotion(self, user_input: str, result: Dict, 
                   conversation_history: list = None,
                   priority: CachePriority = CachePriority.MEDIUM):
        """设置情绪分析缓存"""
        key = CacheKeyBuilder.build_key(CacheType.EMOTION, user_input,
                                      conversation_history=conversation_history)
        
        ttl = self.emotion_cache.priority_ttl.get(priority)
        self.emotion_cache.set(key, result, ttl=ttl, priority=priority)
    
    def get_role(self, user_input: str, conversation_history: list = None) -> Optional[Dict]:
        """获取角色分析缓存"""
        key = CacheKeyBuilder.build_key(CacheType.ROLE, user_input,
                                      conversation_history=conversation_history)
        return self.role_cache.get(key)
    
    def set_role(self, user_input: str, result: Dict,
                conversation_history: list = None,
                priority: CachePriority = CachePriority.MEDIUM):
        """设置角色分析缓存"""
        key = CacheKeyBuilder.build_key(CacheType.ROLE, user_input,
                                      conversation_history=conversation_history)
        
        ttl = self.role_cache.priority_ttl.get(priority)
        self.role_cache.set(key, result, ttl=ttl, priority=priority)
    
    def get_user_data(self, user_id: str) -> Optional[Dict]:
        """获取用户数据缓存"""
        key = CacheKeyBuilder.build_user_key(CacheType.USER_DATA, user_id)
        return self.user_data_cache.get(key)
    
    def set_user_data(self, user_id: str, data: Dict,
                     priority: CachePriority = CachePriority.HIGH):
        """设置用户数据缓存"""
        key = CacheKeyBuilder.build_user_key(CacheType.USER_DATA, user_id)
        
        ttl = self.user_data_cache.priority_ttl.get(priority)
        self.user_data_cache.set(key, data, ttl=ttl, priority=priority)
    
    def invalidate_user_data(self, user_id: str):
        """失效用户数据缓存"""
        pattern = f"user:{user_id}"
        for cache in [self.user_data_cache, self.conversation_cache]:
            cache.invalidate_pattern(pattern)
    
    def get_vector(self, text: str, memory_type: str = "text") -> Optional[List[float]]:
        """获取向量缓存"""
        key = CacheKeyBuilder.build_vector_key(text, memory_type)
        return self.vector_cache.get(key)
    
    def set_vector(self, text: str, vector: List[float], 
                  memory_type: str = "text",
                  priority: CachePriority = CachePriority.HIGH):
        """设置向量缓存"""
        key = CacheKeyBuilder.build_vector_key(text, memory_type)
        
        ttl = self.vector_cache.priority_ttl.get(priority)
        self.vector_cache.set(key, vector, ttl=ttl, priority=priority)
    
    def get_all_stats(self) -> Dict:
        """获取所有缓存的统计信息"""
        return {
            "emotion": self.emotion_cache.get_stats(),
            "role": self.role_cache.get_stats(),
            "user_data": self.user_data_cache.get_stats(),
            "conversation": self.conversation_cache.get_stats(),
            "vector": self.vector_cache.get_stats()
        }
    
    def save_all_caches(self):
        """保存所有缓存到磁盘"""
        if self.enable_persistence:
            self.emotion_cache.save_to_disk()
            self.role_cache.save_to_disk()
            self.user_data_cache.save_to_disk()
            self.conversation_cache.save_to_disk()
            self.vector_cache.save_to_disk()
    
    def clear_all(self):
        """清空所有缓存"""
        self.emotion_cache.clear()
        self.role_cache.clear()
        self.user_data_cache.clear()
        self.conversation_cache.clear()
        self.vector_cache.clear()

# 全局缓存管理器实例
_cache_manager: Optional[UnifiedCacheManager] = None

def get_cache_manager(redis_client=None, enable_persistence: bool = True) -> UnifiedCacheManager:
    """获取缓存管理器实例（单例）"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = UnifiedCacheManager(redis_client, enable_persistence)
    return _cache_manager

