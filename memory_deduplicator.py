"""
记忆去重机制
使用余弦相似度和SimHash快速检测重复记忆
"""
from typing import List, Dict, Optional, Set
import hashlib
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

# SimHash支持
try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False
    print("⚠ simhash未安装，将使用简化的去重机制")

@dataclass
class MemorySignature:
    """记忆签名"""
    memory_id: str
    simhash: Optional[int] = None
    vector: Optional[List[float]] = None
    content_hash: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MemoryDeduplicator:
    """记忆去重器"""
    
    def __init__(self, similarity_threshold: float = 0.95, simhash_threshold: int = 10):
        """
        :param similarity_threshold: 余弦相似度阈值（0-1），超过此值视为重复
        :param simhash_threshold: SimHash汉明距离阈值，小于此值视为重复
        """
        self.similarity_threshold = similarity_threshold
        self.simhash_threshold = simhash_threshold
        self._signatures: Dict[str, MemorySignature] = {}  # memory_id -> signature
        self._simhash_index: Dict[int, List[str]] = defaultdict(list)  # simhash -> [memory_ids]
        self._content_hash_index: Dict[str, str] = {}  # content_hash -> memory_id
    
    def _compute_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _compute_simhash(self, content: str) -> Optional[int]:
        """计算SimHash"""
        if not SIMHASH_AVAILABLE:
            return None
        
        try:
            # 简化：使用词级别的SimHash
            words = content.split()
            if len(words) < 3:
                return None
            
            simhash = Simhash(words)
            return simhash.value
        except Exception as e:
            print(f"SimHash计算失败: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # 归一化
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
            
            return float(np.dot(vec1_norm, vec2_norm))
        except Exception as e:
            print(f"余弦相似度计算失败: {e}")
            return 0.0
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        return bin(hash1 ^ hash2).count('1')
    
    def add_signature(self, memory_id: str, content: str, vector: Optional[List[float]] = None):
        """添加记忆签名"""
        content_hash = self._compute_content_hash(content)
        simhash = self._compute_simhash(content)
        
        signature = MemorySignature(
            memory_id=memory_id,
            simhash=simhash,
            vector=vector,
            content_hash=content_hash
        )
        
        self._signatures[memory_id] = signature
        
        # 更新索引
        if simhash is not None:
            self._simhash_index[simhash].append(memory_id)
        
        self._content_hash_index[content_hash] = memory_id
    
    def is_duplicate(self, content: str, vector: Optional[List[float]] = None) -> Optional[str]:
        """
        检查是否为重复记忆
        返回：如果重复，返回已存在的memory_id；否则返回None
        """
        # 1. 快速检查：完全相同的文本（内容哈希）
        content_hash = self._compute_content_hash(content)
        if content_hash in self._content_hash_index:
            return self._content_hash_index[content_hash]
        
        # 2. SimHash快速检查（如果可用）
        if SIMHASH_AVAILABLE and vector is None:
            simhash = self._compute_simhash(content)
            if simhash is not None:
                # 检查汉明距离相近的签名
                for existing_hash, memory_ids in self._simhash_index.items():
                    distance = self._hamming_distance(simhash, existing_hash)
                    if distance < self.simhash_threshold:
                        # 进一步检查向量相似度（如果有）
                        existing_id = memory_ids[0]
                        existing_sig = self._signatures.get(existing_id)
                        if existing_sig and existing_sig.vector and vector:
                            similarity = self._cosine_similarity(vector, existing_sig.vector)
                            if similarity >= self.similarity_threshold:
                                return existing_id
                        elif distance < self.simhash_threshold // 2:  # 非常相似
                            return existing_id
        
        # 3. 向量相似度检查（最准确但最慢）
        if vector is not None:
            for memory_id, signature in self._signatures.items():
                if signature.vector is None:
                    continue
                
                similarity = self._cosine_similarity(vector, signature.vector)
                if similarity >= self.similarity_threshold:
                    return memory_id
        
        return None
    
    def remove_signature(self, memory_id: str):
        """移除记忆签名"""
        signature = self._signatures.pop(memory_id, None)
        if signature:
            # 从索引中移除
            if signature.simhash is not None and signature.simhash in self._simhash_index:
                self._simhash_index[signature.simhash].remove(memory_id)
                if not self._simhash_index[signature.simhash]:
                    del self._simhash_index[signature.simhash]
            
            if signature.content_hash in self._content_hash_index:
                del self._content_hash_index[signature.content_hash]
    
    def cleanup_old_signatures(self, max_age_days: int = 90):
        """清理旧签名"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        to_remove = []
        for memory_id, signature in self._signatures.items():
            if signature.timestamp < cutoff_date:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            self.remove_signature(memory_id)
        
        return len(to_remove)

# 全局去重器实例（按用户ID）
_deduplicators: Dict[str, MemoryDeduplicator] = {}

def get_deduplicator(user_id: str) -> MemoryDeduplicator:
    """获取去重器实例（单例模式）"""
    if user_id not in _deduplicators:
        _deduplicators[user_id] = MemoryDeduplicator()
    return _deduplicators[user_id]

