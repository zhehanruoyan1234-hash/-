"""
批量操作管理器
使用Qdrant批量接口和异步批量写入提高效率
"""
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class PendingMemory:
    """待保存的记忆"""
    memory_id: str
    vector: List[float]
    payload: Dict
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BatchMemoryBuffer:
    """批量记忆缓冲区"""
    
    def __init__(self, user_id: str, batch_size: int = 10, flush_interval: float = 5.0):
        """
        :param user_id: 用户ID
        :param batch_size: 批量大小（达到此数量时立即刷新）
        :param flush_interval: 刷新间隔（秒）
        """
        self.user_id = user_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._buffer: deque = deque()
        self._lock = asyncio.Lock()
        self._flush_task = None
        self._running = False
    
    async def add(self, memory: PendingMemory):
        """添加记忆到缓冲区"""
        async with self._lock:
            self._buffer.append(memory)
            
            # 如果达到批量大小，立即刷新
            if len(self._buffer) >= self.batch_size:
                await self._flush()
            
            # 启动后台刷新任务（如果未启动）
            if not self._running:
                self._start_flush_task()
    
    def _start_flush_task(self):
        """启动后台刷新任务"""
        if self._running:
            return
        
        self._running = True
        
        async def flush_loop():
            while self._running:
                try:
                    await asyncio.sleep(self.flush_interval)
                    async with self._lock:
                        if self._buffer:
                            await self._flush()
                except Exception as e:
                    logger.error(f"刷新任务错误: {e}")
        
        self._flush_task = asyncio.create_task(flush_loop())
    
    async def _flush(self):
        """刷新缓冲区（批量写入Qdrant）"""
        if not self._buffer:
            return
        
        memories_to_save = list(self._buffer)
        self._buffer.clear()
        
        try:
            await self._batch_upsert(memories_to_save)
            logger.info(f"✓ 批量保存了 {len(memories_to_save)} 条记忆")
        except Exception as e:
            logger.error(f"批量保存失败: {e}")
            # 失败时重新加入缓冲区（避免丢失）
            self._buffer.extendleft(memories_to_save)
    
    async def _batch_upsert(self, memories: List[PendingMemory]):
        """批量写入Qdrant"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct
            from memory_manager import qdrant_client
            
            if not qdrant_client:
                logger.warning("Qdrant不可用，跳过批量写入")
                return
            
            collection_name = f"user_{self.user_id}_memories"
            
            # 构建批量点
            points = []
            for memory in memories:
                points.append(PointStruct(
                    id=hash(memory.memory_id),
                    vector=memory.vector,
                    payload=memory.payload
                ))
            
            # 使用批量接口
            # 在线程池中运行同步的Qdrant操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
            )
            
        except Exception as e:
            logger.error(f"批量写入Qdrant失败: {e}")
            raise
    
    async def flush(self):
        """手动刷新缓冲区"""
        async with self._lock:
            await self._flush()
    
    async def close(self):
        """关闭缓冲区（刷新剩余数据）"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self.flush()

# 全局缓冲区注册表（按用户ID）
_buffers: Dict[str, BatchMemoryBuffer] = {}

def get_batch_buffer(user_id: str) -> BatchMemoryBuffer:
    """获取批量缓冲区实例（单例模式）"""
    if user_id not in _buffers:
        _buffers[user_id] = BatchMemoryBuffer(user_id)
    return _buffers[user_id]

async def batch_retrieve_memories(
    user_id: str,
    queries: List[str],
    top_k: int = 5,
    collection_name: str = None
) -> List[List[Dict]]:
    """
    批量检索记忆（使用Qdrant的批量接口，使用embedding_manager）
    
    :param user_id: 用户ID
    :param queries: 查询列表
    :param top_k: 每个查询返回的结果数量
    :param collection_name: 集合名称
    :return: 每个查询的结果列表
    """
    try:
        from qdrant_client import QdrantClient
        from memory_manager import qdrant_client
        from embedding_manager import MemoryType, encode_text
        
        if not qdrant_client:
            return [[] for _ in queries]
        
        if collection_name is None:
            collection_name = f"user_{user_id}_memories"
        
        # 批量编码查询（使用embedding_manager）
        query_vectors = []
        for q in queries:
            vector = encode_text(q, MemoryType.TEXT)
            if vector:
                query_vectors.append(vector)
            else:
                query_vectors.append(None)
        
        # 过滤掉None的向量
        valid_queries = []
        valid_vectors = []
        for i, vec in enumerate(query_vectors):
            if vec is not None:
                valid_queries.append(i)
                valid_vectors.append(vec)
        
        if not valid_vectors:
            return [[] for _ in queries]
        
        # 使用Qdrant批量检索
        # 注意：Qdrant的批量检索需要使用scroll或search_batch
        # 这里使用异步并行执行多个search
        
        loop = asyncio.get_event_loop()
        
        async def search_one(query_vector):
            return await loop.run_in_executor(
                None,
                lambda: qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )
            )
        
        results = await asyncio.gather(*[search_one(qv) for qv in valid_vectors])
        
        # 处理结果
        formatted_results = []
        result_idx = 0
        for i in range(len(queries)):
            if i in valid_queries:
                formatted = []
                for result in results[result_idx]:
                    formatted.append({
                        "memory_id": result.payload.get("memory_id", ""),
                        "content": result.payload.get("content", ""),
                        "metadata": {
                            "summary": result.payload.get("summary", ""),
                            "memory_type": result.payload.get("memory_type", ""),
                            "is_schedule": result.payload.get("is_schedule", False),
                            "importance_score": result.payload.get("importance_score", 0.5)
                        },
                        "similarity": result.score,
                        "source": "qdrant"
                    })
                formatted_results.append(formatted)
                result_idx += 1
            else:
                formatted_results.append([])
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"批量检索失败: {e}")
        return [[] for _ in queries]

