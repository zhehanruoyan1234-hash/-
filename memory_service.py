"""
记忆服务接口（事件驱动架构）
解决LangGraph和MemoryManager之间的循环依赖问题
"""
from typing import Optional, Dict, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import asyncio
from enum import Enum

class MemoryEventType(Enum):
    """记忆事件类型"""
    EXTRACT_MEMORY = "extract_memory"
    SAVE_MEMORY = "save_memory"
    UPDATE_PROFILE = "update_profile"
    RETRIEVE_MEMORIES = "retrieve_memories"

@dataclass
class MemoryEvent:
    """记忆事件"""
    event_type: MemoryEventType
    user_id: str
    data: Dict
    callback: Optional[Callable] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MemoryServiceInterface(ABC):
    """记忆服务接口（抽象基类）"""
    
    @abstractmethod
    def extract_memory(self, conversation_data: Dict, emotion_analysis: Dict, 
                      role_preference: Dict) -> Optional[Dict]:
        """提取记忆"""
        pass
    
    @abstractmethod
    def save_memory(self, db, conversation_id: str, memory_info: Dict, content: str):
        """保存记忆"""
        pass
    
    @abstractmethod
    def update_user_profile(self, db, emotion_analysis: Dict, role_preference: Dict, user_input: str):
        """更新用户画像"""
        pass
    
    @abstractmethod
    def retrieve_memories(self, query: str, top_k: int = 5, prioritize_schedule: bool = True) -> List[Dict]:
        """检索记忆"""
        pass

class MemoryService(MemoryServiceInterface):
    """记忆服务实现（事件驱动）"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._memory_manager = None  # 延迟初始化，避免循环依赖
        self._event_handlers = {}
        self._event_queue = asyncio.Queue()
        self._background_task = None
    
    def _get_memory_manager(self):
        """延迟获取MemoryManager实例（使用缓存函数）"""
        if self._memory_manager is None:
            from memory_manager import get_memory_manager
            self._memory_manager = get_memory_manager(self.user_id)
        return self._memory_manager
    
    def register_handler(self, event_type: MemoryEventType, handler: Callable):
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def _process_event(self, event: MemoryEvent):
        """处理事件"""
        handlers = self._event_handlers.get(event.event_type, [])
        results = []
        
        for handler in handlers:
            try:
                result = await handler(event)
                results.append(result)
            except Exception as e:
                print(f"事件处理器失败: {e}")
        
        # 执行回调
        if event.callback:
            try:
                await event.callback(results)
            except Exception as e:
                print(f"事件回调失败: {e}")
        
        return results
    
    async def _event_loop(self):
        """事件循环（后台任务）"""
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"事件循环错误: {e}")
    
    def _start_event_loop(self):
        """启动事件循环"""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._event_loop())
    
    async def emit_event(self, event: MemoryEvent):
        """发送事件"""
        self._start_event_loop()
        await self._event_queue.put(event)
    
    def extract_memory(self, conversation_data: Dict, emotion_analysis: Dict, 
                      role_preference: Dict) -> Optional[Dict]:
        """提取记忆（同步接口）"""
        memory_manager = self._get_memory_manager()
        return memory_manager.extract_memory(conversation_data, emotion_analysis, role_preference)
    
    def save_memory(self, db, conversation_id: str, memory_info: Dict, content: str):
        """保存记忆（同步接口）"""
        memory_manager = self._get_memory_manager()
        return memory_manager._save_memory(db, conversation_id, memory_info, content)
    
    def update_user_profile(self, db, emotion_analysis: Dict, role_preference: Dict, user_input: str):
        """更新用户画像（同步接口）"""
        memory_manager = self._get_memory_manager()
        return memory_manager._update_user_profile(db, emotion_analysis, role_preference, user_input)
    
    def retrieve_memories(self, query: str, top_k: int = 5, prioritize_schedule: bool = True) -> List[Dict]:
        """检索记忆（同步接口）"""
        memory_manager = self._get_memory_manager()
        return memory_manager.retrieve_memories(query, top_k, prioritize_schedule)

# 全局服务注册表（避免重复创建）
_service_registry = {}

def get_memory_service(user_id: str) -> MemoryService:
    """获取记忆服务实例（单例模式）"""
    if user_id not in _service_registry:
        _service_registry[user_id] = MemoryService(user_id)
    return _service_registry[user_id]

