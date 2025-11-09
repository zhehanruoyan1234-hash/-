"""
错误处理和降级策略模块
提供多层级降级机制和错误分类
"""
from typing import Optional, Callable, Any, List
from enum import Enum
import logging
from functools import wraps
import asyncio
import inspect

logger = logging.getLogger(__name__)

class ErrorLevel(Enum):
    """错误级别"""
    CRITICAL = "critical"  # 严重错误，需要人工干预
    ERROR = "error"        # 一般错误，可以使用降级策略
    WARNING = "warning"     # 警告，不影响功能
    INFO = "info"          # 信息

class FallbackLevel(Enum):
    """降级级别"""
    PRIMARY = "primary"      # 主方案
    SECONDARY = "secondary"  # 备用方案1
    TERTIARY = "tertiary"    # 备用方案2
    KEYWORD = "keyword"      # 关键词策略（最后降级）

class FallbackStrategy:
    """降级策略配置"""
    def __init__(
        self, 
        name: str, 
        fallback_func: Callable,
        timeout: float = 5.0,
        error_types: List[type] = None,
        condition: Optional[Callable] = None
    ):
        self.name = name
        self.fallback_func = fallback_func
        self.timeout = timeout
        self.error_types = error_types
        self.condition = condition
    
    def should_trigger(self, error: Exception) -> bool:
        """判断是否应该触发此降级策略"""
        if self.error_types and type(error) not in self.error_types:
            return False
        if self.condition and not self.condition(error):
            return False
        return True

class FallbackManager:
    """降级管理器"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.strategies: List[FallbackStrategy] = []
        self.current_level = FallbackLevel.PRIMARY
    
    def add_strategy(self, strategy: FallbackStrategy):
        """添加降级策略"""
        self.strategies.append(strategy)
    
    async def execute_with_fallback(self, primary_func: Callable, *args, **kwargs) -> Any:
        """执行主函数，失败时自动降级"""
        # 尝试主方案
        try:
            logger.debug(f"[{self.task_name}] 尝试主方案")
            # 检查是否是异步函数
            if asyncio.iscoroutinefunction(primary_func):
                result = await asyncio.wait_for(
                    primary_func(*args, **kwargs),
                    timeout=self.strategies[0].timeout if self.strategies else 5.0
                )
            else:
                # 同步函数在线程池中运行
                loop = asyncio.get_event_loop()
                # 修复：检查函数签名，正确处理参数传递
                try:
                    sig = inspect.signature(primary_func)
                    # 如果函数不接受参数，直接调用
                    if len(sig.parameters) == 0:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, primary_func),
                            timeout=self.strategies[0].timeout if self.strategies else 5.0
                        )
                    else:
                        # 函数接受参数，传递参数
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: primary_func(*args, **kwargs)),
                            timeout=self.strategies[0].timeout if self.strategies else 5.0
                        )
                except (ValueError, TypeError):
                    # 如果无法检查签名（如 lambda），尝试传递参数
                    if args or kwargs:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: primary_func(*args, **kwargs)),
                            timeout=self.strategies[0].timeout if self.strategies else 5.0
                        )
                    else:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, primary_func),
                            timeout=self.strategies[0].timeout if self.strategies else 5.0
                        )
            self.current_level = FallbackLevel.PRIMARY
            logger.info(f"[{self.task_name}] 主方案执行成功")
            return result
        except Exception as e:
            logger.warning(f"[{self.task_name}] 主方案失败: {type(e).__name__}: {e}")
            
            # 尝试降级策略
            for i, strategy in enumerate(self.strategies):
                if not strategy.should_trigger(e):
                    continue
                
                try:
                    logger.info(f"[{self.task_name}] 尝试降级方案 {i+1}: {strategy.name}")
                    if asyncio.iscoroutinefunction(strategy.fallback_func):
                        result = await asyncio.wait_for(
                            strategy.fallback_func(*args, **kwargs),
                            timeout=strategy.timeout
                        )
                    else:
                        # 同步函数在线程池中运行
                        loop = asyncio.get_event_loop()
                        # 修复：检查函数签名，正确处理参数传递
                        try:
                            sig = inspect.signature(strategy.fallback_func)
                            # 如果函数不接受参数，直接调用
                            if len(sig.parameters) == 0:
                                result = await asyncio.wait_for(
                                    loop.run_in_executor(None, strategy.fallback_func),
                                    timeout=strategy.timeout
                                )
                            else:
                                # 函数接受参数，传递参数
                                result = await asyncio.wait_for(
                                    loop.run_in_executor(None, lambda: strategy.fallback_func(*args, **kwargs)),
                                    timeout=strategy.timeout
                                )
                        except (ValueError, TypeError):
                            # 如果无法检查签名，尝试传递参数
                            if args or kwargs:
                                result = await asyncio.wait_for(
                                    loop.run_in_executor(None, lambda: strategy.fallback_func(*args, **kwargs)),
                                    timeout=strategy.timeout
                                )
                            else:
                                result = await asyncio.wait_for(
                                    loop.run_in_executor(None, strategy.fallback_func),
                                    timeout=strategy.timeout
                                )
                    
                    self.current_level = FallbackLevel.SECONDARY if i == 0 else FallbackLevel.TERTIARY
                    logger.info(f"[{self.task_name}] 降级方案 {i+1} 执行成功")
                    return result
                except Exception as fallback_error:
                    logger.warning(
                        f"[{self.task_name}] 降级方案 {i+1} 失败: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
                    continue
            
            # 所有方案都失败
            logger.error(f"[{self.task_name}] 所有降级方案均失败")
            raise e

def classify_error(error: Exception) -> ErrorLevel:
    """错误分类"""
    error_type = type(error).__name__
    
    # 严重错误
    if error_type in ["MemoryError", "SystemError", "KeyboardInterrupt"]:
        return ErrorLevel.CRITICAL
    
    # 一般错误
    if error_type in ["TimeoutError", "ConnectionError", "HTTPException"]:
        return ErrorLevel.ERROR
    
    # 警告
    if error_type in ["ValueError", "TypeError", "KeyError"]:
        return ErrorLevel.WARNING
    
    return ErrorLevel.INFO

# 预定义的降级策略
def create_emotion_analysis_fallback():
    """创建情绪分析的降级策略"""
    def keyword_fallback(user_input: str = None, *args, **kwargs):
        """关键词降级策略"""
        # 修复：添加空值检查，防止 NoneType 错误
        if not user_input:
            return {
                "emotion_type": "neutral",
                "emotion_intensity": "中等"
            }
        # 动态导入避免循环依赖
        from app import quick_emotion_check
        result = quick_emotion_check(user_input)
        return result or {
            "emotion_type": "neutral",
            "emotion_intensity": "中等"
        }
    
    return [
        FallbackStrategy(
            name="关键词策略",
            fallback_func=keyword_fallback,
            timeout=1.0
        )
    ]

def create_role_analysis_fallback():
    """创建角色分析的降级策略"""
    def default_role_fallback(*args, **kwargs):
        """默认角色降级策略"""
        return {
            "role_type": "朋友",
            "communication_style": "温和关怀",
            "tone": "随和"
        }
    
    return [
        FallbackStrategy(
            name="默认角色策略",
            fallback_func=default_role_fallback,
            timeout=0.1
        )
    ]

