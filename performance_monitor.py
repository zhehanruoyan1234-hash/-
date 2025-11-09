"""
性能监控和可观测性模块
提供任务执行时间统计、错误率监控、资源使用追踪
"""
import time
import asyncio
from functools import wraps
from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import logging
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class TaskMetrics:
    """任务执行指标"""
    task_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fallback_used: bool = False
    
    def finish(self, success: bool = True, error: Optional[Exception] = None):
        """完成指标记录"""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        self.success = success
        if error:
            self.error_type = type(error).__name__
            self.error_message = str(error)

# 全局指标存储
_task_metrics: Dict[str, list] = {}

def record_task_metrics(task_name: str):
    """记录任务指标装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = TaskMetrics(task_name=task_name)
            try:
                result = await func(*args, **kwargs)
                metrics.finish(success=True)
                return result
            except Exception as e:
                metrics.finish(success=False, error=e)
                logger.error(f"任务 {task_name} 执行失败: {e}", exc_info=True)
                raise
            finally:
                # 记录指标
                if task_name not in _task_metrics:
                    _task_metrics[task_name] = []
                _task_metrics[task_name].append(metrics)
                # 保持最近100条记录
                if len(_task_metrics[task_name]) > 100:
                    _task_metrics[task_name] = _task_metrics[task_name][-100:]
        return async_wrapper
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = TaskMetrics(task_name=task_name)
            try:
                result = func(*args, **kwargs)
                metrics.finish(success=True)
                return result
            except Exception as e:
                metrics.finish(success=False, error=e)
                logger.error(f"任务 {task_name} 执行失败: {e}", exc_info=True)
                raise
            finally:
                # 记录指标
                if task_name not in _task_metrics:
                    _task_metrics[task_name] = []
                _task_metrics[task_name].append(metrics)
                # 保持最近100条记录
                if len(_task_metrics[task_name]) > 100:
                    _task_metrics[task_name] = _task_metrics[task_name][-100:]
        return sync_wrapper
        
        # 根据函数类型返回对应包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

@contextmanager
def task_timer(task_name: str):
    """任务计时上下文管理器"""
    metrics = TaskMetrics(task_name=task_name)
    try:
        yield metrics
        metrics.finish(success=True)
    except Exception as e:
        metrics.finish(success=False, error=e)
        raise
    finally:
        # 记录指标
        if task_name not in _task_metrics:
            _task_metrics[task_name] = []
        _task_metrics[task_name].append(metrics)
        # 保持最近100条记录
        if len(_task_metrics[task_name]) > 100:
            _task_metrics[task_name] = _task_metrics[task_name][-100:]
        
        # 记录日志
        if metrics.success:
            logger.info(f"任务 {task_name} 完成，耗时: {metrics.duration:.3f}s")
        else:
            logger.warning(
                f"任务 {task_name} 失败，耗时: {metrics.duration:.3f}s，"
                f"错误: {metrics.error_type}: {metrics.error_message}"
            )

def get_task_stats(task_name: str) -> Dict[str, Any]:
    """获取任务统计信息"""
    if task_name not in _task_metrics or not _task_metrics[task_name]:
        return {
            "task_name": task_name,
            "total_count": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_duration": 0,
            "min_duration": 0,
            "max_duration": 0,
            "error_rate": 0
        }
    
    metrics_list = _task_metrics[task_name]
    total = len(metrics_list)
    success_count = sum(1 for m in metrics_list if m.success)
    error_count = total - success_count
    
    durations = [m.duration for m in metrics_list if m.duration is not None]
    
    return {
        "task_name": task_name,
        "total_count": total,
        "success_count": success_count,
        "error_count": error_count,
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "min_duration": min(durations) if durations else 0,
        "max_duration": max(durations) if durations else 0,
        "error_rate": error_count / total if total > 0 else 0,
        "recent_errors": [
            {"type": m.error_type, "message": m.error_message}
            for m in metrics_list[-10:] if not m.success
        ]
    }

def get_all_task_stats() -> Dict[str, Dict[str, Any]]:
    """获取所有任务统计信息"""
    return {task_name: get_task_stats(task_name) for task_name in _task_metrics.keys()}

def reset_task_stats(task_name: Optional[str] = None):
    """重置任务统计（用于测试或定期清理）"""
    if task_name:
        _task_metrics.pop(task_name, None)
    else:
        _task_metrics.clear()

