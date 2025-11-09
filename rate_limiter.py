"""
API限流模块
Redis已移除，限流功能已禁用
"""
from typing import Optional
from fastapi import Request, Depends

# Redis已移除，限流功能已禁用
REDIS_AVAILABLE = False

from config import settings

async def init_rate_limiter():
    """初始化限流器（已禁用）"""
    print("⚠ API限流已禁用（Redis已移除）")

async def close_rate_limiter():
    """关闭限流器连接（已禁用）"""
    pass

def get_client_ip(request: Request) -> str:
    """获取客户端IP地址"""
    if request.client:
        return request.client.host
    return "unknown"

# 限流装饰器（已禁用）
def rate_limit(times: int = 60, seconds: int = 60):
    """
    限流装饰器（已禁用）
    :param times: 允许的请求次数（未使用）
    :param seconds: 时间窗口（秒）（未使用）
    """
    # 返回空依赖，不进行限流
    return Depends(lambda: None)

# 预定义的限流规则（已禁用）
RATE_LIMIT_CHAT = rate_limit(times=30, seconds=60)  # 聊天接口：每分钟30次（已禁用）
RATE_LIMIT_AUTH = rate_limit(times=5, seconds=60)   # 认证接口：每分钟5次（已禁用）
RATE_LIMIT_DEFAULT = rate_limit(times=60, seconds=60)  # 默认：每分钟60次（已禁用）

