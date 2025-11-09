"""
全局异常处理和日志记录中间件
"""
import logging
import time
from typing import Callable
from fastapi import Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from config import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.StreamHandler(),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志记录中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """记录请求日志"""
        start_time = time.time()
        
        # 记录请求开始
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            f"请求开始: {request.method} {request.url.path} "
            f"来自 {client_ip}"
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算耗时
            process_time = time.time() - start_time
            
            # 记录请求完成
            logger.info(
                f"请求完成: {request.method} {request.url.path} "
                f"状态码: {response.status_code} "
                f"耗时: {process_time:.3f}s"
            )
            
            # 添加响应头
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # 记录异常
            process_time = time.time() - start_time
            logger.error(
                f"请求异常: {request.method} {request.url.path} "
                f"错误: {str(e)} "
                f"耗时: {process_time:.3f}s",
                exc_info=True
            )
            raise

# 全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理"""
    logger.warning(f"请求验证失败: {request.url.path} - {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "请求参数验证失败",
            "detail": exc.errors()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    logger.warning(f"HTTP异常: {request.url.path} - {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(
        f"未处理的异常: {request.method} {request.url.path}",
        exc_info=True
    )
    
    # 生产环境隐藏详细错误信息
    if settings.DEBUG:
        error_detail = str(exc)
    else:
        error_detail = "服务器内部错误，请联系管理员"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "服务器内部错误",
            "detail": error_detail
        }
    )

# 注意：这个文件需要在app.py导入后使用，所以需要调整导入顺序

