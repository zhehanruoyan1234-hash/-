from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi import status
from google import genai
from google.genai import types
from datetime import datetime, timedelta
import json
import re
import torch
from transformers import pipeline
import numpy as np
from database import get_db, get_or_create_user, User, UserProfile, Conversation, Memory
from memory_manager import maintain_memories, get_memory_manager
from cache_manager import (
    get_emotion_cache, set_emotion_cache,
    get_role_cache, set_role_cache,
    get_user_data_cache, set_user_data_cache,
    get_conversation_history_cache, set_conversation_history_cache,
    clear_user_caches
)
# 尝试导入新的高级缓存系统（向后兼容）
try:
    from advanced_cache import get_cache_manager, CachePriority
    ADVANCED_CACHE_AVAILABLE = True
except ImportError:
    ADVANCED_CACHE_AVAILABLE = False
    print("⚠ 高级缓存系统未安装，使用传统缓存")

# 导入配置模块
from config import settings
# 导入优化的角色分析模块
try:
    from role_analysis import analyze_role_preference_optimized
    OPTIMIZED_ROLE_ANALYSIS_AVAILABLE = True
except ImportError:
    OPTIMIZED_ROLE_ANALYSIS_AVAILABLE = False
    print("⚠ 优化角色分析模块未导入，使用传统方法")
import asyncio
import schedule
import time
import logging
import threading
from starlette.middleware.base import BaseHTTPMiddleware

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="未来搭子 AI陪伴智能体", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# 静态文件服务
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"⚠ 静态文件目录不存在，跳过静态文件服务: {e}")

# 请求日志记录中间件
class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志记录中间件"""
    
    async def dispatch(self, request: Request, call_next):
        """记录请求日志"""
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        logger.info(
            f"请求开始: {request.method} {request.url.path} "
            f"来自 {client_ip}"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"请求完成: {request.method} {request.url.path} "
                f"状态码: {response.status_code} "
                f"耗时: {process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"请求异常: {request.method} {request.url.path} "
                f"错误: {str(e)} "
                f"耗时: {process_time:.3f}s",
                exc_info=True
            )
            raise

# 添加日志中间件
app.add_middleware(LoggingMiddleware)

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

# 启动时初始化限流器和维护任务
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    # 初始化高级缓存系统（如果可用）
    if ADVANCED_CACHE_AVAILABLE:
        try:
            cache_manager = get_cache_manager(None, enable_persistence=True)
            logger.info("[OK] 高级缓存系统初始化成功")
        except Exception as e:
            logger.warning(f"高级缓存系统初始化失败: {e}")
    
    # 启动维护任务调度器
    asyncio.create_task(run_maintenance_scheduler_async())
    logger.info("[OK] 维护任务调度器已启动")
    
    # 初始化限流器
    try:
        from rate_limiter import init_rate_limiter
        await init_rate_limiter()
    except Exception as e:
        logger.warning(f"限流器初始化失败: {e}")
    
    logger.info("[OK] 应用启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    # 保存缓存到磁盘
    if ADVANCED_CACHE_AVAILABLE:
        try:
            cache_manager = get_cache_manager()
            cache_manager.save_all_caches()
            logger.info("[OK] 缓存已保存到磁盘")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    try:
        from rate_limiter import close_rate_limiter
        await close_rate_limiter()
    except Exception as e:
        logger.warning(f"限流器关闭失败: {e}")

# 默认用户ID（简易版本，后续可以扩展为真实用户系统）
DEFAULT_USER_ID = "default_user"

# 启动定期维护任务（使用asyncio替代threading）
async def run_maintenance_scheduler_async():
    """异步运行定期维护任务"""
    import schedule
    
    # 每天凌晨2点执行维护
    schedule.every().day.at("02:00").do(maintain_memories)
    # 每小时也执行一次（用于测试，生产环境可以改为每天）
    schedule.every().hour.do(maintain_memories)
    
    while True:
        schedule.run_pending()
        await asyncio.sleep(3600)  # 每小时检查一次

# 设置环境变量，禁用 Hugging Face symlink 警告（Windows 不支持 symlink）
import os
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

# 初始化 Gemini 客户端，直接使用硬编码 API key
try:
    client = genai.Client(api_key="AIzaSyAJjK0XYvoitU0saAqb0useLbDed4nH3dQ")
    logger.info("[OK] Gemini客户端初始化成功")
except Exception as e:
    logger.error(f"⚠ Gemini客户端初始化失败: {e}")
    client = None

# 初始化 Transformers 模型（延迟加载，带降级）
_transformers_models = {}

def _init_transformer_model(model_name: str, model_type: str):
    """初始化Transformers模型（统一逻辑，带降级）
    优化1：使用公共工具模块设置缓存
    """
    if model_name not in _transformers_models:
        try:
            from model_utils import setup_transformers_cache, get_device_id
            
            device = get_device_id()
            
            # 优化1：使用公共工具模块设置缓存
            cache_dir = setup_transformers_cache()
            
            _transformers_models[model_name] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
            print(f"[OK] {model_type}加载成功 (Transformers, 缓存目录: {cache_dir})")
        except Exception as e:
            print(f"加载{model_type}失败，将使用 Gemini API 作为备用方案: {e}")
            _transformers_models[model_name] = None
    return _transformers_models.get(model_name)

def get_emotion_analyzer():
    """延迟加载情绪分析模型（使用GoEmotions多语言模型）
    模型：SchuylerH/bert-multilingual-go-emtions
    支持28种情绪类别，准确率85.95%，支持中英文
    
    使用Transformers pipeline作为高级辅助工具，简化代码
    优化1：使用公共工具模块设置缓存
    """
    if "emotion_analyzer" not in _transformers_models:
        try:
            from transformers import pipeline
            from model_utils import setup_transformers_cache, get_device_id
            
            device = get_device_id()
            
            # 优化1：使用公共工具模块设置缓存
            cache_dir = setup_transformers_cache()
            
            # 使用pipeline作为高级辅助工具（推荐方式）
            # 这会自动处理tokenizer和model的加载
            _transformers_models["emotion_analyzer"] = pipeline(
                "text-classification",
                model="SchuylerH/bert-multilingual-go-emtions",
                device=device
            )
            print(f"[OK] 情绪分析模型加载成功 (GoEmotions多语言模型, 缓存目录: {cache_dir})")
        except Exception as e:
            print(f"加载GoEmotions情绪分析模型失败: {e}，将使用 Gemini API 作为备用方案")
            _transformers_models["emotion_analyzer"] = None
    
    return _transformers_models.get("emotion_analyzer")

# 已删除 get_role_classifier 函数
# 原因：DeBERTa-v3 需要 SentencePiece 库，环境不支持，已完全使用 Gemini API
# def get_role_classifier():
#     """延迟加载角色分类器（带降级）"""
#     return _init_transformer_model("role_classifier", "角色分类器")


def get_beijing_time():
    """返回当前北京时间字符串"""
    now_utc = datetime.utcnow()
    beijing_time = now_utc + timedelta(hours=8)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

def get_beijing_datetime():
    """返回当前北京时间的datetime对象"""
    now_utc = datetime.utcnow()
    beijing_time = now_utc + timedelta(hours=8)
    return beijing_time

def get_user_by_id_or_name(db, identifier: str):
    """通过ID或名称查询用户（统一逻辑）"""
    user = db.query(User).filter(User.user_id == identifier).first()
    if not user:
        user = db.query(User).filter(User.name == identifier).first()
    return user

def safe_json_parse(json_str: str, default=None):
    """安全的JSON解析（带降级）"""
    if not json_str:
        return default or {}
    try:
        return json.loads(json_str)
    except:
        return default or {}

def get_time_period(dt: datetime = None) -> str:
    """获取时间段（凌晨/早上/上午/中午/下午/晚上/深夜）
    22-2点：深夜，2-5点：凌晨，6点开始：早上
    """
    if dt is None:
        dt = get_beijing_datetime()
    
    hour = dt.hour
    
    if hour >= 22 or hour < 2:  # 22点-2点：深夜
        return "深夜"
    elif 2 <= hour < 6:  # 2点-5点：凌晨
        return "凌晨"
    elif 6 <= hour < 9:  # 6点开始是早上
        return "早上"
    elif 9 <= hour < 12:
        return "上午"
    elif 12 <= hour < 14:
        return "中午"
    elif 14 <= hour < 18:
        return "下午"
    else:  # 18 <= hour < 22
        return "晚上"

def get_time_interval_desc(last_time: datetime) -> str:
    """获取时间间隔描述"""
    if last_time is None:
        return "首次对话"
    
    now = get_beijing_datetime()
    delta = now - last_time
    
    if delta.days > 7:
        return f"{delta.days}天前"
    elif delta.days > 0:
        return f"{delta.days}天前"
    elif delta.seconds >= 3600:
        hours = delta.seconds // 3600
        return f"{hours}小时前"
    elif delta.seconds >= 60:
        minutes = delta.seconds // 60
        return f"{minutes}分钟前"
    else:
        return "刚刚"

def should_generate_greeting(user_input: str, last_conversation_time: datetime = None, recent_conversations_count: int = 0, very_recent_conv_count: int = 0) -> bool:
    """
    判断是否应该生成问候（仅在开启新对话时）
    规则：仅在首次对话或距离上次对话超过30分钟时问候，避免连续对话中重复问候
    """
    # 如果最近有连续对话（5分钟内），不生成问候
    if very_recent_conv_count > 0:
        return False
    
    # 如果没有历史对话，肯定是首次问候
    if last_conversation_time is None:
        return True
    
    # 如果距离上次对话超过30分钟，生成问候（从3小时改为30分钟，更合理）
    if last_conversation_time:
        now = get_beijing_datetime()
        delta = now - last_conversation_time
        if delta.total_seconds() >= 1800:  # 30分钟（1800秒）
            return True
    
    # 如果用户输入包含明确的问候关键词，且距离上次对话超过15分钟，可能是重新开始对话
    greeting_keywords = ["你好", "hello", "hi", "在吗", "开始", "凌晨好", "早上好", "晚上好", "下午好", "回来了"]
    if any(keyword in user_input.lower() for keyword in greeting_keywords):
        if last_conversation_time:
            now = get_beijing_datetime()
            delta = now - last_conversation_time
            if delta.total_seconds() >= 900:  # 15分钟
                return True
    
    # 如果这是本次会话的第一次对话（没有最近的对话记录），且距离上次超过30分钟，生成问候
    if recent_conversations_count == 0 and last_conversation_time:
        now = get_beijing_datetime()
        delta = now - last_conversation_time
        # 如果没有最近的对话，且距离上次超过30分钟，可能是新会话开始
        if delta.total_seconds() >= 1800:  # 30分钟
            return True
    
    return False

def generate_time_aware_greeting(last_conversation_time: datetime = None, recent_topics: list = None, 
                                  user_profile: dict = None, conversation_history: list = None) -> str:
    """生成基于时间、用户画像和对话历史的智能问候语"""
    time_period = get_time_period()
    time_interval = get_time_interval_desc(last_conversation_time)
    
    greeting_templates = {
        "凌晨": ["凌晨好", "这么早", "凌晨时分"],
        "早上": ["早上好", "早安", "新的一天开始了"],
        "上午": ["上午好", "上午时光"],
        "中午": ["中午好", "午安", "午间时光"],
        "下午": ["下午好", "下午时光"],
        "晚上": ["晚上好", "晚上好", "夜晚时光"],
        "深夜": ["这么晚了", "深夜时分", "这么晚了还没休息"]
    }
    
    base_greeting = greeting_templates.get(time_period, ["你好"])[0]
    
    # 从用户画像中获取信息
    comm_prefs = user_profile.get("communication_preferences", {}) if user_profile else {}
    emotion_patterns = user_profile.get("emotional_patterns", {}) if user_profile else {}
    
    # 获取用户偏好的角色
    preferred_role = None
    if "role_stats" in comm_prefs and comm_prefs["role_stats"]:
        preferred_role = max(comm_prefs["role_stats"].items(), key=lambda x: x[1])[0]
    
    # 获取用户最活跃时间段
    active_period = emotion_patterns.get("most_active_period", "")
    
    # 获取最近的重要记忆（行程等）
    important_memories = []
    if conversation_history:
        for conv in conversation_history[-3:]:
            # 检查是否有行程相关的内容
            user_msg = conv.get("user_input", "")
            if any(kw in user_msg for kw in ["要去", "准备", "计划", "行程", "出差", "旅行", "开会", "见面", "约会"]):
                important_memories.append(user_msg[:40])
    
    # 构建更自然的问候语（不使用直白的连接词）
    greeting = ""
    
    # 首次对话的问候
    if last_conversation_time is None:
        if time_period == "凌晨":
            greeting = "凌晨好，这么早就开始新的一天了"
        elif time_period == "早上":
            greeting = "早上好"
        elif time_period == "上午":
            greeting = "上午好"
        elif time_period == "中午":
            greeting = "中午好"
        elif time_period == "下午":
            greeting = "下午好"
        elif time_period == "晚上":
            greeting = "晚上好"
        elif time_period == "深夜":
            greeting = "这么晚了还没休息啊"
    else:
        # 再次对话的问候
        if "天前" in time_interval:
            days = time_interval.replace("天前", "").strip()
            try:
                days_num = int(days)
                if days_num >= 7:
                    greeting = f"好久不见，{base_greeting}"
                elif days_num >= 3:
                    greeting = f"{base_greeting}，几天不见了呢"
                else:
                    greeting = base_greeting
            except:
                greeting = base_greeting
        elif "小时前" in time_interval:
            hours = time_interval.replace("小时前", "").strip()
            try:
                hours_num = int(hours)
                if hours_num >= 12:
                    greeting = base_greeting
                else:
                    # 短时间内再次对话，用更自然的问候
                    greeting = base_greeting
            except:
                greeting = base_greeting
        else:
            greeting = base_greeting
    
    # 结合时间段和活跃模式的自然提醒（不直白）
    if active_period and active_period != time_period:
        if time_period == "深夜" and active_period in ["早上", "上午", "下午"]:
            greeting += "，今天这么晚还在呢"
        elif time_period == "凌晨":
            greeting += "，这么早就起来了"
        elif time_period in ["早上", "上午"] and active_period == "晚上":
            greeting += "，难得这个时间见到你"
    
    # 提及重要记忆（行程）- 更自然的表达
    if important_memories:
        last_memory = important_memories[-1]
        if len(last_memory) < 40:
            # 提取关键信息，自然融入
            memory_summary = last_memory.replace("要去", "").replace("准备", "").replace("计划", "").strip()
            if memory_summary:
                # 自然地提及，而不是直接说"还记得"
                if "出差" in last_memory or "旅行" in last_memory:
                    greeting += f"，{memory_summary}的事怎么样了？"
                elif "考试" in last_memory or "面试" in last_memory:
                    greeting += f"，{memory_summary}还好吗？"
                else:
                    greeting += f"，{memory_summary}？"
    
    # 结合最近话题（自然提及，不直接说"上次聊到"）
    elif recent_topics and len(recent_topics) > 0:
        last_topic = recent_topics[0]
        if last_topic and len(last_topic) < 30:
            # 更自然的表达，不提"上次"
            if "考试" in last_topic or "工作" in last_topic:
                greeting += f"，关于{last_topic}的事有进展吗？"
            else:
                greeting += f"，{last_topic}后来怎么样了？"
    
    return greeting

def get_recent_conversation_history(db, user_id: str, limit: int = 5) -> list:
    """获取最近的对话历史（带缓存优化）"""
    from database import Conversation
    
    # 先检查缓存
    cached = get_conversation_history_cache(user_id, limit)
    if cached is not None:
        return cached
    
    # 查询数据库
    recent_convs = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.timestamp.desc()).limit(limit).all()
    
    history = []
    for conv in reversed(recent_convs):  # 按时间正序
        history.append({
            "user_input": conv.user_input,
            "ai_response": conv.ai_response,
            "emotion_analysis": safe_json_parse(conv.emotion_analysis, {}),
            "role_preference": safe_json_parse(conv.role_preference, {}),
            "timestamp": conv.timestamp.isoformat() if conv.timestamp else None
        })
    
    # 缓存结果
    set_conversation_history_cache(user_id, limit, history)
    return history

def get_user_data_batch(db, user_id: str) -> dict:
    """批量获取用户数据（优化：一次性查询用户、画像、最近对话，带缓存）"""
    # 先检查缓存（但返回原始对象，不依赖缓存）
    cached = get_user_data_cache(user_id)
    
    # 批量查询：用户、画像、最近一条对话（合并查询减少数据库访问）
    user = db.query(User).filter(User.user_id == user_id).first()
    
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    
    last_conversation = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.timestamp.desc()).first()
    
    # 解析用户画像
    user_profile = {}
    if profile:
        user_profile = {
            "communication_preferences": safe_json_parse(profile.communication_preferences, {}),
            "emotional_patterns": safe_json_parse(profile.emotional_patterns, {})
        }
    
    last_conversation_time = last_conversation.timestamp if last_conversation else None
    recent_topics = [last_conversation.user_input[:30]] if last_conversation else []
    
    # 缓存结果（5分钟过期）
    set_user_data_cache(user_id, {
        "last_conversation_time": last_conversation_time.isoformat() if last_conversation_time else None,
        "user_profile": user_profile,
        "recent_topics": recent_topics
    })
    
    # 返回原始对象（供后续使用）
    return {
        "user": user,
        "profile": profile,
        "last_conversation": last_conversation,
        "last_conversation_time": last_conversation_time,
        "user_profile": user_profile,
        "recent_topics": recent_topics
    }

def analyze_emotion_with_history(user_input: str, current_time: str, conversation_history: list = None) -> dict:
    """
    综合分析用户情绪（考虑对话历史，带缓存优化）
    如果提供了历史，会综合分析情绪趋势
    """
    # 先检查缓存
    cached = get_emotion_cache(user_input, conversation_history)
    if cached is not None:
        return cached
    
    # 先分析当前输入
    current_emotion = analyze_emotion(user_input, current_time)
    
    # 如果没有历史，直接返回当前分析
    if not conversation_history or len(conversation_history) == 0:
        return current_emotion
    
    # 分析历史情绪趋势
    emotion_counts = {}
    emotion_intensities = []
    
    for conv in conversation_history:
        if conv.get("emotion_analysis"):
            emotion_type = conv["emotion_analysis"].get("emotion_type", "neutral")
            emotion_counts[emotion_type] = emotion_counts.get(emotion_type, 0) + 1
            
            # 记录强度（转换为数值）
            intensity = conv["emotion_analysis"].get("emotion_intensity", "中等")
            intensity_value = {"轻微": 1, "中等": 2, "强烈": 3}.get(intensity, 2)
            emotion_intensities.append(intensity_value)
    
    # 计算平均强度和情绪趋势
    avg_intensity_value = sum(emotion_intensities) / len(emotion_intensities) if emotion_intensities else 2
    
    # 确定主导情绪（历史中最常见的）
    if emotion_counts:
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    else:
        dominant_emotion = current_emotion["emotion_type"]
    
    # 判断情绪是否在变化
    emotion_trend = "稳定"
    if len(emotion_counts) > 1:
        # 如果有多种情绪，判断趋势
        recent_emotions = [conv.get("emotion_analysis", {}).get("emotion_type", "neutral") 
                          for conv in conversation_history[-3:]]
        if len(set(recent_emotions)) > 1:
            emotion_trend = "波动"
    
    # 综合判断：结合当前情绪和历史趋势
    final_emotion_type = current_emotion["emotion_type"]
    
    # 如果历史中某种情绪出现频率很高（超过50%），且当前也是这种情绪，则加强
    if emotion_counts.get(final_emotion_type, 0) / len(conversation_history) >= 0.5:
        # 情绪一致性高，保持当前判断
        pass
    elif dominant_emotion != final_emotion_type and emotion_counts.get(dominant_emotion, 0) >= 2:
        # 如果历史中有明显的情绪模式，且当前情绪不同，可能需要调整
        # 这里可以选择跟随趋势或保持当前（保守策略：保持当前）
        pass
    
    # 根据平均强度和当前强度综合判断
    if avg_intensity_value >= 2.5:
        final_intensity = "强烈"
    elif avg_intensity_value >= 1.5:
        final_intensity = "中等"
    else:
        final_intensity = "轻微"
    
    # 如果当前强度和历史平均差异大，使用当前强度；否则使用综合判断
    current_intensity_value = {"轻微": 1, "中等": 2, "强烈": 3}.get(current_emotion["emotion_intensity"], 2)
    if abs(current_intensity_value - avg_intensity_value) >= 1:
        # 差异大，使用当前
        final_intensity = current_emotion["emotion_intensity"]
    
    # 检查当前情绪是否被压抑
    is_suppressed = current_emotion.get("is_suppressed", False)
    
    result = {
        "emotion_type": final_emotion_type,
        "emotion_intensity": final_intensity,
        "emotion_trend": emotion_trend,  # 新增：情绪趋势
        "dominant_emotion_history": dominant_emotion,  # 历史主导情绪
        "is_consistent": emotion_trend == "稳定",  # 是否一致
        "is_suppressed": is_suppressed  # 是否压抑情绪
    }
    
    # 缓存结果（1小时过期）
    set_emotion_cache(user_input, result, conversation_history, ttl=3600)
    
    return result

def analyze_role_preference_with_history(user_input: str, current_time: str, conversation_history: list = None) -> dict:
    """
    综合分析用户期望的AI角色类型（考虑对话历史）
    返回：{"role_type": "朋友", "communication_style": "轻松幽默", "tone": "随和"}
    """
    # 先分析当前输入
    current_role = analyze_role_preference(user_input, current_time)
    
    # 如果没有历史，直接返回当前分析
    if not conversation_history or len(conversation_history) == 0:
        return current_role
    
    # 分析历史角色偏好
    role_counts = {}
    style_counts = {}
    tone_counts = {}
    
    for conv in conversation_history:
        if conv.get("role_preference"):
            role_type = conv["role_preference"].get("role_type", "朋友")
            style = conv["role_preference"].get("communication_style", "温和关怀")
            tone = conv["role_preference"].get("tone", "随和")
            
            role_counts[role_type] = role_counts.get(role_type, 0) + 1
            style_counts[style] = style_counts.get(style, 0) + 1
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
    
    # 综合判断：如果历史中有明显偏好，且与当前一致，则使用；否则优先使用当前
    final_role_type = current_role["role_type"]
    final_style = current_role["communication_style"]
    final_tone = current_role["tone"]
    
    # 如果历史中某种角色出现频率很高，可以考虑使用
    if role_counts:
        dominant_role = max(role_counts.items(), key=lambda x: x[1])[0]
        if role_counts.get(dominant_role, 0) / len(conversation_history) >= 0.6:
            # 如果历史偏好很明显（60%以上），且当前也类似，使用历史偏好
            if dominant_role == current_role["role_type"]:
                final_role_type = dominant_role
    
    if style_counts:
        dominant_style = max(style_counts.items(), key=lambda x: x[1])[0]
        if style_counts.get(dominant_style, 0) / len(conversation_history) >= 0.6:
            if dominant_style == current_role["communication_style"]:
                final_style = dominant_style
    
    if tone_counts:
        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])[0]
        if tone_counts.get(dominant_tone, 0) / len(conversation_history) >= 0.6:
            if dominant_tone == current_role["tone"]:
                final_tone = dominant_tone
    
    result = {
        "role_type": final_role_type,
        "communication_style": final_style,
        "tone": final_tone,
        "preference_consistency": len(set([r.get("role_preference", {}).get("role_type", "朋友") 
                                          for r in conversation_history])) <= 2  # 偏好是否一致
    }
    
    # 缓存结果（1小时过期）
    set_role_cache(user_input, result, conversation_history, ttl=3600)
    
    return result

def analyze_role_preference(user_input: str, current_time: str) -> dict:
    """
    使用 Gemini API 分析用户期望的AI角色类型（单句版本）
    已删除 DeBERTa-v3 方案，直接使用 Gemini API
    返回：{"role_type": "朋友", "communication_style": "轻松幽默", "tone": "随和"}
    """
    valid_roles = ["朋友", "导师", "倾听者", "玩伴", "安慰者", "伙伴", "陪伴者"]
    valid_styles = ["轻松幽默", "温和关怀", "专业理性", "活泼热情", "平静内敛", "随性自然"]
    valid_tones = ["随和", "正式", "亲密", "中性"]
    
    # 直接使用 Gemini API（已删除 DeBERTa-v3 方案）
    if client is None:
        return {
            "role_type": "朋友",
            "communication_style": "温和关怀",
            "tone": "随和"
        }
    
    try:
        role_prompt = f"""你是一位专业的沟通分析专家。请分析用户输入，判断用户期望的对话伙伴类型。

当前时间：{current_time}
用户输入：{user_input}

请分析：
1. **角色类型**：从以下类型中选择最匹配的一个
   - "朋友"：像好朋友一样轻松聊天
   - "导师"：提供建议和指导
   - "倾听者"：专注倾听和陪伴
   - "玩伴"：一起娱乐和放松
   - "安慰者"：提供情感支持和安慰
   - "伙伴"：平等交流的伙伴关系
   - "陪伴者"：温和陪伴的伙伴

2. **沟通风格**：选择最适合的沟通方式
   - "轻松幽默"：轻松愉快，带点幽默
   - "温和关怀"：温和体贴，充满关怀
   - "专业理性"：专业理性，条理清晰
   - "活泼热情"：活泼热情，充满活力
   - "平静内敛"：平静内敛，温和沉静
   - "随性自然"：随性自然，不拘束

3. **语调特点**：分析用户期望的语调
   - "随和"：随和友好，不拘小节
   - "正式"：稍微正式，保持距离
   - "亲密"：亲密自然，像熟人
   - "中性"：中性平衡，不过于随意也不过于正式

请以JSON格式输出，格式如下：
{{
    "role_type": "角色类型",
    "communication_style": "沟通风格",
    "tone": "语调特点"
}}

只输出JSON，不要其他文字说明。"""

        role_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=role_prompt
        )
        
        response_text = role_response.text.strip()
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            role_data = json.loads(json_match.group())
        else:
            role_data = json.loads(response_text)
        
        role_type = role_data.get("role_type", "朋友")
        communication_style = role_data.get("communication_style", "温和关怀")
        tone = role_data.get("tone", "随和")
        
        if role_type not in valid_roles:
            role_type = "朋友"
        if communication_style not in valid_styles:
            communication_style = "温和关怀"
        if tone not in valid_tones:
            tone = "随和"
            
        return {
            "role_type": role_type,
            "communication_style": communication_style,
            "tone": tone
        }
        
    except Exception as e:
        print(f"角色分析错误: {e}")
        return {
            "role_type": "朋友",
            "communication_style": "温和关怀",
            "tone": "随和"
        }

def quick_emotion_check(user_input: str) -> dict:
    """快速情绪检查（关键词匹配，用于明显情况）"""
    user_input_lower = user_input.lower()
    
    # GoEmotions相关的快速关键词匹配
    negative_keywords = {
        "sadness": ["差", "不好", "糟糕", "难过", "失落", "伤心", "哭了", "想哭"],
        "disappointment": ["失望", "失望了", "不满意"],
        "fear": ["担心", "害怕", "紧张", "恐惧", "慌"],
        "nervousness": ["焦虑", "压力", "不安"],
        "anger": ["生气", "愤怒", "气死了"],
        "grief": ["失去", "分手", "离开", "被辞", "裁员"]
    }
    
    positive_keywords = {
        "joy": ["好", "开心", "高兴", "快乐", "棒", "不错"],
        "excitement": ["太棒", "太好了", "兴奋", "激动", "完美"],
        "pride": ["成功", "通过", "赢了", "自豪"],
        "gratitude": ["感谢", "谢谢", "感激"]
    }
    
    # 强负面情绪词（高置信度）
    strong_negative_patterns = {
        "sadness": ["很差", "非常差", "太糟糕", "很难过", "很伤心"],
        "grief": ["考试差", "挂科", "失败", "丢了", "分手"],
        "fear": ["很焦虑", "非常担心", "很害怕"]
    }
    
    # 强正面情绪词（高置信度）
    strong_positive_patterns = {
        "joy": ["太好了", "非常开心", "很高兴"],
        "excitement": ["太棒", "很兴奋", "完美"],
        "pride": ["成功", "通过", "赢了"]
    }
    
    # 检查强负面情绪
    for emotion, patterns in strong_negative_patterns.items():
        if any(pattern in user_input for pattern in patterns):
            return {"emotion_type": emotion, "emotion_intensity": "强烈", "quick": True}
    
    # 检查强正面情绪
    for emotion, patterns in strong_positive_patterns.items():
        if any(pattern in user_input for pattern in patterns):
            return {"emotion_type": emotion, "emotion_intensity": "强烈", "quick": True}
    
    # 匹配GoEmotions情绪类别（中等强度）
    for emotion, keywords in negative_keywords.items():
        if any(word in user_input_lower for word in keywords):
            return {"emotion_type": emotion, "emotion_intensity": "中等", "quick": True}
    
    for emotion, keywords in positive_keywords.items():
        if any(word in user_input_lower for word in keywords):
            return {"emotion_type": emotion, "emotion_intensity": "中等", "quick": True}
    
    return None  # 不明显，需要完整分析

def analyze_emotion(user_input: str, current_time: str) -> dict:
    """
    深层情绪分析：使用GoEmotions多语言模型进行情绪识别
    直接使用GoEmotions模型的28种情绪分类，不进行映射
    返回：{"emotion_type": "joy", "emotion_intensity": "中等"}
    
    GoEmotions的28种情绪：
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    """
    # GoEmotions的28种情绪标签
    go_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    valid_intensities = ["轻微", "中等", "强烈"]
    
    try:
        analyzer = get_emotion_analyzer()
        
        if analyzer is not None:
            # 使用GoEmotions模型进行情绪分类
            # 该模型直接输出28种情绪类别之一
            result = analyzer(user_input)
            
            # 处理pipeline返回的结果格式
            if isinstance(result, list) and len(result) > 0:
                # pipeline返回格式: [{"label": "joy", "score": 0.95}]
                go_emotion_label = result[0].get("label", "neutral")
                emotion_score = result[0].get("score", 0.5)
            elif isinstance(result, dict):
                # 兼容其他可能的返回格式
                go_emotion_label = result.get("label", result.get("labels", ["neutral"])[0] if isinstance(result.get("labels"), list) else "neutral")
                emotion_score = result.get("score", result.get("scores", [0.5])[0] if isinstance(result.get("scores"), list) else 0.5)
            else:
                go_emotion_label = "neutral"
                emotion_score = 0.5
            
            # 标准化情绪标签（转换为小写）
            go_emotion_label = go_emotion_label.lower().strip()
            
            # 确保情绪标签在有效列表中
            if go_emotion_label not in go_emotions:
                # 尝试部分匹配
                matched = False
                for valid_emotion in go_emotions:
                    if valid_emotion in go_emotion_label or go_emotion_label in valid_emotion:
                        go_emotion_label = valid_emotion
                        matched = True
                        break
                if not matched:
                    print(f"⚠ 未识别的GoEmotions情绪标签: {go_emotion_label}，使用默认值：neutral")
                    go_emotion_label = "neutral"
            
            # 根据置信度分数调整强度
            # GoEmotions模型通常有较高的置信度，使用更细粒度的阈值
            if emotion_score >= 0.85:
                emotion_intensity = "强烈"
            elif emotion_score >= 0.65:
                emotion_intensity = "中等"
            elif emotion_score >= 0.45:
                emotion_intensity = "轻微"
            else:
                emotion_intensity = "中等"  # 默认中等强度
            
            # 确保强度在有效列表中
            if emotion_intensity not in valid_intensities:
                emotion_intensity = "中等"
                    
            return {
                "emotion_type": go_emotion_label,  # 直接使用GoEmotions标签
                "emotion_intensity": emotion_intensity,
                "confidence": emotion_score
            }
        
        # 如果 Transformers 模型不可用，使用 Gemini API 进行深层分析
        print("使用 Gemini API 进行深层情绪分析")
        emotion_prompt = f"""你是一位专业的情感分析专家，擅长理解用户话语的真实含义和隐含情绪。
请深入分析用户输入，不仅要看表面语气，还要理解话语背后的真实情感。

当前时间：{current_time}
用户输入：{user_input}

**重要提示**：
1. 用户可能用平静的语气描述负面事件（如"考试差了"、"工作丢了"），但实际情绪应该是负面情绪
2. 用户可能压抑真实情绪，用平淡的语气掩盖内心感受
3. 要理解话语的内容和含义，而不仅仅是语气
4. 负面事件即使语气平静，情绪也应该是负面的

请从GoEmotions的28种情绪中选择最符合的一项：
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
'relief', 'remorse', 'sadness', 'surprise', 'neutral'

**判断规则**：
- 如果用户描述负面事件（失败、失去、困难等），即使语气平静，也应该是sadness/disappointment/fear等
- 如果用户描述正面事件（成功、获得、好事等），应该是joy/excitement/pride等
- 如果用户表达困惑、不确定，可能是confusion/nervousness/fear
- 如果用户说"还好"、"没事"但描述的是负面事件，可能是disappointment/sadness（在压抑情绪）

请以JSON格式输出，格式如下：
{{
    "emotion_type": "GoEmotions情绪标签（小写，如joy、sadness、fear等）",
    "emotion_intensity": "轻微/中等/强烈",
    "is_suppressed": true/false
}}

只输出JSON，不要其他文字说明。"""

        emotion_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=emotion_prompt
        )
        
        response_text = emotion_response.text.strip()
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            emotion_data = json.loads(json_match.group())
        else:
            emotion_data = json.loads(response_text)
        
        # 获取GoEmotions情绪标签
        emotion_type = emotion_data.get("emotion_type", "neutral").lower().strip()
        emotion_intensity = emotion_data.get("emotion_intensity", "中等")
        is_suppressed = emotion_data.get("is_suppressed", False)
        
        # 验证情绪标签是否在GoEmotions的28种情绪中
        go_emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        if emotion_type not in go_emotions:
            # 尝试部分匹配
            matched = False
            for valid_emotion in go_emotions:
                if valid_emotion in emotion_type or emotion_type in valid_emotion:
                    emotion_type = valid_emotion
                    matched = True
                    break
            if not matched:
                print(f"⚠ Gemini返回的未知情绪标签: {emotion_type}，使用默认值：neutral")
                emotion_type = "neutral"
        
        if emotion_intensity not in valid_intensities:
            emotion_intensity = "中等"
        
        # 如果检测到情绪被压抑，适当提高强度
        if is_suppressed and emotion_type in ["sadness", "disappointment", "fear", "grief", "remorse"]:
            if emotion_intensity == "轻微":
                emotion_intensity = "中等"
            
        return {
            "emotion_type": emotion_type,  # 直接使用GoEmotions标签
            "emotion_intensity": emotion_intensity,
            "is_suppressed": is_suppressed  # 标记是否压抑情绪
        }
        
    except Exception as e:
        print(f"情绪分析错误: {e}")
        return {
            "emotion_type": "neutral",  # 使用GoEmotions的neutral标签
            "emotion_intensity": "中等",
            "is_suppressed": False
        }


def detect_user_intent(user_input: str) -> dict:
    """识别用户意图：问题求解、信息查询、情感支持等"""
    intent = {
        "is_question": False,
        "needs_advice": False,
        "needs_info": False,
        "needs_comfort": False
    }
    
    if not user_input or len(user_input.strip()) < 2:
        return intent
    
    # 问题关键词
    question_keywords = ["怎么", "如何", "有没有", "什么方法", "怎么办", "如何做", "建议", "方法", "技巧", "怎么学", "如何提高", "怎样"]
    advice_keywords = ["方法", "建议", "技巧", "怎么学", "如何提高", "怎么办", "怎样做"]
    info_keywords = ["是什么", "什么意思", "解释", "介绍", "定义"]
    
    user_input_lower = user_input.lower()
    
    if any(kw in user_input_lower for kw in question_keywords):
        intent["is_question"] = True
    
    if any(kw in user_input_lower for kw in advice_keywords):
        intent["needs_advice"] = True
    
    if any(kw in user_input_lower for kw in info_keywords):
        intent["needs_info"] = True
    
    # 如果包含负面情绪词但没有明确问题，可能是需要安慰
    if not intent["is_question"] and not intent["needs_advice"]:
        negative_words = ["伤心", "难过", "沮丧", "焦虑", "担心", "害怕", "失望"]
        if any(kw in user_input_lower for kw in negative_words):
            intent["needs_comfort"] = True
    
    return intent

def build_response_prompt(user_input: str, current_time: str, emotion_analysis: dict, role_preference: dict, user_context: str = "", greeting: str = "") -> str:
    """
    根据情绪分析和角色偏好构建AI回复的Prompt（时间感知版本）
    增强版：识别用户意图，平衡情感支持和实用建议
    """
    emotion_type = emotion_analysis["emotion_type"]
    emotion_intensity = emotion_analysis["emotion_intensity"]
    role_type = role_preference["role_type"]
    communication_style = role_preference["communication_style"]
    tone = role_preference["tone"]
    time_period = get_time_period()
    
    # 识别用户意图
    user_intent = detect_user_intent(user_input)
    
    # 根据情绪类型和强度调整回复策略（适配GoEmotions的28种情绪）
    # 将GoEmotions情绪分类为积极/消极/中性，用于指导回复策略
    positive_emotions = ['joy', 'amusement', 'excitement', 'pride', 'optimism', 'love', 'gratitude', 'admiration', 'approval', 'relief']
    negative_emotions = ['sadness', 'grief', 'disappointment', 'remorse', 'disapproval', 'anger', 'annoyance', 'disgust', 'fear', 'nervousness', 'confusion', 'embarrassment']
    neutral_emotions = ['neutral', 'realization', 'curiosity', 'desire', 'caring', 'surprise']
    
    # 判断情绪类型（积极/消极/中性）
    is_positive = emotion_type in positive_emotions
    is_negative = emotion_type in negative_emotions
    is_neutral = emotion_type in neutral_emotions
    
    # 根据用户意图和情绪倾向生成策略
    if user_intent["needs_advice"] or user_intent["is_question"]:
        # 用户需要建议或答案（优先处理）
        if is_negative:
            strategy = "先简短表达理解和共情（1-2句），然后**必须提供具体实用的建议和方法**。如果用户询问学习方法，提供3-5个具体可行的学习技巧。如果用户询问如何解决问题，提供清晰的步骤或方案。建议要具体、可操作，不要只是泛泛而谈。"
        else:
            strategy = "直接提供具体实用的建议和方法，回答要清晰、有条理。如果用户询问学习方法，提供3-5个具体可行的学习技巧。"
    elif user_intent["needs_info"]:
        # 用户需要信息
        strategy = "提供准确、清晰的信息，直接回答用户的问题。"
    elif user_intent["needs_comfort"]:
        # 用户需要情感支持
        if is_negative:
            if emotion_intensity == "强烈":
                strategy = "用非常温暖耐心的语气回应，给予充分的理解和情感支持，提供详细的疏导和安全感。"
            elif emotion_intensity == "中等":
                strategy = "用关怀支持的语气回应，深入理解对方的感受，提供情感陪伴。"
            else:
                strategy = "用温和安抚的语气回应，提供理解和倾听，给予轻度的安慰和鼓励。"
        else:
            strategy = "用温和友好的语气回应，提供陪伴和支持。"
    else:
        # 默认策略：根据情绪倾向和强度生成
        if is_positive:
            if emotion_intensity == "强烈":
                strategy = "用兴奋祝贺的语气回应，充分表达理解和共鸣，可以提出一些活动建议来延续这份快乐。"
            elif emotion_intensity == "中等":
                strategy = "用积极温暖的语气回应，可以表达为对方感到高兴，并鼓励分享更多。"
            else:
                strategy = "用轻松愉快的语气回应，可以分享一些有趣的观察或鼓励保持这份好心情。"
        elif is_negative:
            if emotion_intensity == "强烈":
                strategy = "用非常温暖耐心的语气回应，给予充分的理解和情感支持，提供详细的疏导和安全感。"
            elif emotion_intensity == "中等":
                strategy = "用关怀支持的语气回应，深入理解对方的感受，提供具体的建议和陪伴。"
            else:
                strategy = "用温和安抚的语气回应，提供理解和倾听，给予轻度的安慰和鼓励。"
        else:  # 中性情绪
            if emotion_intensity == "强烈":
                strategy = "用非常平和温暖的语气回应，提供深度陪伴，营造宁静舒适的氛围。"
            elif emotion_intensity == "中等":
                strategy = "用温和友好的语气回应，提供陪伴和支持，保持舒适的对话氛围。"
            else:
                strategy = "用平和自然的语气回应，保持轻松舒适的对话节奏。"
    
    # 根据角色类型构建角色描述（增强版：强调提供实用建议）
    role_descriptions = {
        "朋友": "像好朋友一样，轻松自然，可以开玩笑，也可以分享心事。当朋友需要建议时，要提供实用的帮助。",
        "导师": "提供建议和指导，但保持温和，不过于说教。**当用户询问问题时，必须提供具体可行的建议和方法**。",
        "倾听者": "专注倾听和理解，给予陪伴和情感支持。但如果用户明确询问问题，也要提供建议。",
        "玩伴": "一起娱乐放松，用轻松愉快的方式交流。当玩伴需要帮助时，也要提供建议。",
        "安慰者": "提供情感安慰和支持，帮助缓解负面情绪。**但如果用户询问具体问题，也要提供实用建议**。",
        "伙伴": "平等的伙伴关系，互相交流，互相陪伴。当伙伴需要帮助时，提供实用的建议。",
        "陪伴者": "温和陪伴，给对方安全感和温暖。当陪伴者需要建议时，也要提供帮助。"
    }
    
    role_description = role_descriptions.get(role_type, "温和陪伴的伙伴")
    
    # 沟通风格描述
    style_descriptions = {
        "轻松幽默": "语气轻松愉快，可以适当幽默，但不过度",
        "温和关怀": "语气温和体贴，充满关怀和理解",
        "专业理性": "语气专业理性，条理清晰，但保持温暖",
        "活泼热情": "语气活泼热情，充满活力，让对话更有趣",
        "平静内敛": "语气平静内敛，温和沉静，给人安全感",
        "随性自然": "语气随性自然，不拘束，像和熟人聊天"
    }
    
    style_description = style_descriptions.get(communication_style, "温和关怀")
    
    # 语调描述
    tone_descriptions = {
        "随和": "语调随和友好，不拘小节，像和熟人聊天",
        "正式": "语调稍微正式，保持一定距离感，但不冷漠",
        "亲密": "语调亲密自然，像老朋友一样，但尊重边界",
        "中性": "语调中性平衡，不过于随意也不过于正式"
    }
    
    tone_description = tone_descriptions.get(tone, "随和")
    
    # 构建上下文信息
    context_section = ""
    if user_context:
        context_section = f"\n**用户上下文信息**：\n{user_context}\n"
    
    # 问候语部分（仅在真正需要问候时添加）
    greeting_section = ""
    if greeting:
        # 检查用户是否主动问候
        greeting_keywords = ["你好", "hello", "hi", "在吗", "开始", "凌晨好", "早上好", "晚上好", "下午好", "回来了"]
        user_input_lower = user_input.lower() if user_input else ""
        user_actively_greeted = any(kw in user_input_lower for kw in greeting_keywords)
        
        if user_actively_greeted:
            # 用户主动问候，可以回应
            greeting_section = f"\n**重要提示**：用户主动问候，可以在回复开头自然地回应以下问候：{greeting}\n"
        else:
            # 用户没有主动问候，这是系统主动问候（首次对话或长时间未对话）
            greeting_section = f"\n**重要提示**：这是新对话的开始，用户可能刚刚启动对话。可以在回复开头自然地融入以下问候：{greeting}\n"
    
    # 情绪趋势信息
    emotion_trend_info = ""
    if emotion_analysis.get("emotion_trend"):
        trend = emotion_analysis.get("emotion_trend", "稳定")
        dominant = emotion_analysis.get("dominant_emotion_history", "")
        is_consistent = emotion_analysis.get("is_consistent", True)
        is_suppressed = emotion_analysis.get("is_suppressed", False)
        
        if trend == "波动" or not is_consistent:
            emotion_trend_info = f"\n- 情绪趋势：{trend}（最近情绪有所变化，注意观察）"
        if dominant and dominant != emotion_type:
            emotion_trend_info += f"\n- 历史主导情绪：{dominant}（但当前是{emotion_type}，注意情绪变化）"
        
        # 如果检测到情绪被压抑，特别提示
        if is_suppressed:
            emotion_trend_info += f"\n- ⚠️ 重要提示：用户可能用平静的语气描述负面事件，实际情绪应该是{emotion_type}，需要更多的关怀和理解"
    
    # 偏好一致性信息
    preference_info = ""
    if role_preference.get("preference_consistency") is False:
        preference_info = "\n- 偏好模式：用户的角色偏好在不同对话中有所变化，需要灵活适配"
    
    # 根据意图添加特殊提示
    intent_section = ""
    if user_intent["needs_advice"] or user_intent["is_question"]:
        intent_section = f"""
**⚠️ 重要提示**：用户明确询问了问题或需要建议（"{user_input}"），**必须提供具体实用的答案**：
- 如果询问学习方法，提供3-5个具体的学习技巧（如词根词缀法、联想记忆法、间隔重复等）
- 如果询问如何解决问题，提供清晰的步骤或方案
- 先简短共情（1-2句），然后重点回答实际问题
- 建议要具体、可操作，避免泛泛而谈
- **不要只安慰，必须回答问题**
"""
    elif user_intent["needs_info"]:
        intent_section = """
**⚠️ 重要提示**：用户需要信息，**必须直接回答用户的问题**，提供准确、清晰的信息。
"""
    
    # 根据意图调整字数限制
    word_limit = "150字内" if (user_intent["needs_advice"] or user_intent["is_question"] or user_intent["needs_info"]) else "100字内"
    
    # 精简Prompt以提高生成速度
    prompt = f"""你是AI陪伴智能体。角色：{role_description}。风格：{style_description}。语调：{tone_description}。

时间：{time_period}({current_time})
用户：{user_input if user_input else "问候场景"}
情绪：{emotion_type}({emotion_intensity}){emotion_trend_info}
{context_section}{greeting_section}{intent_section}

回复策略：{strategy}

原则：
1.理解真实情绪：{emotion_type}情绪需要{("更多关怀" if is_negative else ("积极回应" if is_positive else "平和陪伴"))}
2.保持角色一致性：{role_type}，{communication_style}风格
3.自然融入上下文和记忆，不要刻意提及
4.根据{time_period}调整语气
5.简洁回复，{word_limit}
6.**如果用户询问问题或需要建议，必须提供具体实用的答案，不要只安慰**
7.**如果是连续对话（用户没有主动问候），不要重复问候，直接回答用户的问题或继续对话**

{("如果用户压抑情绪或描述负面事件，给予更多关怀" if is_negative else "")}
{("**重要**：如果这是连续对话（用户没有主动问候），不要重复问候，直接回答用户的问题" if not greeting else "")}

回复："""

    return prompt

@app.get("/", response_class=HTMLResponse)
async def get_login(request: Request):
    """登录/用户选择页面"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/monitor", response_class=HTMLResponse)
async def get_monitor(request: Request):
    """监测页面"""
    return templates.TemplateResponse("monitor.html", {"request": request})

# API版本路由组
from fastapi import APIRouter

# v1版本API路由
api_v1 = APIRouter(prefix=settings.API_V1_PREFIX, tags=["v1"])

# 兼容旧版本路由（重定向到v1）
@app.get("/api/greeting/{user_id}")
@app.get("/greeting/{user_id}")  # 前端直接请求的路由
@app.get(f"{settings.API_V1_PREFIX}/greeting/{{user_id}}")
@api_v1.get("/greeting/{user_id}")
async def get_auto_greeting(user_id: str):
    """获取自动问候（对话开始时无需用户输入）"""
    try:
        current_time = get_beijing_time()
        db = next(get_db())
        
        # 获取用户信息
        user = get_or_create_user(db, user_id)
        
        # 获取上次对话时间
        last_conversation = db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.timestamp.desc()).first()
        last_conversation_time = last_conversation.timestamp if last_conversation else None
        
        # 获取最近话题
        recent_topics = []
        if last_conversation:
            recent_topics = [last_conversation.user_input[:30]]
        
        # 获取用户画像
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        user_profile = {}
        if profile:
            user_profile = {
                "communication_preferences": safe_json_parse(profile.communication_preferences, {}),
                "emotional_patterns": safe_json_parse(profile.emotional_patterns, {})
            }
        
        # 获取对话历史
        conversation_history = get_recent_conversation_history(db, user_id, limit=5)
        
        # 初始化记忆管理器（使用缓存实例）
        memory_manager = get_memory_manager(user_id)
        
        # 生成问候语
        greeting = generate_time_aware_greeting(
            last_conversation_time, 
            recent_topics, 
            user_profile,
            conversation_history
        )
        
        # 生成空的情绪和角色分析（用于问候）
        emotion_analysis = {
            "emotion_type": "neutral",
            "emotion_intensity": "轻微",
            "emotion_trend": "稳定",
            "is_consistent": True
        }
        
        # 根据用户画像确定角色偏好
        role_preference = {
            "role_type": "朋友",
            "communication_style": "温和关怀",
            "tone": "随和"
        }
        if user_profile.get("communication_preferences", {}).get("role_stats"):
            preferred_role = max(user_profile["communication_preferences"]["role_stats"].items(), key=lambda x: x[1])[0]
            role_preference["role_type"] = preferred_role
        
        # 生成问候回复
        user_context = memory_manager.get_user_context(db, "")  # 空输入，获取一般上下文
        
        # 构建问候专用Prompt
        greeting_prompt = f"""你现在是一位AI陪伴智能体，需要主动向用户问候。

        当前北京时间：{current_time}
当前时间段：{get_time_period()}

**问候信息**：{greeting}

**用户画像信息**：
{json.dumps(user_profile, ensure_ascii=False, indent=2) if user_profile else "新用户，暂无画像"}

**用户上下文**：
{user_context if user_context else "首次对话"}

**重要原则**：
1. 根据当前时间段（{get_time_period()}）调整问候的语气和内容
2. 如果是首次对话，要热情欢迎；如果是再次见面，要体现记忆
3. 如果用户画像显示有偏好角色（{role_preference.get('role_type', '朋友')}），用对应的语气
4. 如果有行程记忆，可以自然地提及
5. 问候要简洁温暖，控制在80字以内
6. 不要重复问候信息，要自然地展开对话邀请

请生成你的问候回复："""
        
        # 生成回复
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=greeting_prompt
        )
        
        ai_greeting_response = response.text if hasattr(response, 'text') else str(response)
        
        db.close()
        
        return JSONResponse({
            "greeting": greeting,
            "ai_response": ai_greeting_response,
            "timestamp": current_time,
            "emotion_analysis": emotion_analysis,
            "role_preference": role_preference
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)})

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str = None):
    """删除单个对话记录"""
    try:
        db = next(get_db())
        current_user_id = user_id if user_id else DEFAULT_USER_ID
        
        # 查找对话记录
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == current_user_id
        ).first()
        
        if not conversation:
            return JSONResponse(
                {"error": "对话记录不存在或无权访问"},
                status_code=404
            )
        
        # 删除对话记录
        db.delete(conversation)
        
        # 更新用户对话计数
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if user and user.total_conversations > 0:
            user.total_conversations -= 1
        
        db.commit()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": "对话记录已删除",
            "conversation_id": conversation_id
        })
    except Exception as e:
        return JSONResponse(
            {"error": f"删除失败: {str(e)}"},
            status_code=500
        )

@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = None):
    """删除单个记忆"""
    try:
        db = next(get_db())
        current_user_id = user_id if user_id else DEFAULT_USER_ID
        
        memory = db.query(Memory).filter(
            Memory.memory_id == memory_id,
            Memory.user_id == current_user_id
        ).first()
        
        if not memory:
            db.close()
            return JSONResponse(
                {"error": "记忆不存在或无权访问"},
                status_code=404
            )
        
        # 从ChromaDB删除
        try:
            memory_manager = get_memory_manager(current_user_id)
            memory_manager.collection.delete(ids=[memory_id])
        except:
            pass
        
        # 从SQLite删除
        db.delete(memory)
        db.commit()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": "记忆已删除",
            "memory_id": memory_id
        })
    except Exception as e:
        return JSONResponse(
            {"error": f"删除失败: {str(e)}"},
            status_code=500
        )

@app.delete("/api/memories")
async def delete_memories(
    request: Request,
    user_id: str = None,
    all: str = "false"
):
    """批量删除记忆"""
    try:
        db = next(get_db())
        current_user_id = user_id if user_id else DEFAULT_USER_ID
        
        query = db.query(Memory).filter(Memory.user_id == current_user_id)
        
        if all.lower() != "true":
            db.close()
            return JSONResponse(
                {"error": "请设置 all=true 来删除所有记忆"},
                status_code=400
            )
        
        memories_to_delete = query.all()
        deleted_count = len(memories_to_delete)
        
        if deleted_count == 0:
            db.close()
            return JSONResponse({
                "success": True,
                "message": "没有找到要删除的记忆",
                "deleted_count": 0
            })
        
        # 从ChromaDB删除
        try:
            memory_manager = get_memory_manager(current_user_id)
            memory_ids = [mem.memory_id for mem in memories_to_delete]
            memory_manager.collection.delete(ids=memory_ids)
        except Exception as e:
            print(f"从ChromaDB删除失败: {e}")
        
        # 从SQLite删除
        for memory in memories_to_delete:
            db.delete(memory)
        
        db.commit()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": f"已删除 {deleted_count} 条记忆",
            "deleted_count": deleted_count
        })
    except Exception as e:
        return JSONResponse(
            {"error": f"删除失败: {str(e)}"},
            status_code=500
        )

@app.delete("/api/user/{user_id}")
async def delete_user_data(user_id: str):
    """删除用户所有数据（对话、记忆、画像）"""
    try:
        db = next(get_db())
        
        # 检查用户是否存在
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            db.close()
            return JSONResponse({"error": "用户不存在"})
        
        # 删除所有对话
        conversations = db.query(Conversation).filter(Conversation.user_id == user_id).all()
        for conv in conversations:
            db.delete(conv)
        
        # 删除所有记忆
        memories = db.query(Memory).filter(Memory.user_id == user_id).all()
        # 从ChromaDB删除
        try:
            from memory_manager import get_memory_manager
            memory_manager = get_memory_manager(user_id)
            memory_ids = [mem.memory_id for mem in memories]
            if memory_ids:
                memory_manager.collection.delete(ids=memory_ids)
        except Exception as e:
            print(f"从ChromaDB删除失败: {e}")
        
        for mem in memories:
            db.delete(mem)
        
        # 删除用户画像
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if profile:
            db.delete(profile)
        
        # 重置用户数据（但不删除用户记录）
        user.total_conversations = 0
        user.preferred_role_patterns = "{}"
        user.dominant_emotions = "{}"
        
        db.commit()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": "用户所有数据已删除",
            "deleted_conversations": len(conversations),
            "deleted_memories": len(memories)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"删除失败: {str(e)}"},
            status_code=500
        )

@app.delete("/api/conversations")
async def delete_conversations(
    request: Request,
    user_id: str = None,
    before_date: str = None,  # ISO格式日期，删除此日期之前的记录
    all_records: str = "false"  # 是否删除所有记录（字符串参数）
):
    """
    批量删除对话记录
    参数（查询参数）:
    - user_id: 用户ID（可选，默认使用默认用户）
    - before_date: 删除此日期之前的记录（ISO格式，如"2024-01-01T00:00:00"）
    - all_records: "true" 表示删除所有记录（默认"false"）
    """
    try:
        from datetime import datetime as dt
        
        db = next(get_db())
        current_user_id = user_id if user_id else DEFAULT_USER_ID
        
        # 构建查询条件
        query = db.query(Conversation).filter(Conversation.user_id == current_user_id)
        
        # 如果请求删除所有记录
        if all_records.lower() == "true":
            # 获取所有记录以便统计
            conversations_to_delete = query.all()
        # 如果指定了日期，删除该日期之前的记录
        elif before_date:
            try:
                before_datetime = dt.fromisoformat(before_date.replace('Z', '+00:00'))
                query = query.filter(Conversation.timestamp < before_datetime)
                conversations_to_delete = query.all()
            except ValueError:
                db.close()
                return JSONResponse(
                    {"error": "日期格式错误，请使用ISO格式（如：2024-01-01T00:00:00）"},
                    status_code=400
                )
        else:
            db.close()
            return JSONResponse(
                {"error": "请指定删除条件：before_date（日期）或 all_records=true（删除所有）"},
                status_code=400
            )
        
        deleted_count = len(conversations_to_delete)
        
        if deleted_count == 0:
            db.close()
            return JSONResponse({
                "success": True,
                "message": "没有找到要删除的对话记录",
                "deleted_count": 0
            })
        
        # 删除记录
        for conversation in conversations_to_delete:
            db.delete(conversation)
        
        # 更新用户对话计数
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if user:
            user.total_conversations = max(0, user.total_conversations - deleted_count)
        
        db.commit()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": f"已删除 {deleted_count} 条对话记录",
            "deleted_count": deleted_count
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"删除失败: {str(e)}"},
            status_code=500
        )

@app.get("/api/conversations")
async def get_conversations(
    user_id: str = None,
    limit: int = 50,
    offset: int = 0
):
    """获取对话记录列表"""
    try:
        db = next(get_db())
        current_user_id = user_id if user_id else DEFAULT_USER_ID
        
        # 查询对话记录
        conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user_id
        ).order_by(Conversation.timestamp.desc()).offset(offset).limit(limit).all()
        
        # 获取总数
        total_count = db.query(Conversation).filter(
            Conversation.user_id == current_user_id
        ).count()
        
        # 转换为字典
        result = [conv.to_dict() for conv in conversations]
        
        db.close()
        
        return JSONResponse({
            "success": True,
            "conversations": result,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        return JSONResponse(
            {"error": f"获取失败: {str(e)}"},
            status_code=500
        )

@app.post("/chat")
@api_v1.post("/chat")
async def chat(
    request: Request, 
    user_input: str = Form(""), 
    user_id: str = Form(None),
    image: UploadFile = File(None)
):
    """
    聊天接口（一次性返回，提高响应速度）
    集成长期记忆系统，支持图片上传（多模态理解）
    根据Gemini API文档：https://ai.google.dev/gemini-api/docs/image-understanding
    """
    try:
        # 获取用户ID：优先使用传入的user_id，否则从Header获取，最后使用默认值
        if not user_id:
            # 尝试从请求头获取（前端通过Header传递）
            user_id = request.headers.get("X-User-ID")
        
        current_user_id = user_id if user_id else DEFAULT_USER_ID
        
        # 处理图片上传（如果有）
        image_part = None
        if image and image.filename:
            try:
                # 读取图片数据
                image_bytes = await image.read()
                
                # 验证文件大小（小于20MB）
                if len(image_bytes) > 20 * 1024 * 1024:
                    return JSONResponse({"error": "图片文件太大，请上传小于20MB的图片"}, status_code=400)
                
                # 验证MIME类型
                allowed_mime_types = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]
                if image.content_type not in allowed_mime_types:
                    return JSONResponse({"error": f"不支持的图片格式，支持：PNG、JPEG、WEBP、HEIC、HEIF"}, status_code=400)
                
                # 使用Gemini的types.Part.from_bytes创建图片部分
                # 根据文档：https://ai.google.dev/gemini-api/docs/image-understanding
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image.content_type
                )
            except Exception as e:
                print(f"处理图片失败: {e}")
                return JSONResponse({"error": f"处理图片失败：{str(e)}"}, status_code=400)
        
        # 验证用户是否存在
        db_check = next(get_db())
        user_check = db_check.query(User).filter(User.user_id == current_user_id).first()
        db_check.close()
        
        if not user_check:
            return JSONResponse({"error": "用户不存在，请先登录"}, status_code=401)
        
        # 验证：必须至少有文字输入或图片
        if not user_input.strip() and not image_part:
            return JSONResponse({"error": "请输入文字或上传图片"}, status_code=400)
        
        # 获取当前北京时间
        current_time = get_beijing_time()

        # 初始化记忆管理器（使用缓存实例）
        memory_manager = get_memory_manager(current_user_id)
        
        # 获取数据库会话
        db = next(get_db())
        
        # 确保用户存在
        user = get_or_create_user(db, current_user_id)
        
        # 批量获取用户数据（优化：一次性查询）
        user_data = get_user_data_batch(db, current_user_id)
        last_conversation = user_data["last_conversation"]
        last_conversation_time = user_data["last_conversation_time"]
        user_profile = user_data["user_profile"]
        recent_topics = user_data["recent_topics"]
        
        # 第二步：获取最近的对话历史用于综合分析（先获取，用于判断是否应该问候）
        conversation_history = get_recent_conversation_history(db, current_user_id, limit=5)
        
        # 第一步：判断是否需要生成问候（仅在开启新对话时）
        greeting = ""
        # 检查最近5分钟内是否有对话（判断是否是连续对话）
        recent_conversations_count = len(conversation_history)
        now = get_beijing_datetime()
        very_recent_conv_count = 0
        if last_conversation_time:
            for conv in conversation_history[:3]:  # 检查最近3条
                if conv.get("timestamp"):
                    try:
                        conv_time = datetime.fromisoformat(conv["timestamp"].replace('Z', '+00:00'))
                        delta = (now - conv_time.replace(tzinfo=None)).total_seconds()
                        if delta < 300:  # 5分钟内
                            very_recent_conv_count += 1
                    except:
                        pass
        
        # 如果只有图片没有文字，使用默认提示（提前定义，供后续使用）
        effective_user_input = user_input.strip() if user_input.strip() else (
            "请分析这张图片" if image_part else ""
        )
        
        # 如果最近5分钟内有对话，说明是连续对话，不生成问候
        # 问候判断使用原始user_input（主要基于时间和关键词）
        should_greet = should_generate_greeting(user_input, last_conversation_time, recent_conversations_count, very_recent_conv_count)
        
        if should_greet:  # 函数内部已经检查了very_recent_conv_count
            conversation_history_for_greeting = conversation_history[:3] if conversation_history else []
            greeting = generate_time_aware_greeting(
                last_conversation_time, 
                recent_topics,
                user_profile,
                conversation_history_for_greeting
            )
        
        # 第三步：快速路径检查（明显情绪可跳过完整分析）
        quick_emotion = quick_emotion_check(effective_user_input)
        
        # 第四步：异步并行执行多个操作以提高速度（使用asyncio替代ThreadPoolExecutor）
        from performance_monitor import task_timer
        from error_handling import (
            create_emotion_analysis_fallback, 
            create_role_analysis_fallback,
            FallbackManager
        )
        
        # 创建降级管理器
        emotion_fallback = FallbackManager("emotion_analysis")
        for strategy in create_emotion_analysis_fallback():
            emotion_fallback.add_strategy(strategy)
        
        role_fallback = FallbackManager("role_analysis")
        for strategy in create_role_analysis_fallback():
            role_fallback.add_strategy(strategy)
        
        # 并行执行任务（使用asyncio.gather）
        # ⚠️ 重要：确保情绪分析在角色分析之前完成
        tasks = []
        
        # 第一步：先执行情绪分析任务（必须完成）
        if quick_emotion:
            emotion_analysis = quick_emotion
            emotion_task = None
        else:
            async def emotion_task_wrapper():
                with task_timer("emotion_analysis"):
                    # 在线程池中运行同步函数
                    loop = asyncio.get_event_loop()
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda: analyze_emotion_with_history(
                                    effective_user_input, current_time, conversation_history
                                )
                            ),
                            timeout=3.0
                        )
                        return result
                    except Exception as e:
                        # 使用降级策略
                        logger.warning(f"情绪分析失败: {e}，使用降级策略")
                        # 修复：lambda 函数接受参数，即使不使用也要兼容参数传递
                        return await emotion_fallback.execute_with_fallback(
                            lambda *args, **kwargs: quick_emotion_check(effective_user_input) or {
                                "emotion_type": "neutral",
                                "emotion_intensity": "中等"
                            },
                            (effective_user_input,)  # 传递参数给降级策略
                        )
            emotion_task = asyncio.create_task(emotion_task_wrapper())
            tasks.append(("emotion", emotion_task))
        
        # 第二步：等待情绪分析完成，然后创建角色分析任务（依赖情绪分析结果）
        async def role_task_wrapper(emotion_result: dict):
            """角色分析任务包装器，接收情绪分析结果作为参数"""
            with task_timer("role_analysis"):
                loop = asyncio.get_event_loop()
                try:
                    # 使用已完成的情绪分析结果
                    if OPTIMIZED_ROLE_ANALYSIS_AVAILABLE:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda: analyze_role_preference_optimized(
                                    effective_user_input, current_time, 
                                    emotion_result,  # 使用已完成的情绪分析结果
                                    conversation_history, current_user_id
                                )
                            ),
                            timeout=3.0
                        )
                    else:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda: analyze_role_preference_with_history(
                                    effective_user_input, current_time, conversation_history
                                )
                            ),
                            timeout=3.0
                        )
                    return result
                except Exception as e:
                    # 使用降级策略
                    logger.warning(f"角色分析失败: {e}，使用降级策略")
                    # 直接返回默认值，不使用fallback（因为fallback期望参数）
                    return {
                        "role_type": "朋友",
                        "communication_style": "温和关怀",
                        "tone": "随和"
                    }
        
        # 如果情绪分析任务存在，先等待它完成
        if emotion_task:
            try:
                emotion_analysis = await asyncio.wait_for(emotion_task, timeout=3.5)
                # 情绪分析完成后，创建角色分析任务
                role_task = asyncio.create_task(role_task_wrapper(emotion_analysis))
                tasks.append(("role", role_task))
            except asyncio.TimeoutError:
                logger.warning("情绪分析超时，使用默认值")
                emotion_analysis = quick_emotion or {
                    "emotion_type": "neutral",
                    "emotion_intensity": "中等"
                }
                # 使用默认情绪分析结果创建角色分析任务
                role_task = asyncio.create_task(role_task_wrapper(emotion_analysis))
                tasks.append(("role", role_task))
        else:
            # 如果使用快速情绪检查，直接创建角色分析任务
            role_task = asyncio.create_task(role_task_wrapper(emotion_analysis))
            tasks.append(("role", role_task))
        
        # 上下文获取任务（需要独立的数据库会话）
        async def context_task_wrapper():
            with task_timer("context_retrieval"):
                # 使用上下文管理器确保数据库会话正确关闭
                db_context = next(get_db())
                try:
                    # 在线程池中运行同步函数
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: memory_manager.get_user_context(db_context, effective_user_input)
                    )
                    return result
                except Exception as e:
                    logger.warning(f"上下文检索失败: {e}")
                    return ""
                finally:
                    # 确保数据库会话关闭
                    try:
                        db_context.close()
                    except Exception as e:
                        logger.warning(f"关闭数据库会话失败: {e}")
        context_task = asyncio.create_task(context_task_wrapper())
        tasks.append(("context", context_task))
        
        # 等待所有任务完成（带超时和错误处理）
        try:
            # 计算剩余超时时间（情绪分析已完成，给角色分析和上下文更多时间）
            remaining_timeout = 4.0  # 剩余任务4秒超时
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=remaining_timeout
            )
            
            # 处理结果
            task_results = {}
            for i, (name, task) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.warning(f"任务 {name} 执行失败: {result}")
                    task_results[name] = None
                    # 取消任务（如果还在运行）
                    if not task.done():
                        task.cancel()
                else:
                    task_results[name] = result
            
            # 获取情绪分析结果
            # 注意：如果emotion_task存在，已经在前面等待完成并赋值了
            if not emotion_task:
                # 如果使用快速情绪检查，确保emotion_analysis已设置
                if 'emotion_analysis' not in locals():
                    emotion_analysis = quick_emotion or {
                        "emotion_type": "neutral", 
                        "emotion_intensity": "中等"
                    }
            
            # 获取角色分析结果
            role_preference = task_results.get("role") or {
                "role_type": "朋友", 
                "communication_style": "温和关怀", 
                "tone": "随和"
            }
            
            # 获取上下文结果
            user_context = task_results.get("context") or ""
            
        except asyncio.TimeoutError:
            logger.warning("并行任务执行超时，取消未完成的任务并使用默认值")
            # 取消所有未完成的任务
            for name, task in tasks:
                if not task.done():
                    task.cancel()
            
            # 等待取消完成（不抛出异常）
            await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # 使用默认值
            if not emotion_task or not quick_emotion:
                emotion_analysis = quick_emotion or {
                    "emotion_type": "neutral", 
                    "emotion_intensity": "中等"
                }
            role_preference = {
                "role_type": "朋友", 
                "communication_style": "温和关怀", 
                "tone": "随和"
            }
            user_context = ""
        
        # 第五步：生成AI回复（一次性生成，不流式，支持多模态）
        # 使用有效输入（如果有图片但没有文字，使用默认提示）
        response_prompt = build_response_prompt(
            effective_user_input, 
            current_time, 
            emotion_analysis, 
            role_preference, 
            user_context, 
            greeting
        )
        
        # 构建内容（支持文本+图片多模态）
        # 根据Gemini文档：https://ai.google.dev/gemini-api/docs/image-understanding
        # 如果将单个图片与文本搭配使用，请在 contents 数组中将文本提示放在图片部分之后
        if image_part:
            # 多模态：图片 + 文本提示
            contents = [
                image_part,
                response_prompt
            ]
        else:
            # 仅文本
            contents = response_prompt
        
        # 直接生成完整回复
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        
        ai_response_text = response.text if hasattr(response, 'text') else str(response)
        
        # 异步保存对话记录（不阻塞响应）
        # 保存时使用原始user_input（如果为空，记录为图片描述）
        def save_conversation():
            try:
                db_save = next(get_db())
                save_input = user_input.strip() if user_input.strip() else (
                    "[图片]" if image_part else ""
                )
                memory_manager.save_conversation(
                    db_save, save_input, ai_response_text, emotion_analysis, role_preference
                )
                db_save.close()
                
                # 清除相关缓存（因为数据已更新）
                try:
                    clear_user_caches(current_user_id)
                    # 清除对话历史缓存
                    from cache_manager import generate_cache_key, _conversation_cache
                    for limit in [3, 5, 10, 50]:
                        try:
                            cache_key = generate_cache_key("conv_history", current_user_id, limit=limit)
                            _conversation_cache.remove(cache_key)
                        except:
                            pass  # 缓存键可能不存在，忽略
                except:
                    pass  # 清除缓存失败不影响主流程
            except Exception as e:
                print(f"保存对话记录失败: {e}")
        
        threading.Thread(target=save_conversation, daemon=True).start()
        
        # 返回一次性响应（包含是否有图片的标记）
        return JSONResponse({
            "ai_response": ai_response_text,
            "emotion_analysis": emotion_analysis,
            "role_preference": role_preference,
            "timestamp": current_time,
            "has_image": image_part is not None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)})

@app.get("/api/monitor/user/{user_id}")
@app.get("/monitor/user/{user_id}")  # 前端直接请求的路由
async def get_user_monitor_data(user_id: str):
    """获取用户监测数据（支持按user_id或name查询）"""
    try:
        db = next(get_db())
        
        user = get_user_by_id_or_name(db, user_id)
        if not user:
            db.close()
            return JSONResponse({"error": f"用户不存在（未找到ID或名称为 '{user_id}' 的用户）"})
        actual_user_id = user.user_id
        
        # 获取用户画像
        profile = db.query(UserProfile).filter(UserProfile.user_id == actual_user_id).first()
        
        # 获取最近的情绪分析统计
        recent_conversations = db.query(Conversation).filter(
            Conversation.user_id == actual_user_id
        ).order_by(Conversation.timestamp.desc()).limit(50).all()
        
        # 统计情绪分布
        emotion_stats = {}
        role_stats = {}
        style_stats = {}
        
        for conv in recent_conversations:
            # 情绪统计
            emotion_data = safe_json_parse(conv.emotion_analysis, {})
            emotion_type = emotion_data.get("emotion_type", "neutral")
            emotion_stats[emotion_type] = emotion_stats.get(emotion_type, 0) + 1
            
            # 角色统计
            role_data = safe_json_parse(conv.role_preference, {})
            role_type = role_data.get("role_type", "朋友")
            role_stats[role_type] = role_stats.get(role_type, 0) + 1
            style_stats[role_data.get("communication_style", "温和关怀")] = style_stats.get(role_data.get("communication_style", "温和关怀"), 0) + 1
        
        # 获取重要记忆（实时，按创建时间倒序，确保最新记忆优先显示）
        memories = db.query(Memory).filter(
            Memory.user_id == actual_user_id
        ).order_by(Memory.created_at.desc(), Memory.relevance_score.desc()).limit(20).all()  # 增加到20条，确保实时性
        
        # 构建用户画像数据
        profile_data = {}
        if profile:
            profile_data = {
                "interests": safe_json_parse(profile.interests, []),
                "personality_traits": safe_json_parse(profile.personality_traits, []),
                "communication_preferences": safe_json_parse(profile.communication_preferences, {}),
                "emotional_patterns": safe_json_parse(profile.emotional_patterns, {}),
                "topics_history": safe_json_parse(profile.topics_history, []),
                "important_events": safe_json_parse(profile.important_events, []),
                "relationships_info": safe_json_parse(profile.relationships_info, {}),
                "updated_at": profile.updated_at.isoformat() if profile.updated_at else None
            }
        
        # 获取最近对话摘要
        recent_convs_summary = []
        for conv in recent_conversations[:10]:
            recent_convs_summary.append({
                "timestamp": conv.timestamp.isoformat() if conv.timestamp else None,
                "user_input": conv.user_input[:50] + "..." if len(conv.user_input) > 50 else conv.user_input,
                "emotion": safe_json_parse(conv.emotion_analysis, {}),
                "role": safe_json_parse(conv.role_preference, {})
            })
        
        # 记忆摘要
        memories_summary = []
        for mem in memories:
            memories_summary.append({
                "memory_id": mem.memory_id,
                "memory_type": mem.memory_type,
                "summary": mem.summary,
                "content": mem.content[:100] + "..." if len(mem.content) > 100 else mem.content,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "access_count": mem.access_count,
                "relevance_score": mem.relevance_score
            })
        
        db.close()
        
        return JSONResponse({
            "user_id": actual_user_id,
            "user_info": {
                "name": user.name,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_active": user.last_active.isoformat() if user.last_active else None,
                "total_conversations": user.total_conversations
            },
            "user_profile": profile_data,
            "statistics": {
                "emotion_distribution": emotion_stats,
                "role_preference_distribution": role_stats,
                "communication_style_distribution": style_stats,
                "total_conversations": len(recent_conversations)
            },
            "recent_conversations": recent_convs_summary,
            "important_memories": memories_summary
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)})

@app.get("/api/monitor/real-time/{user_id}")
async def get_real_time_observation(user_id: str):
    """获取实时观测数据（最近一次对话的分析结果，包含最新记忆）
    优化：使用缓存减少数据库查询，提高响应速度
    """
    try:
        db = next(get_db())
        
        try:
            user = get_user_by_id_or_name(db, user_id)
            if not user:
                return JSONResponse({"error": f"用户不存在"})
            actual_user_id = user.user_id
            
            # 获取最近一次对话（使用索引优化查询）
            last_conversation = db.query(Conversation).filter(
                Conversation.user_id == actual_user_id
            ).order_by(Conversation.timestamp.desc()).limit(1).first()
            
            if not last_conversation:
                return JSONResponse({"message": "暂无对话记录"})
            
            emotion_analysis = safe_json_parse(last_conversation.emotion_analysis, {})
            role_preference = safe_json_parse(last_conversation.role_preference, {})
            
            # 获取当前使用的上下文（记忆）- 实时获取最新记忆
            memory_manager = get_memory_manager(actual_user_id)
            user_context = memory_manager.get_user_context(db, last_conversation.user_input)
            
            # 获取最新记忆（实时，按创建时间倒序，限制查询数量提高性能）
            latest_memories = db.query(Memory).filter(
                Memory.user_id == actual_user_id
            ).order_by(Memory.created_at.desc()).limit(5).all()
            
            latest_memories_data = []
            for mem in latest_memories:
                latest_memories_data.append({
                    "memory_id": mem.memory_id,
                    "memory_type": mem.memory_type,
                    "summary": mem.summary,
                    "content": mem.content[:80] + "..." if len(mem.content) > 80 else mem.content,
                    "created_at": mem.created_at.isoformat() if mem.created_at else None,
                    "relevance_score": mem.relevance_score,
                    "access_count": mem.access_count
                })
            
            # 添加服务器时间戳，便于前端判断数据新鲜度
            from datetime import datetime
            server_timestamp = datetime.now().isoformat()
            
            return JSONResponse({
                "timestamp": last_conversation.timestamp.isoformat() if last_conversation.timestamp else None,
                "user_input": last_conversation.user_input,
                "server_time": server_timestamp,  # 服务器时间戳
                "current_observation": {
                    "emotion_analysis": emotion_analysis,
                    "role_preference": role_preference,
                    "user_context": user_context,
                    "importance_score": last_conversation.importance_score,
                    "latest_memories": latest_memories_data  # 添加最新记忆
                }
            })
        finally:
            db.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"获取实时观测数据失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ==================== 用户管理API ====================

# 兼容旧版本路由（重定向到v1）
@app.get("/api/users")
@app.get("/users")  # 前端直接请求的路由
@api_v1.get("/users")
async def get_users():
    """获取所有用户列表"""
    try:
        db = next(get_db())
        users = db.query(User).order_by(User.last_active.desc()).all()
        
        users_list = []
        for user in users:
            users_list.append({
                "user_id": user.user_id,
                "name": user.name,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_active": user.last_active.isoformat() if user.last_active else None,
                "total_conversations": user.total_conversations
            })
        
        db.close()
        return JSONResponse({"users": users_list})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/users")
@app.post("/users")  # 前端直接请求的路由
@api_v1.post("/users")
async def create_user(request: Request):
    """创建新用户"""
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        
        if not name:
            return JSONResponse({"error": "名称不能为空"}, status_code=400)
        
        if len(name) > 20:
            return JSONResponse({"error": "名称不能超过20个字符"}, status_code=400)
        
        db = next(get_db())
        
        # 生成用户ID（基于时间戳和名称的哈希）
        import hashlib
        import uuid
        user_id = f"user_{hashlib.md5((name + str(uuid.uuid4())).encode()).hexdigest()[:12]}"
        
        # 检查用户ID是否已存在（极小概率）
        existing = db.query(User).filter(User.user_id == user_id).first()
        if existing:
            user_id = f"user_{hashlib.md5((name + str(uuid.uuid4())).encode()).hexdigest()[:12]}"
        
        # 创建用户
        user = User(
            user_id=user_id,
            name=name,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        db.add(user)
        
        # 创建初始用户画像
        profile = UserProfile(user_id=user_id)
        db.add(profile)
        
        db.commit()
        db.refresh(user)
        
        db.close()
        
        return JSONResponse({
            "user_id": user.user_id,
            "name": user.name,
            "message": "用户创建成功"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/users/{user_id}")
async def get_user_info(user_id: str):
    """获取用户信息"""
    try:
        db = next(get_db())
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            db.close()
            return JSONResponse({"error": "用户不存在"}, status_code=404)
        
        user_info = {
            "user_id": user.user_id,
            "name": user.name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_active": user.last_active.isoformat() if user.last_active else None,
            "total_conversations": user.total_conversations
        }
        
        db.close()
        return JSONResponse(user_info)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.put("/api/users/{user_id}")
async def update_user_name(user_id: str, request: Request):
    """更新用户名称"""
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        
        if not name:
            return JSONResponse({"error": "名称不能为空"}, status_code=400)
        
        if len(name) > 20:
            return JSONResponse({"error": "名称不能超过20个字符"}, status_code=400)
        
        db = next(get_db())
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            db.close()
            return JSONResponse({"error": "用户不存在"}, status_code=404)
        
        user.name = name
        db.commit()
        db.close()
        
        return JSONResponse({
            "user_id": user.user_id,
            "name": user.name,
            "message": "名称更新成功"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@api_v1.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计信息"""
    try:
        if ADVANCED_CACHE_AVAILABLE:
            cache_manager = get_cache_manager()
            stats = cache_manager.get_all_stats()
            return JSONResponse({
                "success": True,
                "stats": stats,
                "cache_system": "advanced"
            })
        else:
            # 降级：返回传统缓存信息
            from cache_manager import (
                _emotion_cache, _role_cache, _user_data_cache, _conversation_cache
            )
            return JSONResponse({
                "success": True,
                "stats": {
                    "emotion": {"size": _emotion_cache.size()},
                    "role": {"size": _role_cache.size()},
                    "user_data": {"size": _user_data_cache.size()},
                    "conversation": {"size": _conversation_cache.size()}
                },
                "cache_system": "legacy"
            })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@api_v1.post("/cache/invalidate/user/{user_id}")
async def invalidate_user_cache(user_id: str):
    """失效指定用户的所有缓存"""
    try:
        if ADVANCED_CACHE_AVAILABLE:
            cache_manager = get_cache_manager()
            cache_manager.invalidate_user_data(user_id)
        
        # 也清除传统缓存
        clear_user_caches(user_id)
        
        return JSONResponse({
            "success": True,
            "message": f"用户 {user_id} 的缓存已失效"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@api_v1.post("/cache/save")
async def save_cache_to_disk():
    """手动保存缓存到磁盘"""
    try:
        if ADVANCED_CACHE_AVAILABLE:
            cache_manager = get_cache_manager()
            cache_manager.save_all_caches()
            return JSONResponse({
                "success": True,
                "message": "缓存已保存到磁盘"
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "高级缓存系统未启用"
            })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# 注册v1 API路由（必须在所有路由定义之后）
app.include_router(api_v1)





