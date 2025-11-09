"""
记忆管理系统
负责记忆提取、存储、检索和更新
使用Qdrant向量数据库 + LangGraph构建长期记忆管理流程
优化：HNSW索引、向量缓存、记忆过期机制、NER提取
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, TypedDict
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from database import (
    Conversation, Memory, UserProfile, User,
    get_or_create_user, get_db
)
import json
import re
import hashlib
import os

# Qdrant 导入
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠ Qdrant未安装，将使用传统记忆系统")

# Redis已移除，使用SQLite和内存缓存
REDIS_AVAILABLE = False

# LangGraph 导入
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠ LangGraph未安装，将使用传统记忆系统")

# 初始化 Qdrant 客户端（支持本地和远程模式）
if QDRANT_AVAILABLE:
    try:
        from config import settings
        
        # 优先使用远程模式（如果配置了HOST）
        if settings.QDRANT_HOST:
            qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT or 6333
            )
            print(f"[OK] Qdrant客户端初始化成功 (远程模式: {settings.QDRANT_HOST}:{settings.QDRANT_PORT or 6333})")
        else:
            # 本地模式
            qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
            print(f"[OK] Qdrant客户端初始化成功 (本地模式: {settings.QDRANT_PATH})")
    except Exception as e:
        print(f"⚠ Qdrant初始化失败: {e}")
        qdrant_client = None
else:
    qdrant_client = None

# SQLite缓存（使用内存缓存）
_vector_cache = {}  # 简单的内存缓存

# 工作流实例缓存（按用户ID）
_memory_workflows = {}  # LangGraphMemoryWorkflow 实例缓存
_memory_managers = {}  # MemoryManager 实例缓存

# 初始化向量模型（延迟加载）- 使用中文模型
_embedding_model = None
_embedding_dimension = 768  # 默认维度

def get_embedding_model():
    """获取向量化模型（延迟加载）- 使用中文模型
    优化：使用本地缓存目录，避免每次重启都重新下载
    """
    global _embedding_model, _embedding_dimension
    if _embedding_model is None:
        try:
            # 优化1：使用公共工具模块设置缓存
            from model_utils import get_model_cache_dir
            cache_dir = get_model_cache_dir()
            
            # 使用中文Sentence-BERT模型
            # 如果安装失败，回退到多语言模型
            try:
                _embedding_model = SentenceTransformer(
                    'renmada/sentence_bert_chinese',
                    cache_folder=cache_dir  # 指定缓存目录
                )
                _embedding_dimension = 768
                print(f"[OK] 中文向量化模型加载成功 (缓存目录: {cache_dir})")
            except Exception as e:
                print(f"中文模型加载失败: {e}，使用多语言模型")
                _embedding_model = SentenceTransformer(
                    'paraphrase-multilingual-mpnet-base-v2',
                    cache_folder=cache_dir  # 指定缓存目录
                )
                _embedding_dimension = 768
                print(f"[OK] 多语言向量化模型加载成功 (缓存目录: {cache_dir})")
        except Exception as e:
            print(f"加载向量化模型失败: {e}")
            _embedding_model = None
    return _embedding_model

def get_vector_cache_key(query: str) -> str:
    """生成向量缓存键（使用新的结构化键生成器）"""
    try:
        from advanced_cache import CacheKeyBuilder, CacheType
        return CacheKeyBuilder.build_vector_key(query)
    except ImportError:
        # 降级到旧方法
        return hashlib.md5(f"vec_cache:{query}".encode('utf-8')).hexdigest()

def get_cached_vector(query: str) -> Optional[List[float]]:
    """从缓存获取向量（使用新的统一缓存管理器）"""
    try:
        from advanced_cache import get_cache_manager
        cache_manager = get_cache_manager(None)
        return cache_manager.get_vector(query)
    except Exception as e:
        print(f"⚠ 获取向量缓存失败: {e}，使用旧缓存")
        # 降级到内存缓存
        cache_key = get_vector_cache_key(query)
        
        if cache_key in _vector_cache:
            cached_data = _vector_cache[cache_key]
            # 检查是否过期（24小时）
            if datetime.now() - cached_data['timestamp'] < timedelta(hours=24):
                return cached_data['vector']
            else:
                del _vector_cache[cache_key]
        
        return None

def set_cached_vector(query: str, vector: List[float]):
    """缓存向量（使用新的统一缓存管理器）"""
    try:
        from advanced_cache import get_cache_manager, CachePriority
        cache_manager = get_cache_manager(None)
        cache_manager.set_vector(query, vector, priority=CachePriority.HIGH)
    except Exception as e:
        print(f"⚠ 设置向量缓存失败: {e}，使用内存缓存")
        # 降级到内存缓存
        cache_key = get_vector_cache_key(query)
        
        _vector_cache[cache_key] = {
            'vector': vector,
            'timestamp': datetime.now()
        }
        
        # 限制内存缓存大小（最多1000条）
        if len(_vector_cache) > 1000:
            # 删除最旧的条目
            oldest_key = min(_vector_cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del _vector_cache[oldest_key]

# LangGraph 记忆状态定义
class MemoryState(TypedDict):
    """记忆管理状态"""
    user_id: str
    user_input: str
    ai_response: str
    emotion_analysis: dict
    role_preference: dict
    memory_info: Optional[dict]
    memories: List[dict]
    conversation_id: str
    should_extract: bool  # 是否应该提取记忆

# LangGraph 工作流构建器（优化版：使用Router节点）
class LangGraphMemoryWorkflow:
    """基于LangGraph的记忆管理工作流（优化版）"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph = None
        self.checkpointer = None
        self._error_count = 0  # 错误计数，用于错误处理
        if LANGGRAPH_AVAILABLE:
            self._build_workflow()
    
    def _build_workflow(self):
        """构建LangGraph记忆管理工作流（使用Router节点）"""
        try:
            # 创建检查点保存器
            self.checkpointer = MemorySaver()
            
            # 创建状态图
            workflow = StateGraph(MemoryState)
            
            # 添加节点
            workflow.add_node("router", self._router_node)  # 决策节点
            workflow.add_node("extract_memory", self._extract_memory_node)
            workflow.add_node("save_memory", self._save_memory_node)
            workflow.add_node("update_profile", self._update_profile_node)
            # 注意：retrieve_memories节点已定义但未使用，保留以备将来使用
            # workflow.add_node("retrieve_memories", self._retrieve_memories_node)
            
            # 定义流程：从router开始
            workflow.set_entry_point("router")
            
            # Router的决策边
            workflow.add_conditional_edges(
                "router",
                self._router_decision,
                {
                    "extract": "extract_memory",
                    "skip": END
                }
            )
            
            # 提取记忆后保存
            workflow.add_edge("extract_memory", "save_memory")
            
            # 保存后判断是否更新画像
            workflow.add_conditional_edges(
                "save_memory",
                self._should_update_profile,
                {
                    "yes": "update_profile",
                    "no": END
                }
            )
            
            workflow.add_edge("update_profile", END)
            
            # 编译图
            self.graph = workflow.compile(checkpointer=self.checkpointer)
            print("[OK] LangGraph记忆管理工作流构建成功（优化版）")
        except Exception as e:
            print(f"⚠ 构建LangGraph工作流失败: {e}，将使用传统方法")
            self.graph = None
    
    def _router_node(self, state: MemoryState) -> MemoryState:
        """Router节点：判断是否需要提取记忆"""
        user_input = state.get("user_input", "")
        emotion_analysis = state.get("emotion_analysis", {})
        
        # 快速过滤逻辑
        should_extract = False
        
        if not user_input or len(user_input.strip()) < 3:
            should_extract = False
        else:
            # 高情绪强度时更可能包含重要信息
            emotion_intensity = emotion_analysis.get("emotion_intensity", "中等")
            if emotion_intensity == "强烈":
                should_extract = True
            else:
                # 关键词匹配
                important_keywords = [
                    "要去", "准备去", "计划", "行程", "出差", "旅行", "旅游", "开会", 
                    "见面", "约会", "约了", "预约", "安排", "明天", "后天", "下周",
                    "喜欢", "讨厌", "习惯", "偏好", "习惯", "经常", "总是", "从不",
                    "生日", "纪念日", "重要", "难忘", "记得"
                ]
                should_extract = any(kw in user_input for kw in important_keywords)
        
        state["should_extract"] = should_extract
        return state
    
    def _router_decision(self, state: MemoryState) -> str:
        """Router决策函数"""
        if state.get("should_extract", False):
            return "extract"
        return "skip"
    
    def _extract_memory_node(self, state: MemoryState) -> MemoryState:
        """提取记忆节点（使用事件驱动接口，避免循环依赖，带错误处理）"""
        from memory_service import get_memory_service
        
        user_input = state.get("user_input", "")
        emotion_analysis = state.get("emotion_analysis", {})
        role_preference = state.get("role_preference", {})
        
        try:
            # 使用MemoryService接口（事件驱动）
            memory_service = get_memory_service(self.user_id)
            conversation_data = {
                "user_input": user_input,
                "ai_response": state.get("ai_response", ""),
                "emotion_analysis": emotion_analysis,
                "role_preference": role_preference
            }
            
            memory_info = memory_service.extract_memory(conversation_data, emotion_analysis, role_preference)
            state["memory_info"] = memory_info
            state["error"] = None  # 清除之前的错误
            
        except Exception as e:
            import traceback
            error_msg = f"提取记忆失败: {e}"
            print(error_msg)
            traceback.print_exc()
            
            # 记录错误，但不中断工作流（允许继续保存对话）
            state["error"] = error_msg
            state["failed_node"] = "extract_memory"
            state["memory_info"] = None  # 标记为未提取记忆
            self._error_count += 1
            
            # 如果错误次数过多，标记工作流失败
            if self._error_count >= 3:
                state["workflow_failed"] = True
        
        return state
    
    def _save_memory_node(self, state: MemoryState) -> MemoryState:
        """保存记忆节点（使用事件驱动接口，带错误处理和回滚）"""
        from memory_service import get_memory_service
        
        memory_info = state.get("memory_info")
        if not memory_info:
            return state
        
        db = None
        try:
            db = next(get_db())
            memory_service = get_memory_service(self.user_id)
            conversation_id = state.get("conversation_id", str(uuid.uuid4()))
            
            memory_service.save_memory(
                db,
                conversation_id,
                memory_info,
                state.get("user_input", "")
            )
            
            state["memory_info"] = memory_info
            state["error"] = None  # 清除之前的错误
            self._error_count = 0  # 重置错误计数
            
        except Exception as e:
            import traceback
            error_msg = f"保存记忆失败: {e}"
            print(error_msg)
            traceback.print_exc()
            
            # 记录错误
            state["error"] = error_msg
            state["failed_node"] = "save_memory"
            self._error_count += 1
            
            # 如果错误次数过多，标记工作流失败
            if self._error_count >= 3:
                state["workflow_failed"] = True
                raise RuntimeError(f"工作流失败：连续{self._error_count}次错误") from e
            
        finally:
            if db:
                try:
                    db.close()
                except Exception as e:
                    print(f"关闭数据库会话失败: {e}")
        
        return state
    
    def _should_update_profile(self, state: MemoryState) -> str:
        """判断是否应该更新用户画像"""
        memory_info = state.get("memory_info")
        if memory_info and memory_info.get("importance_score", 0) > 0.5:
            return "yes"
        return "no"
    
    def _update_profile_node(self, state: MemoryState) -> MemoryState:
        """更新用户画像节点（使用事件驱动接口，带错误处理）"""
        from memory_service import get_memory_service
        
        db = None
        try:
            db = next(get_db())
            memory_service = get_memory_service(self.user_id)
            memory_service.update_user_profile(
                db,
                state.get("emotion_analysis", {}),
                state.get("role_preference", {}),
                state.get("user_input", "")
            )
            state["error"] = None  # 清除之前的错误
            
        except Exception as e:
            import traceback
            error_msg = f"更新用户画像失败: {e}"
            print(error_msg)
            traceback.print_exc()
            
            # 记录错误，但不中断工作流（画像更新失败不影响对话保存）
            state["error"] = error_msg
            state["failed_node"] = "update_profile"
            self._error_count += 1
            
        finally:
            if db:
                try:
                    db.close()
                except Exception as e:
                    print(f"关闭数据库会话失败: {e}")
        
        return state
    
    def _retrieve_memories_node(self, state: MemoryState) -> MemoryState:
        """检索记忆节点（使用事件驱动接口）"""
        from memory_service import get_memory_service
        
        query = state.get("user_input", "")
        memory_service = get_memory_service(self.user_id)
        memories = memory_service.retrieve_memories(query, top_k=5)
        state["memories"] = memories
        return state
    
    def process_conversation(self, user_input: str, ai_response: str, 
                           emotion_analysis: dict, role_preference: dict, 
                           memory_manager_instance=None) -> dict:
        """
        处理对话并管理记忆
        
        Args:
            memory_manager_instance: MemoryManager实例（避免重复创建）
        """
        if not self.graph:
            # 降级到传统方法（使用传入的实例，避免重复创建）
            if memory_manager_instance:
                db = next(get_db())
                try:
                    conversation_id = memory_manager_instance.save_conversation(
                        db, user_input, ai_response, emotion_analysis, role_preference
                    )
                    return {"conversation_id": conversation_id}
                finally:
                    db.close()
            else:
                # 如果没有传入实例，使用缓存函数获取（向后兼容）
                db = next(get_db())
                try:
                    memory_manager = get_memory_manager(self.user_id)
                    conversation_id = memory_manager.save_conversation(
                        db, user_input, ai_response, emotion_analysis, role_preference
                    )
                    return {"conversation_id": conversation_id}
                finally:
                    db.close()
        
        # 使用LangGraph工作流
        try:
            config = {"configurable": {"thread_id": self.user_id}}
            conversation_id = str(uuid.uuid4())
            initial_state = {
                "user_id": self.user_id,
                "user_input": user_input,
                "ai_response": ai_response,
                "emotion_analysis": emotion_analysis,
                "role_preference": role_preference,
                "memory_info": None,
                "memories": [],
                "conversation_id": conversation_id,
                "should_extract": False
            }
            
            result = self.graph.invoke(initial_state, config=config)
            
            # ⚠️ 重要：工作流只保存记忆，需要单独保存对话记录
            if memory_manager_instance:
                db = next(get_db())
                try:
                    # 保存对话记录
                    from database import Conversation
                    import json
                    from datetime import datetime
                    
                    importance_score = 0.5
                    if result.get("memory_info"):
                        importance_score = result.get("memory_info").get("importance_score", 0.5)
                    
                    conversation = Conversation(
                        conversation_id=conversation_id,
                        user_id=self.user_id,
                        timestamp=datetime.now(),
                        user_input=user_input,
                        ai_response=ai_response,
                        emotion_analysis=json.dumps(emotion_analysis, ensure_ascii=False),
                        role_preference=json.dumps(role_preference, ensure_ascii=False),
                        importance_score=importance_score
                    )
                    db.add(conversation)
                    
                    # 更新用户对话计数
                    from database import get_or_create_user
                    user = get_or_create_user(db, self.user_id)
                    user.total_conversations += 1
                    db.commit()
                finally:
                    db.close()
            
            return {
                "conversation_id": conversation_id,
                "memory_info": result.get("memory_info")
            }
        except Exception as e:
            print(f"LangGraph工作流执行失败: {e}，使用传统方法")
            # 降级到传统方法（使用传入的实例，避免重复创建）
            if memory_manager_instance:
                db = next(get_db())
                try:
                    conversation_id = memory_manager_instance.save_conversation(
                        db, user_input, ai_response, emotion_analysis, role_preference
                    )
                    return {"conversation_id": conversation_id}
                finally:
                    db.close()
            else:
                # 如果没有传入实例，使用缓存函数获取（向后兼容）
                db = next(get_db())
                try:
                    memory_manager = get_memory_manager(self.user_id)
                    conversation_id = memory_manager.save_conversation(
                        db, user_input, ai_response, emotion_analysis, role_preference
                    )
                    return {"conversation_id": conversation_id}
                finally:
                    db.close()

def get_langgraph_memory_workflow(user_id: str) -> Optional['LangGraphMemoryWorkflow']:
    """获取或创建LangGraph记忆管理工作流（单例模式）"""
    if not LANGGRAPH_AVAILABLE:
        return None
    
    if user_id not in _memory_workflows:
        _memory_workflows[user_id] = LangGraphMemoryWorkflow(user_id)
        print(f"[OK] 创建LangGraph记忆管理工作流: {user_id}")
    return _memory_workflows[user_id]

def get_memory_manager(user_id: str) -> 'MemoryManager':
    """获取或创建MemoryManager实例（单例模式）"""
    if user_id not in _memory_managers:
        _memory_managers[user_id] = MemoryManager(user_id)
        print(f"[OK] 创建MemoryManager实例: {user_id}")
    return _memory_managers[user_id]

class MemoryManager:
    """记忆管理器（使用Qdrant + LangGraph优化）
    保持与现有系统的兼容性
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.collection_name = f"user_{user_id}_memories"
        self._init_collection()
        
        # 使用缓存的工作流实例
        self.workflow = get_langgraph_memory_workflow(user_id) if LANGGRAPH_AVAILABLE else None
    
    def _init_collection(self):
        """初始化Qdrant集合（支持多种向量维度，使用可配置的HNSW索引）"""
        if not qdrant_client:
            print("⚠ Qdrant不可用，跳过集合初始化")
            return
        
        try:
            from embedding_manager import MemoryType, get_embedding_dimension
            from qdrant_config import get_qdrant_config
            
            qdrant_config = get_qdrant_config()
            
            # 检查集合是否存在
            collections = qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                # 创建新集合，使用默认维度（文本记忆的维度）
                # 注意：Qdrant一个集合只能有一种向量维度
                # 如果需要多维度，需要创建多个集合或使用payload存储类型信息
                default_dimension = get_embedding_dimension(MemoryType.TEXT)
                
                # 获取向量参数和HNSW配置
                vectors_config = qdrant_config.get_vector_params(default_dimension)
                hnsw_config = qdrant_config.get_hnsw_config_obj("small")  # 新集合使用小规模配置
                optimizers_config = qdrant_config.get_optimizers_config_obj()
                
                qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    hnsw_config=hnsw_config,
                    optimizers_config=optimizers_config
                )
                
                distance_str = qdrant_config.get_distance().name
                print(f"[OK] Qdrant集合创建成功: {self.collection_name} "
                      f"(HNSW: m={hnsw_config.m}, ef_construct={hnsw_config.ef_construct}, "
                      f"{default_dimension}维, {distance_str})")
            else:
                # 集合已存在，检查是否需要动态调优
                try:
                    collection_info = qdrant_client.get_collection(self.collection_name)
                    vector_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                    
                    # 根据向量数量自动调优（如果启用）
                    if qdrant_config.config.get("auto_tune", True) and vector_count > 0:
                        profile = qdrant_config.auto_tune_profile(vector_count)
                        current_profile = self._get_current_profile(collection_info)
                        
                        if current_profile != profile:
                            print(f"🔄 检测到集合 {self.collection_name} 向量数量: {vector_count}，"
                                  f"建议切换到 {profile} profile")
                            # 可以选择性地更新配置
                            # qdrant_config.update_collection_config(self.collection_name, qdrant_client, profile, vector_count)
                except Exception as e:
                    print(f"⚠ 检查集合配置失败: {e}")
                
                print(f"[OK] Qdrant集合已存在: {self.collection_name}")
        except Exception as e:
            print(f"⚠ 初始化Qdrant集合失败: {e}")
    
    def _get_current_profile(self, collection_info) -> str:
        """根据集合信息推断当前使用的profile"""
        return _get_collection_profile(collection_info)
    
    def _quick_filter_memory(self, user_input: str, emotion_analysis: dict) -> bool:
        """快速过滤：判断是否需要提取记忆"""
        if not user_input or len(user_input.strip()) < 3:
            return False
        
        # 高情绪强度时更可能包含重要信息
        emotion_intensity = emotion_analysis.get("emotion_intensity", "中等")
        if emotion_intensity == "强烈":
            return True
        
        # 关键词匹配
        important_keywords = [
            "要去", "准备去", "计划", "行程", "出差", "旅行", "旅游", "开会", 
            "见面", "约会", "约了", "预约", "安排", "明天", "后天", "下周",
            "喜欢", "讨厌", "习惯", "偏好", "习惯", "经常", "总是", "从不",
            "生日", "纪念日", "重要", "难忘", "记得"
        ]
        
        return any(kw in user_input for kw in important_keywords)
    
    def _extract_entities_ner(self, user_input: str) -> Dict[str, str]:
        """使用NER提取关键实体（简化版：基于规则）"""
        entities = {}
        
        # 时间提取
        time_patterns = [
            (r"明天|后天", "时间"),
            (r"下周[一二三四五六日]|下周一|下周二|下周三|下周四|下周五|下周六|下周日", "时间"),
            (r"这个周末|这周末|下周末", "时间"),
            (r"下个月", "时间"),
            (r"\d+号|\d+日", "时间"),
        ]
        for pattern, entity_type in time_patterns:
            match = re.search(pattern, user_input)
            if match:
                entities["时间"] = match.group()
                break
        
        # 地点提取（简化）
        location_keywords = ["去", "到", "在", "去", "回"]
        for kw in location_keywords:
            idx = user_input.find(kw)
            if idx >= 0:
                # 提取关键字后的内容作为地点
                after = user_input[idx + len(kw):].strip()
                if len(after) > 0 and len(after) < 20:
                    entities["地点"] = after.split()[0]
                    break
        
        # 人物提取（简化）
        person_keywords = ["和", "跟", "与", "一起"]
        for kw in person_keywords:
            idx = user_input.find(kw)
            if idx >= 0:
                after = user_input[idx + len(kw):].strip()
                if len(after) > 0 and len(after) < 10:
                    entities["人物"] = after.split()[0]
                    break
        
        return entities
    
    def _calculate_importance_score(self, memory_info: dict, emotion_analysis: dict, 
                                    novelty_factor: float = 1.0, 
                                    access_count: int = 0,
                                    created_at: datetime = None) -> float:
        """
        计算记忆重要性评分（多因子评分）
        
        评分公式：
        importance = (base_score * emotion_weight * type_weight * novelty_factor * 
                     recency_factor * frequency_factor)
        
        因子说明：
        - base_score: LLM生成的基础重要性评分
        - emotion_weight: 情绪强度权重（强烈>中等>轻微）
        - type_weight: 记忆类型权重（schedule>event>preference>emotion）
        - novelty_factor: 新颖性因子（实体丰富度）
        - recency_factor: 时间衰减因子（越新越重要）
        - frequency_factor: 交互频率因子（访问次数越多越重要）
        """
        base_score = memory_info.get("importance_score", 0.5)
        emotion_intensity = emotion_analysis.get("emotion_intensity", "中等")
        
        # 1. 情绪权重
        emotion_weights = {
            "强烈": 1.3,
            "中等": 1.0,
            "轻微": 0.7
        }
        emotion_weight = emotion_weights.get(emotion_intensity, 1.0)
        
        # 2. 记忆类型权重
        type_weights = {
            "schedule": 1.2,
            "event": 1.0,
            "preference": 0.9,
            "emotion": 0.8
        }
        memory_type = memory_info.get("memory_type", "event")
        type_weight = type_weights.get(memory_type, 1.0)
        
        # 3. 时间衰减因子（Recency）
        # 新记忆（7天内）权重为1.0，随时间衰减
        if created_at:
            days_old = (datetime.now() - created_at).days
            if days_old <= 7:
                recency_factor = 1.0
            elif days_old <= 30:
                recency_factor = 0.9
            elif days_old <= 90:
                recency_factor = 0.7
            else:
                recency_factor = max(0.5, 1.0 - days_old / 180.0)  # 180天后最低0.5
        else:
            recency_factor = 1.0  # 新记忆，默认权重
        
        # 4. 交互频率因子（Frequency）
        # 访问次数越多，重要性越高（对数增长）
        if access_count > 0:
            frequency_factor = 1.0 + min(0.3, access_count * 0.05)  # 最多增加30%
        else:
            frequency_factor = 1.0
        
        # 计算最终重要性
        importance = (base_score * emotion_weight * type_weight * novelty_factor * 
                     recency_factor * frequency_factor)
        
        # 归一化到[0, 1]范围
        return min(1.0, max(0.0, importance))
    
    def _extract_memory_with_llm(self, user_input: str, emotion_analysis: dict, 
                                 role_preference: dict, current_time: str, time_period: str) -> Optional[Dict]:
        """使用LLM智能提取记忆（Gemini API + NER）"""
        try:
            from app import client
            
            emotion_type = emotion_analysis.get("emotion_type", "neutral")
            emotion_intensity = emotion_analysis.get("emotion_intensity", "中等")
            
            # 先使用NER提取实体
            key_entities = self._extract_entities_ner(user_input)
            
            # 构建提示词
            memory_prompt = f"""分析以下用户对话，提取重要记忆信息。

用户输入：{user_input}
当前时间：{current_time} ({time_period})
情绪状态：{emotion_type}（强度：{emotion_intensity}）
已提取实体：{json.dumps(key_entities, ensure_ascii=False) if key_entities else "无"}

请判断这是否是重要记忆，并提取以下信息：
1. 记忆类型：schedule（行程/计划）、preference（偏好/习惯）、event（重要事件）、emotion（情感记忆）
2. 记忆内容：简洁描述
3. 摘要：50字以内的摘要
4. 重要性评分：0.0-1.0（行程和重要事件通常0.8+，偏好0.6+，普通事件0.5-）
5. 关键实体：时间、地点、人物、事件等（补充和验证已提取的实体）

如果是行程类记忆，请特别标注is_schedule=true，并提取详细的时间、地点、人物信息。

只输出JSON格式：
{{
    "memory_type": "schedule/preference/event/emotion",
    "content": "记忆内容",
    "summary": "摘要",
    "importance_score": 0.8,
    "is_schedule": true/false,
    "key_entities": {{
        "时间": "如果有",
        "地点": "如果有",
        "人物": "如果有",
        "事件": "如果有"
    }}
}}

只输出JSON，不要其他文字说明。"""

            memory_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=memory_prompt
            )
            
            response_text = memory_response.text.strip()
            
            # 更健壮的JSON解析
            memory_data = None
            try:
                # 尝试直接解析
                memory_data = json.loads(response_text)
            except json.JSONDecodeError:
                # 尝试提取JSON部分
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        memory_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # 尝试修复常见的JSON错误
                        json_str = json_match.group()
                        # 移除注释和多余的逗号
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        try:
                            memory_data = json.loads(json_str)
                        except:
                            pass
            
            if not memory_data:
                print(f"LLM记忆提取失败: 无法解析JSON响应: {response_text[:200]}")
                return None
            
            # 验证和规范化数据
            valid_types = ["schedule", "preference", "event", "emotion"]
            memory_type = memory_data.get("memory_type", "event")
            if memory_type not in valid_types:
                memory_type = "event"
            
            # 合并NER提取的实体和LLM提取的实体
            llm_entities = memory_data.get("key_entities", {})
            key_entities.update(llm_entities)
            
            # 计算重要性评分（考虑情绪和新颖性）
            base_importance = float(memory_data.get("importance_score", 0.5))
            base_importance = max(0.0, min(1.0, base_importance))
            
            # 计算新颖性因子（简化：如果有很多实体，说明信息丰富）
            novelty_factor = 1.0 + len(key_entities) * 0.1
            
            memory_info = {
                "memory_type": memory_type,
                "content": user_input,
                "summary": memory_data.get("summary", user_input[:50] + "..."),
                "time_period": time_period,
                "timestamp": datetime.now().isoformat(),
                "importance_score": base_importance,
                "key_entities": key_entities,
                "is_schedule": memory_data.get("is_schedule", False) or memory_type == "schedule",
                "emotion_type": emotion_type,
                "emotion_intensity": emotion_intensity
            }
            
            # 计算最终重要性（多因子评分）
            memory_info["importance_score"] = self._calculate_importance_score(
                memory_info, emotion_analysis, novelty_factor,
                access_count=0,  # 新记忆，访问次数为0
                created_at=datetime.now()
            )
            
            return memory_info
            
        except Exception as e:
            print(f"LLM记忆提取失败: {e}")
            return None
    
    def extract_memory(self, conversation_data: dict, emotion_analysis: dict, role_preference: dict) -> Optional[Dict]:
        """
        智能提取重要记忆（混合方法：关键词快速过滤 + NER + LLM深度分析）
        提高准确性和效率
        """
        from app import get_time_period, get_beijing_datetime, get_beijing_time
        
        user_input = conversation_data.get("user_input", "")
        current_time = get_beijing_time()
        current_datetime = get_beijing_datetime()
        time_period = get_time_period(current_datetime)
        
        # 第一步：快速过滤（避免不必要的API调用）
        if not self._quick_filter_memory(user_input, emotion_analysis):
            return None
        
        # 第二步：使用LLM进行智能分析（更准确）
        memory_info = self._extract_memory_with_llm(
            user_input, emotion_analysis, role_preference, current_time, time_period
        )
        
        # 如果LLM分析失败，回退到关键词方法（保持兼容性）
        if memory_info is None:
            # 回退到关键词方法（简化版）
            schedule_keywords = ["要去", "准备去", "计划", "行程", "出差", "旅行", "旅游", "开会", 
                                "见面", "约会", "约了", "预约", "安排", "明天", "后天", "下周"]
            has_schedule = any(kw in user_input for kw in schedule_keywords)
            
            if has_schedule:
                return {
                    "memory_type": "schedule",
                    "content": f"[行程/{time_period}] {user_input}",
                    "summary": user_input[:50] + "..." if len(user_input) > 50 else user_input,
                    "time_period": time_period,
                    "timestamp": current_datetime.isoformat(),
                    "importance_score": 0.9,
                    "is_schedule": True,
                    "key_entities": self._extract_entities_ner(user_input)
                }
        
        return memory_info
    
    def save_conversation(self, db: Session, user_input: str, ai_response: str, 
                         emotion_analysis: dict, role_preference: dict):
        """保存对话记录"""
        # 如果LangGraph工作流可用，使用工作流（传入self避免重复创建）
        if self.workflow and self.workflow.graph:
            result = self.workflow.process_conversation(
                user_input, ai_response, emotion_analysis, role_preference,
                memory_manager_instance=self  # 传入当前实例
            )
            return result.get("conversation_id", str(uuid.uuid4()))
        
        # 传统方法（降级）
        conversation_id = str(uuid.uuid4())
        
        # 提取记忆
        conversation_data = {
            "user_input": user_input,
            "ai_response": ai_response,
            "emotion_analysis": emotion_analysis,
            "role_preference": role_preference
        }
        memory_info = self.extract_memory(conversation_data, emotion_analysis, role_preference)
        
        # 计算重要性评分
        importance_score = 0.5
        if memory_info:
            importance_score = memory_info.get("importance_score", 0.5)
        
        # 保存到SQLite
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=self.user_id,
            timestamp=datetime.now(),
            user_input=user_input,
            ai_response=ai_response,
            emotion_analysis=json.dumps(emotion_analysis, ensure_ascii=False),
            role_preference=json.dumps(role_preference, ensure_ascii=False),
            importance_score=importance_score
        )
        db.add(conversation)
        
        # 更新用户对话计数
        user = get_or_create_user(db, self.user_id)
        user.total_conversations += 1
        db.commit()
        
        # 如果是重要记忆，保存到记忆表和向量库（带去重检查）
        if memory_info:
            # 检查是否为重复记忆
            from memory_deduplicator import get_deduplicator
            from embedding_manager import get_memory_type, encode_text
            
            deduplicator = get_deduplicator(self.user_id)
            
            # 生成向量用于去重检查
            memory_type = get_memory_type(memory_info)
            enhanced_text = self._build_vector_text(
                user_input, memory_info, memory_info.get("summary", "")
            )
            vector = encode_text(enhanced_text, memory_type)
            
            # 检查重复
            duplicate_id = deduplicator.is_duplicate(enhanced_text, vector)
            
            if duplicate_id:
                # 更新现有记忆的访问次数和重要性评分
                existing_memory = db.query(Memory).filter(Memory.memory_id == duplicate_id).first()
                if existing_memory:
                    existing_memory.access_count += 1
                    existing_memory.last_accessed = datetime.now()
                    # 重新计算重要性评分（考虑新的访问）
                    existing_memory.relevance_score = self._calculate_importance_score(
                        memory_info, emotion_analysis, novelty_factor=1.0,
                        access_count=existing_memory.access_count,
                        created_at=existing_memory.created_at
                    )
                    db.commit()
                    print(f"[OK] 检测到重复记忆，已更新: {duplicate_id}")
                return conversation_id
            
            # 不是重复，保存新记忆
            self._save_memory(db, conversation_id, memory_info, user_input)
            
            # 添加到去重器
            deduplicator.add_signature(
                memory_info.get("memory_id", conversation_id),
                enhanced_text,
                vector
            )
        
        # 更新用户画像
        self._update_user_profile(db, emotion_analysis, role_preference, user_input)
        
        return conversation_id
    
    def _save_memory(self, db: Session, conversation_id: str, memory_info: dict, content: str):
        """保存重要记忆（使用批量操作和多种嵌入模型）"""
        memory_id = memory_info.get("memory_id", str(uuid.uuid4()))
        
        # 优先使用LLM生成的摘要
        summary = memory_info.get("summary")
        
        # 如果没有LLM摘要，根据记忆类型生成摘要
        if not summary or len(summary.strip()) == 0:
            if memory_info.get("is_schedule") or memory_info["memory_type"] == "schedule":
                summary = self._extract_schedule_summary(content)
                if not summary:
                    summary = content[:60] + "..." if len(content) > 60 else content
            else:
                summary = content[:50] + "..." if len(content) > 50 else content
        
        # 重新计算重要性评分（包含时间衰减和访问频率）
        final_importance = self._calculate_importance_score(
            memory_info, 
            {},  # emotion_analysis在保存时可能不可用，使用空字典
            novelty_factor=1.0,
            access_count=0,
            created_at=datetime.now()
        )
        
        # 保存到SQLite（关系型数据库）
        memory = Memory(
            memory_id=memory_id,
            user_id=self.user_id,
            created_at=datetime.now(),
            memory_type=memory_info["memory_type"],
            content=content,
            summary=summary,
            access_count=0,
            last_accessed=datetime.now(),
            relevance_score=final_importance  # 使用重新计算的重要性评分
        )
        db.add(memory)
        db.commit()
        
        # 向量化并存入Qdrant（使用批量操作和多种嵌入模型）
        from embedding_manager import get_memory_type, encode_text, get_embedding_dimension
        from batch_memory_manager import get_batch_buffer, PendingMemory
        
        enhanced_text = self._build_vector_text(content, memory_info, summary)
        
        # 根据记忆类型选择嵌入模型
        memory_type_enum = get_memory_type(memory_info)
        embedding = encode_text(enhanced_text, memory_type_enum)
        
        if embedding and qdrant_client:
            # 构建payload
            payload = {
                "memory_id": memory_id,
                "user_id": self.user_id,
                "memory_type": memory_info["memory_type"],
                "created_at": datetime.now().isoformat(),
                "summary": summary,
                "content": enhanced_text,
                "is_schedule": memory_info.get("is_schedule", False),
                "importance_score": final_importance,
                "decay_factor": 1.0,  # 初始衰减因子
                "timestamp": memory_info.get("timestamp", datetime.now().isoformat()),
                "access_count": 0,
                "last_accessed": datetime.now().isoformat()
            }
            
            # 使用批量缓冲区（异步批量写入）
            try:
                batch_buffer = get_batch_buffer(self.user_id)
                
                pending_memory = PendingMemory(
                    memory_id=memory_id,
                    vector=embedding,
                    payload=payload
                )
                
                # 异步添加到批量缓冲区
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(batch_buffer.add(pending_memory))
                except RuntimeError:
                    # 如果没有事件循环，使用同步方式
                    import threading
                    def async_add():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(batch_buffer.add(pending_memory))
                        new_loop.close()
                    threading.Thread(target=async_add, daemon=True).start()
                
                print(f"[OK] 记忆已添加到批量缓冲区: {memory_id}")
            except Exception as e:
                print(f"⚠ 批量保存失败，使用单条保存: {e}")
                # 降级到单条保存
                try:
                    qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=hash(memory_id),
                                vector=embedding,
                                payload=payload
                            )
                        ]
                    )
                    print(f"[OK] 记忆已保存到Qdrant: {memory_id}")
                except Exception as e2:
                    print(f"⚠ Qdrant保存失败: {e2}")
        
        # 确保memory_info中有memory_id
        memory_info["memory_id"] = memory_id
    
    def _build_vector_text(self, content: str, memory_info: dict, summary: str) -> str:
        """构建向量化文本"""
        key_entities = memory_info.get("key_entities", {})
        parts = [content]
        
        if key_entities:
            entity_str = " ".join([f"{k}:{v}" for k, v in key_entities.items() if v])
            if entity_str:
                parts.append(entity_str)
        
        parts.append(f"类型:{memory_info.get('memory_type', 'event')}")
        if summary:
            parts.append(f"摘要:{summary}")
        
        enhanced_text = " ".join(parts)
        
        # 行程记忆特殊处理
        if memory_info.get("is_schedule") or memory_info["memory_type"] == "schedule":
            enhanced_text = self._build_schedule_vector_text(content, memory_info, key_entities)
        
        return enhanced_text
    
    def _extract_schedule_summary(self, content: str) -> str:
        """提取行程摘要"""
        time_patterns = [
            r"明天", r"后天", r"下周[一二三四五六日]", r"下周一", r"下周二", r"下周三",
            r"这个周末", r"这周末", r"下周末", r"下个月", r"\d+号", r"\d+日"
        ]
        time_info = ""
        for pattern in time_patterns:
            match = re.search(pattern, content)
            if match:
                time_info = match.group()
                break
        
        action_keywords = ["要去", "准备去", "计划", "安排", "约了", "预约"]
        action = ""
        for kw in action_keywords:
            if kw in content:
                action = kw
                break
        
        if time_info and action:
            action_idx = content.find(action)
            if action_idx >= 0:
                after_action = content[action_idx + len(action):].strip()
                if len(after_action) > 30:
                    after_action = after_action[:30] + "..."
                return f"{time_info}{action}{after_action}"
        
        return content[:60] + "..." if len(content) > 60 else content
    
    def _build_schedule_vector_text(self, content: str, memory_info: dict, key_entities: dict = None) -> str:
        """构建行程的增强向量化文本"""
        time_period = memory_info.get("time_period", "")
        
        if key_entities:
            parts = ["用户行程安排", content]
            
            if key_entities.get("时间"):
                parts.append(f"时间:{key_entities['时间']}")
            if key_entities.get("地点"):
                parts.append(f"地点:{key_entities['地点']}")
            if key_entities.get("人物"):
                parts.append(f"人物:{key_entities['人物']}")
            if key_entities.get("事件"):
                parts.append(f"事件:{key_entities['事件']}")
            
            parts.append(f"时间段:{time_period}")
            return " ".join(parts)
        else:
            enhanced = f"用户行程安排：{content} 时间段：{time_period}"
            time_keywords = ["明天", "后天", "下周", "周末", "下个月"]
            for kw in time_keywords:
                if kw in content:
                    enhanced += f" 时间：{kw}"
                    break
            return enhanced
    
    def retrieve_memories(self, query: str, top_k: int = 5, prioritize_schedule: bool = True) -> List[Dict]:
        """
        检索相关记忆（行程记忆优先）
        使用Qdrant向量检索（使用embedding_manager，支持多种嵌入模型）
        """
        if not qdrant_client:
            return []
        
        from embedding_manager import MemoryType, encode_text
        
        # 使用文本类型的嵌入模型进行检索
        query_vector = encode_text(query, MemoryType.TEXT)
        
        if not query_vector:
            return []
        
        try:
            # 使用Qdrant检索
            search_k = min(top_k * 2, 100) if prioritize_schedule else top_k
            
            # 优先检索行程记忆
            schedule_results = []
            other_results = []
            
            if prioritize_schedule:
                # 先检索行程记忆
                schedule_filter = Filter(
                    must=[
                        FieldCondition(key="is_schedule", match=MatchValue(value=True))
                    ]
                )
                schedule_results = qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=schedule_filter,
                    limit=top_k
                )
            
            # 检索所有记忆
            all_results = qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=search_k
            )
            
            # 处理结果
            memories = []
            seen_ids = set()
            
            # 优先添加行程记忆
            for result in schedule_results:
                if result.id not in seen_ids:
                    memories.append({
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
                    seen_ids.add(result.id)
            
            # 添加其他记忆
            for result in all_results:
                if result.id not in seen_ids and len(memories) < top_k:
                    memories.append({
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
                    seen_ids.add(result.id)
            
            return memories[:top_k]
        except Exception as e:
            print(f"Qdrant检索失败: {e}")
            return []
    
    def _update_user_profile(self, db: Session, emotion_analysis: dict, role_preference: dict, user_input: str = ""):
        """更新用户画像（增强版：记录习惯、偏好、成长变化）"""
        from app import get_time_period, get_beijing_datetime
        
        profile = db.query(UserProfile).filter(UserProfile.user_id == self.user_id).first()
        
        if not profile:
            profile = UserProfile(
                user_id=self.user_id,
                updated_at=datetime.now()
            )
            db.add(profile)
        
        current_time = get_beijing_datetime()
        time_period = get_time_period(current_time)
        
        # 更新沟通偏好（统计最常见的选择）
        comm_prefs = json.loads(profile.communication_preferences) if profile.communication_preferences else {}
        
        # 更新角色偏好统计
        role_type = role_preference.get("role_type", "朋友")
        if "role_stats" not in comm_prefs:
            comm_prefs["role_stats"] = {}
        comm_prefs["role_stats"][role_type] = comm_prefs["role_stats"].get(role_type, 0) + 1
        
        # 更新沟通风格统计
        style = role_preference.get("communication_style", "温和关怀")
        if "style_stats" not in comm_prefs:
            comm_prefs["style_stats"] = {}
        comm_prefs["style_stats"][style] = comm_prefs["style_stats"].get(style, 0) + 1
        
        profile.communication_preferences = json.dumps(comm_prefs, ensure_ascii=False)
        
        # 更新情绪模式
        emotion_patterns = json.loads(profile.emotional_patterns) if profile.emotional_patterns else {}
        emotion_type = emotion_analysis.get("emotion_type", "neutral")
        if "emotion_stats" not in emotion_patterns:
            emotion_patterns["emotion_stats"] = {}
        emotion_patterns["emotion_stats"][emotion_type] = emotion_patterns["emotion_stats"].get(emotion_type, 0) + 1
        
        # 计算最常见情绪
        if emotion_patterns["emotion_stats"]:
            most_common = max(emotion_patterns["emotion_stats"].items(), key=lambda x: x[1])
            emotion_patterns["most_common_emotion"] = most_common[0]
        
        # 更新时间模式（活跃时间段统计）
        if "active_time_patterns" not in emotion_patterns:
            emotion_patterns["active_time_patterns"] = {}
        emotion_patterns["active_time_patterns"][time_period] = emotion_patterns["active_time_patterns"].get(time_period, 0) + 1
        
        # 计算最活跃时间段
        if emotion_patterns["active_time_patterns"]:
            most_active = max(emotion_patterns["active_time_patterns"].items(), key=lambda x: x[1])
            emotion_patterns["most_active_period"] = most_active[0]
        
        profile.emotional_patterns = json.dumps(emotion_patterns, ensure_ascii=False)
        
        # 更新用户画像摘要（使用LLM生成，如果可用）
        if user_input and len(user_input) > 10:
            try:
                from app import client
                
                profile_prompt = f"""基于以下信息更新用户画像摘要：
用户输入：{user_input}
当前情绪：{emotion_type}（{emotion_analysis.get('emotion_intensity', '中等')}）
沟通偏好：{role_type} - {style}

现有画像摘要：{getattr(profile, 'profile_summary', None) or '暂无'}

请生成或更新用户画像摘要（100字以内），包括：
1. 用户的主要特征
2. 沟通偏好
3. 情绪模式
4. 可能需要关注的点

只输出摘要文本，不要其他说明。"""
                
                profile_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=profile_prompt
                )
                
                new_summary = profile_response.text.strip()
                if new_summary and len(new_summary) > 20:
                    # 使用setattr安全设置字段（如果不存在会自动创建）
                    setattr(profile, 'profile_summary', new_summary[:200])
            except Exception as e:
                print(f"更新用户画像摘要失败: {e}")
        
        profile.updated_at = datetime.now()
        db.commit()
    
    def get_user_context(self, db: Session, query: str = "") -> str:
        """获取用户上下文信息（用于构建AI回复提示词）"""
        context_parts = []
        
        # 检索相关记忆
        memories = self.retrieve_memories(query, top_k=3, prioritize_schedule=True)
        if memories:
            memory_contexts = []
            for mem in memories[:3]:
                summary = mem.get("metadata", {}).get("summary", mem.get("content", ""))
                if summary:
                    memory_contexts.append(summary)
            if memory_contexts:
                context_parts.append(f"相关记忆：{'；'.join(memory_contexts)}")
        
        # 获取用户画像摘要（安全访问，处理字段可能不存在的情况）
        profile = db.query(UserProfile).filter(UserProfile.user_id == self.user_id).first()
        if profile:
            # 使用getattr安全访问profile_summary字段
            profile_summary = getattr(profile, 'profile_summary', None)
            if profile_summary:
                context_parts.append(f"用户画像：{profile_summary[:100]}")
        
        return "；".join(context_parts) if context_parts else ""

def _get_collection_profile(collection_info) -> str:
    """根据集合信息推断当前使用的profile（辅助函数）"""
    # 尝试从HNSW配置推断profile
    if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'hnsw_config'):
        hnsw = collection_info.config.hnsw_config
        m = getattr(hnsw, 'm', 16)
        ef_construct = getattr(hnsw, 'ef_construct', 64)
        
        # 根据参数推断profile
        if m == 16 and ef_construct <= 64:
            return "small"
        elif m == 24 and ef_construct <= 100:
            return "medium"
        elif m >= 32 and ef_construct >= 128:
            return "large"
    
    return "small"  # 默认

def maintain_memories():
    """定期维护记忆系统：清理过期记忆、优化向量数据库等（包括Qdrant动态调优）"""
    try:
        db = next(get_db())
        
        # 清理低重要性、过期的记忆（带衰减因子）
        cutoff_date = datetime.now() - timedelta(days=90)
        
        # 查询过期记忆
        expired_memories = db.query(Memory).filter(
            Memory.created_at < cutoff_date,
            Memory.relevance_score < 0.5
        ).all()
        
        deleted_count = 0
        for memory in expired_memories:
            # 计算衰减因子（时间越久，衰减越大）
            days_old = (datetime.now() - memory.created_at).days
            decay_factor = max(0.0, 1.0 - days_old / 180.0)  # 180天后完全衰减
            
            # 如果衰减后的重要性很低，删除
            final_score = memory.relevance_score * decay_factor
            if final_score < 0.3:
                # 同时从Qdrant删除
                if qdrant_client:
                    try:
                        qdrant_client.delete(
                            collection_name=f"user_{memory.user_id}_memories",
                            points_selector=[hash(memory.memory_id)]
                        )
                    except Exception as e:
                        print(f"从Qdrant删除失败: {e}")
                
                db.delete(memory)
                deleted_count += 1
        
        if deleted_count > 0:
            print(f"[OK] 清理了 {deleted_count} 条过期记忆")
        
        db.commit()
        
        # Qdrant动态调优：根据向量数量调整HNSW配置
        if qdrant_client:
            try:
                from qdrant_config import get_qdrant_config
                from config import settings
                
                if settings.QDRANT_AUTO_TUNE:
                    qdrant_config = get_qdrant_config()
                    
                    # 获取所有用户的集合
                    collections = qdrant_client.get_collections().collections
                    tuned_count = 0
                    
                    for collection in collections:
                        collection_name = collection.name
                        if collection_name.startswith("user_") and collection_name.endswith("_memories"):
                            try:
                                collection_info = qdrant_client.get_collection(collection_name)
                                vector_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                                
                                if vector_count > 0:
                                    profile = qdrant_config.auto_tune_profile(vector_count)
                                    current_profile = _get_collection_profile(collection_info)
                                    
                                    if current_profile != profile:
                                        # 更新配置
                                        if qdrant_config.update_collection_config(
                                            collection_name, qdrant_client, profile, vector_count
                                        ):
                                            tuned_count += 1
                            except Exception as e:
                                print(f"⚠ 调优集合 {collection_name} 失败: {e}")
                    
                    if tuned_count > 0:
                        print(f"[OK] 动态调优了 {tuned_count} 个Qdrant集合")
            except Exception as e:
                print(f"⚠ Qdrant动态调优失败: {e}")
        
        db.close()
        
        print("[OK] 记忆维护完成")
    except Exception as e:
        print(f"⚠ 记忆维护失败: {e}")
