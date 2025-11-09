"""
数据库模型和初始化
使用 SQLite 存储用户数据
"""
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import json

Base = declarative_base()

class User(Base):
    """用户基础信息表"""
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)  # 用户名称
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now)
    total_conversations = Column(Integer, default=0)
    preferred_role_patterns = Column(Text, default="{}")  # JSON格式
    dominant_emotions = Column(Text, default="{}")  # JSON格式

class UserProfile(Base):
    """用户画像表"""
    __tablename__ = "user_profile"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    interests = Column(Text, default="[]")  # JSON数组
    personality_traits = Column(Text, default="[]")  # JSON数组
    communication_preferences = Column(Text, default="{}")  # JSON对象
    emotional_patterns = Column(Text, default="{}")  # JSON对象
    topics_history = Column(Text, default="[]")  # JSON数组
    important_events = Column(Text, default="[]")  # JSON数组
    relationships_info = Column(Text, default="{}")  # JSON对象
    profile_summary = Column(Text, nullable=True)  # 用户画像摘要

class Conversation(Base):
    """对话历史表"""
    __tablename__ = "conversations"
    
    conversation_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    user_input = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    emotion_analysis = Column(Text, default="{}")  # JSON对象
    role_preference = Column(Text, default="{}")  # JSON对象
    topic_tags = Column(Text, default="[]")  # JSON数组
    importance_score = Column(Float, default=0.0)
    
    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_input": self.user_input,
            "ai_response": self.ai_response,
            "emotion_analysis": json.loads(self.emotion_analysis) if self.emotion_analysis else {},
            "role_preference": json.loads(self.role_preference) if self.role_preference else {},
            "topic_tags": json.loads(self.topic_tags) if self.topic_tags else [],
            "importance_score": self.importance_score
        }

class Memory(Base):
    """重要记忆表"""
    __tablename__ = "memories"
    
    memory_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)
    memory_type = Column(String, nullable=False)  # personal_info, preference, event, relationship, schedule
    content = Column(Text, nullable=False)
    summary = Column(Text)  # AI生成的记忆摘要
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.now)
    relevance_score = Column(Float, default=0.0)

# 数据库初始化
from config import settings
from sqlalchemy.pool import QueuePool
from sqlalchemy import event

DATABASE_URL = settings.DATABASE_URL
# 使用连接池和WAL模式优化SQLite并发性能
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # 自动重连
    connect_args={"check_same_thread": False}
)

# SQLite WAL模式优化（提高并发性能）
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """设置SQLite优化参数"""
    cursor = dbapi_conn.cursor()
    # WAL模式：允许多个读取器和一个写入器同时访问数据库
    cursor.execute("PRAGMA journal_mode=WAL")
    # 同步模式：NORMAL在WAL模式下足够安全
    cursor.execute("PRAGMA synchronous=NORMAL")
    # 缓存大小：提高查询性能
    cursor.execute("PRAGMA cache_size=10000")
    # 外键约束：确保数据完整性
    cursor.execute("PRAGMA foreign_keys=ON")
    # 临时存储：使用内存临时表提高性能
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()

def init_db():
    """初始化数据库并执行迁移"""
    # 创建所有表（如果不存在）
    Base.metadata.create_all(engine)
    
    # 检查并添加缺失的列（数据库迁移）
    try:
        with engine.connect() as conn:
            # 检查users表是否存在
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'"))
            table_exists = result.fetchone()
            
            if table_exists:
                # 检查users表是否有name列
                result = conn.execute(text("PRAGMA table_info(users)"))
                columns = [row[1] for row in result]
                
                if 'name' not in columns:
                    print("正在添加 name 列到 users 表...")
                    conn.execute(text("ALTER TABLE users ADD COLUMN name VARCHAR"))
                    conn.commit()
                    print("✓ name 列添加成功")
            
            # 检查user_profile表是否存在并添加profile_summary列
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profile'"))
            profile_table_exists = result.fetchone()
            
            if profile_table_exists:
                result = conn.execute(text("PRAGMA table_info(user_profile)"))
                columns = [row[1] for row in result]
                
                if 'profile_summary' not in columns:
                    print("正在添加 profile_summary 列到 user_profile 表...")
                    conn.execute(text("ALTER TABLE user_profile ADD COLUMN profile_summary TEXT"))
                    conn.commit()
                    print("✓ profile_summary 列添加成功")
    except Exception as e:
        print(f"数据库迁移检查失败: {e}")
        # 如果出错，仍然继续运行（可能表不存在，会在create_all中创建）

# 初始化数据库
init_db()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_user(db: Session, user_id: str):
    """获取或创建用户"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(
            user_id=user_id,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.last_active = datetime.now()
        db.commit()
    return user

