"""
配置管理模块
从环境变量加载配置，支持默认值和类型转换
"""
import os
from typing import Optional
from pathlib import Path

# 尝试加载python-dotenv
try:
    from dotenv import load_dotenv
    # 加载.env文件
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()  # 尝试从环境变量加载
except ImportError:
    print("⚠ python-dotenv未安装，将使用系统环境变量")

class Settings:
    """应用配置类"""
    
    # API配置
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./memory.db")
    
    # Qdrant配置
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_db")
    QDRANT_HOST: Optional[str] = os.getenv("QDRANT_HOST", None)
    QDRANT_PORT: Optional[int] = int(os.getenv("QDRANT_PORT", "6333")) if os.getenv("QDRANT_PORT") else None
    QDRANT_CONFIG_FILE: str = os.getenv("QDRANT_CONFIG_FILE", "qdrant_config.json")
    QDRANT_AUTO_TUNE: bool = os.getenv("QDRANT_AUTO_TUNE", "true").lower() == "true"
    
    # 安全配置
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production-please")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # API限流配置
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "app.log")
    
    # 调试模式
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API版本
    API_V1_PREFIX: str = "/api/v1"
    
    @classmethod
    def validate(cls):
        """验证必需的配置项"""
        if not cls.GEMINI_API_KEY:
            if cls.DEBUG:
                print("⚠ 警告：GEMINI_API_KEY环境变量未设置，请在.env文件中配置")
                print("⚠ 调试模式：继续运行（可能影响功能）")
            else:
                raise ValueError("GEMINI_API_KEY环境变量未设置，请在.env文件中配置")
        
        if cls.SECRET_KEY == "your-secret-key-change-this-in-production-please":
            if not cls.DEBUG:
                print("⚠ 警告：SECRET_KEY使用默认值，生产环境请修改！")

# 全局配置实例
settings = Settings()

# 验证配置（非阻塞，允许调试模式继续）
try:
    settings.validate()
except ValueError as e:
    print(f"⚠ 配置验证失败: {e}")
    if not settings.DEBUG:
        raise  # 非调试模式下，配置错误应该阻止启动

