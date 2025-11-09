"""
Qdrant配置管理模块
支持动态HNSW参数调优，根据数据规模自动选择最优配置
"""
import os
import json
from typing import Dict, Optional
from pathlib import Path

# Qdrant导入（可选）
try:
    from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠ qdrant_client未安装，Qdrant配置功能将受限")

# 配置文件路径
QDRANT_CONFIG_PATH = Path("qdrant_config.json")

# 默认配置（小规模数据）
DEFAULT_HNSW_CONFIG = {
    "small": {
        "m": 16,
        "ef_construct": 64,
        "full_scan_threshold": 10000,
        "description": "小规模数据（<10K向量）"
    },
    "medium": {
        "m": 24,
        "ef_construct": 100,
        "full_scan_threshold": 50000,
        "description": "中等规模数据（10K-100K向量）"
    },
    "large": {
        "m": 32,
        "ef_construct": 128,
        "full_scan_threshold": 100000,
        "description": "大规模数据（>100K向量，高精度）"
    }
}

# 默认距离度量配置（如果Qdrant可用）
if QDRANT_AVAILABLE:
    DEFAULT_DISTANCE = Distance.COSINE
else:
    DEFAULT_DISTANCE = None

class QdrantConfig:
    """Qdrant配置管理类"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or QDRANT_CONFIG_PATH
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[OK] Qdrant配置加载成功: {self.config_path}")
                return config
            except Exception as e:
                print(f"⚠ 加载Qdrant配置失败: {e}，使用默认配置")
                return self._get_default_config()
        else:
            # 配置文件不存在，创建默认配置
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "distance": "COSINE",
            "hnsw_profiles": DEFAULT_HNSW_CONFIG,
            "default_profile": "small",
            "auto_tune": True,
            "auto_tune_thresholds": {
                "small_to_medium": 10000,
                "medium_to_large": 100000
            },
            "optimizers": {
                "indexing_threshold": 10000,
                "flush_interval_sec": 5,
                "max_optimization_threads": 1
            }
        }
    
    def _save_config(self, config: Dict):
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"[OK] Qdrant配置已保存: {self.config_path}")
        except Exception as e:
            print(f"⚠ 保存Qdrant配置失败: {e}")
    
    def get_distance(self) -> 'Distance':
        """获取距离度量"""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant_client未安装")
        
        distance_str = self.config.get("distance", "COSINE").upper()
        # 只使用COSINE，因为不同版本的Qdrant可能不支持所有距离度量
        # 根据Qdrant文档：https://qdrant.tech/documentation/concepts/distance/
        if distance_str == "COSINE":
            return Distance.COSINE
        else:
            # 如果配置了其他距离度量，检查是否可用
            # 默认使用COSINE（最常用）
            try:
                if hasattr(Distance, distance_str):
                    return getattr(Distance, distance_str)
            except:
                pass
            # 降级到COSINE
            return Distance.COSINE
    
    def get_hnsw_config(self, profile: Optional[str] = None) -> Dict:
        """获取HNSW配置"""
        profiles = self.config.get("hnsw_profiles", DEFAULT_HNSW_CONFIG)
        
        if profile:
            if profile in profiles:
                return profiles[profile]
            else:
                print(f"⚠ HNSW配置profile '{profile}'不存在，使用默认profile")
                profile = None
        
        if not profile:
            profile = self.config.get("default_profile", "small")
        
        return profiles.get(profile, DEFAULT_HNSW_CONFIG["small"])
    
    def get_optimizers_config(self) -> Dict:
        """获取优化器配置"""
        return self.config.get("optimizers", {
            "indexing_threshold": 10000,
            "flush_interval_sec": 5,
            "max_optimization_threads": 1
        })
    
    def auto_tune_profile(self, vector_count: int) -> str:
        """根据向量数量自动选择HNSW配置profile"""
        if not self.config.get("auto_tune", True):
            return self.config.get("default_profile", "small")
        
        thresholds = self.config.get("auto_tune_thresholds", {
            "small_to_medium": 10000,
            "medium_to_large": 100000
        })
        
        small_to_medium = thresholds.get("small_to_medium", 10000)
        medium_to_large = thresholds.get("medium_to_large", 100000)
        
        if vector_count < small_to_medium:
            return "small"
        elif vector_count < medium_to_large:
            return "medium"
        else:
            return "large"
    
    def get_vector_params(self, dimension: int, profile: Optional[str] = None) -> 'VectorParams':
        """获取向量参数配置"""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant_client未安装")
        
        distance = self.get_distance()
        
        return VectorParams(
            size=dimension,
            distance=distance
        )
    
    def get_hnsw_config_obj(self, profile: Optional[str] = None) -> 'HnswConfigDiff':
        """获取HNSW配置对象（用于Qdrant API）"""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant_client未安装")
        
        hnsw_config = self.get_hnsw_config(profile)
        
        return HnswConfigDiff(
            m=hnsw_config.get("m", 16),
            ef_construct=hnsw_config.get("ef_construct", 64),
            full_scan_threshold=hnsw_config.get("full_scan_threshold", 10000)
        )
    
    def get_optimizers_config_obj(self) -> 'OptimizersConfigDiff':
        """获取优化器配置对象"""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant_client未安装")
        
        optimizers_config = self.get_optimizers_config()
        
        return OptimizersConfigDiff(
            indexing_threshold=optimizers_config.get("indexing_threshold", 10000),
            flush_interval_sec=optimizers_config.get("flush_interval_sec", 5),
            max_optimization_threads=optimizers_config.get("max_optimization_threads", 1)
        )
    
    def update_collection_config(self, collection_name: str, qdrant_client, 
                                profile: Optional[str] = None, vector_count: int = 0):
        """动态更新集合配置"""
        if not qdrant_client:
            return False
        
        try:
            # 如果提供了向量数量，自动选择profile
            if vector_count > 0:
                profile = self.auto_tune_profile(vector_count)
            
            # 获取配置
            hnsw_config = self.get_hnsw_config_obj(profile)
            optimizers_config = self.get_optimizers_config_obj()
            
            # 更新集合配置
            qdrant_client.update_collection(
                collection_name=collection_name,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config
            )
            
            print(f"[OK] 集合 {collection_name} 配置已更新: profile={profile}")
            return True
        except Exception as e:
            print(f"⚠ 更新集合配置失败: {e}")
            return False

# 全局配置实例
_qdrant_config = None

def get_qdrant_config() -> QdrantConfig:
    """获取Qdrant配置实例（单例）"""
    global _qdrant_config
    if _qdrant_config is None:
        _qdrant_config = QdrantConfig()
    return _qdrant_config

