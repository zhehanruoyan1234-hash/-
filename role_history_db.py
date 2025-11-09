"""
角色历史数据库模块
使用SQLite存储角色历史，避免内存泄漏
"""
import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import threading
from pathlib import Path

# 数据库文件路径
DB_PATH = Path("data/role_history.db")

# 线程锁（SQLite需要）
_db_lock = threading.Lock()

def get_db_connection():
    """获取数据库连接（线程安全）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库表"""
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS role_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role_type TEXT NOT NULL,
                communication_style TEXT,
                tone TEXT,
                confidence_role REAL,
                confidence_style REAL,
                confidence_tone REAL,
                user_input TEXT,
                emotion_type TEXT,
                emotion_intensity TEXT,
                switch_reason TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引以提高查询性能
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_timestamp 
            ON role_history(user_id, timestamp DESC)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id 
            ON role_history(user_id)
        """)
        
        conn.commit()
        print("[OK] 角色历史数据库初始化成功")
    except Exception as e:
        print(f"⚠ 初始化角色历史数据库失败: {e}")
    finally:
        conn.close()

def save_role_history(user_id: str, role_data: Dict):
    """保存角色历史记录"""
    with _db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT INTO role_history (
                    user_id, role_type, communication_style, tone,
                    confidence_role, confidence_style, confidence_tone,
                    user_input, emotion_type, emotion_intensity, switch_reason, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                role_data.get("role_type", ""),
                role_data.get("communication_style", ""),
                role_data.get("tone", ""),
                role_data.get("confidence", {}).get("role", 0.5),
                role_data.get("confidence", {}).get("style", 0.5),
                role_data.get("confidence", {}).get("tone", 0.5),
                role_data.get("user_input", "")[:200],  # 限制长度
                role_data.get("emotion_type", ""),
                role_data.get("emotion_intensity", ""),
                role_data.get("switch_reason", ""),
                datetime.now().isoformat()
            ))
            conn.commit()
        except Exception as e:
            print(f"⚠ 保存角色历史失败: {e}")
        finally:
            conn.close()

@lru_cache(maxsize=100)
def get_role_history_cached(user_id: str, limit: int = 100):
    """获取角色历史（带LRU缓存）"""
    return get_role_history(user_id, limit)

def get_role_history(user_id: str, limit: int = 100) -> List[Dict]:
    """获取用户角色历史"""
    with _db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM role_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = []
            for row in cursor:
                results.append({
                    "role_type": row["role_type"],
                    "communication_style": row["communication_style"],
                    "tone": row["tone"],
                    "confidence": {
                        "role": row["confidence_role"],
                        "style": row["confidence_style"],
                        "tone": row["confidence_tone"]
                    },
                    "user_input": row["user_input"],
                    "emotion_type": row["emotion_type"],
                    "emotion_intensity": row["emotion_intensity"],
                    "switch_reason": row["switch_reason"],
                    "timestamp": row["timestamp"]
                })
            
            return list(reversed(results))  # 按时间正序返回
        except Exception as e:
            print(f"⚠ 获取角色历史失败: {e}")
            return []
        finally:
            conn.close()

def get_role_statistics(user_id: str, days: int = 30) -> Dict:
    """获取角色统计信息"""
    with _db_lock:
        conn = get_db_connection()
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor = conn.execute("""
                SELECT 
                    role_type,
                    COUNT(*) as count,
                    AVG(confidence_role) as avg_confidence
                FROM role_history
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY role_type
                ORDER BY count DESC
            """, (user_id, cutoff_date))
            
            stats = {
                "total_records": 0,
                "role_distribution": {},
                "most_common_role": None,
                "avg_confidence": 0.0
            }
            
            role_counts = {}
            total_confidence = 0.0
            total_records = 0
            
            for row in cursor:
                role_type = row["role_type"]
                count = row["count"]
                avg_conf = row["avg_confidence"] or 0.5
                
                role_counts[role_type] = count
                stats["role_distribution"][role_type] = {
                    "count": count,
                    "avg_confidence": avg_conf
                }
                total_confidence += avg_conf * count
                total_records += count
            
            if total_records > 0:
                stats["total_records"] = total_records
                stats["avg_confidence"] = total_confidence / total_records
                stats["most_common_role"] = max(role_counts.items(), key=lambda x: x[1])[0] if role_counts else None
            
            return stats
        except Exception as e:
            print(f"⚠ 获取角色统计失败: {e}")
            return {}
        finally:
            conn.close()

def cleanup_old_history(days: int = 90):
    """清理旧的历史记录"""
    with _db_lock:
        conn = get_db_connection()
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            cursor = conn.execute("""
                DELETE FROM role_history
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            # 清除缓存
            get_role_history_cached.cache_clear()
            
            print(f"[OK] 清理了 {deleted_count} 条旧的角色历史记录")
            return deleted_count
        except Exception as e:
            print(f"⚠ 清理角色历史失败: {e}")
            return 0
        finally:
            conn.close()

# 初始化数据库
init_db()

