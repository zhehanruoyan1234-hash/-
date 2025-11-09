"""
角色切换日志系统
记录详细的角色切换信息，支持监控系统对接
"""
import json
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

# 日志配置
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 配置日志记录器
role_logger = logging.getLogger("role_switching")
role_logger.setLevel(logging.INFO)

# 文件处理器（记录到文件）
file_handler = logging.FileHandler(LOG_DIR / "role_switching.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 控制台处理器（可选）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# 格式器
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

role_logger.addHandler(file_handler)
role_logger.addHandler(console_handler)

# 角色切换统计
_switch_stats = {
    "total_switches": 0,
    "switch_by_reason": {},
    "switch_by_role": {}
}

def log_role_switch(user_id: str, old_role: Dict, new_role: Dict,
                   switch_reason: str, confidence: Dict, 
                   emotion_analysis: Dict = None, user_input: str = ""):
    """
    记录角色切换日志
    
    Args:
        user_id: 用户ID
        old_role: 旧角色信息
        new_role: 新角色信息
        switch_reason: 切换原因
        confidence: 置信度信息
        emotion_analysis: 情绪分析结果
        user_input: 用户输入（可选）
    """
    timestamp = datetime.now().isoformat()
    
    # 构建日志记录
    log_entry = {
        "timestamp": timestamp,
        "user_id": user_id,
        "old_role": {
            "role_type": old_role.get("role_type", ""),
            "communication_style": old_role.get("communication_style", ""),
            "tone": old_role.get("tone", "")
        },
        "new_role": {
            "role_type": new_role.get("role_type", ""),
            "communication_style": new_role.get("communication_style", ""),
            "tone": new_role.get("tone", "")
        },
        "switch_reason": switch_reason,
        "confidence": confidence,
        "emotion_analysis": emotion_analysis or {},
        "user_input_preview": user_input[:100] if user_input else ""
    }
    
    # 记录到日志文件
    role_logger.info(f"ROLE_SWITCH | {json.dumps(log_entry, ensure_ascii=False)}")
    
    # 更新统计信息
    _switch_stats["total_switches"] += 1
    
    reason_key = switch_reason.split(":")[0] if ":" in switch_reason else switch_reason
    _switch_stats["switch_by_reason"][reason_key] = \
        _switch_stats["switch_by_reason"].get(reason_key, 0) + 1
    
    old_role_type = old_role.get("role_type", "unknown")
    new_role_type = new_role.get("role_type", "unknown")
    switch_key = f"{old_role_type} -> {new_role_type}"
    _switch_stats["switch_by_role"][switch_key] = \
        _switch_stats["switch_by_role"].get(switch_key, 0) + 1

def log_role_analysis(user_id: str, role_result: Dict, emotion_analysis: Dict = None):
    """
    记录角色分析结果（非切换情况）
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "role_type": role_result.get("role_type", ""),
        "communication_style": role_result.get("communication_style", ""),
        "tone": role_result.get("tone", ""),
        "confidence": role_result.get("confidence", {}),
        "emotion_type": emotion_analysis.get("emotion_type", "") if emotion_analysis else "",
        "emotion_intensity": emotion_analysis.get("emotion_intensity", "") if emotion_analysis else ""
    }
    
    role_logger.debug(f"ROLE_ANALYSIS | {json.dumps(log_entry, ensure_ascii=False)}")

def get_switch_statistics() -> Dict:
    """获取角色切换统计信息"""
    return _switch_stats.copy()

def reset_statistics():
    """重置统计信息"""
    global _switch_stats
    _switch_stats = {
        "total_switches": 0,
        "switch_by_reason": {},
        "switch_by_role": {}
    }

# 监控系统对接（Prometheus风格）
def export_metrics() -> Dict:
    """导出监控指标"""
    return {
        "role_switches_total": _switch_stats["total_switches"],
        "role_switches_by_reason": _switch_stats["switch_by_reason"],
        "role_switches_by_role": _switch_stats["switch_by_role"]
    }

