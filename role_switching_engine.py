"""
角色切换规则引擎
支持从JSON配置文件加载规则，支持规则插件机制
"""
import json
from typing import List, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime

# 规则配置文件路径
RULES_CONFIG_PATH = Path("role_switching_rules.json")

class RulePlugin:
    """规则插件基类"""
    def __init__(self, name: str, priority: float = 0.5):
        self.name = name
        self.priority = priority
    
    def check_condition(self, user_input: str, emotion_analysis: dict, 
                       current_time: str, context: dict = None) -> Optional[Dict]:
        """
        检查是否满足切换条件
        
        Returns:
            None: 不满足条件
            Dict: 满足条件，返回目标角色信息
        """
        raise NotImplementedError

class EmotionRulePlugin(RulePlugin):
    """情绪驱动规则插件"""
    def __init__(self, config: dict):
        super().__init__(config.get("name", "emotion_rule"), config.get("priority", 0.7))
        self.target_roles = config.get("target_roles", [])
        self.conditions = config.get("conditions", {})
    
    def check_condition(self, user_input: str, emotion_analysis: dict,
                       current_time: str, context: dict = None) -> Optional[Dict]:
        emotion_type = emotion_analysis.get("emotion_type", "")
        emotion_intensity = emotion_analysis.get("emotion_intensity", "")
        
        required_types = self.conditions.get("emotion_types", [])
        required_intensity = self.conditions.get("intensity", "")
        
        # 检查情绪类型
        if isinstance(required_types, list):
            if emotion_type not in required_types:
                return None
        elif emotion_type != required_types:
            return None
        
        # 检查情绪强度
        if isinstance(required_intensity, list):
            if emotion_intensity not in required_intensity:
                return None
        elif emotion_intensity != required_intensity:
            return None
        
        return {
            "target_roles": self.target_roles,
            "priority": self.priority,
            "reason": f"情绪触发: {emotion_type} ({emotion_intensity})"
        }

class TopicRulePlugin(RulePlugin):
    """主题驱动规则插件"""
    def __init__(self, config: dict):
        super().__init__(config.get("name", "topic_rule"), config.get("priority", 0.6))
        self.target_roles = config.get("target_roles", [])
        self.keywords = config.get("keywords", [])
        self.min_match_count = config.get("min_match_count", 1)
    
    def check_condition(self, user_input: str, emotion_analysis: dict,
                       current_time: str, context: dict = None) -> Optional[Dict]:
        user_input_lower = user_input.lower()
        match_count = sum(1 for kw in self.keywords if kw in user_input_lower)
        
        if match_count >= self.min_match_count:
            return {
                "target_roles": self.target_roles,
                "priority": self.priority,
                "reason": f"主题匹配: 匹配到 {match_count} 个关键词"
            }
        return None

class ScenarioRulePlugin(RulePlugin):
    """场景驱动规则插件"""
    def __init__(self, config: dict):
        super().__init__(config.get("name", "scenario_rule"), config.get("priority", 0.6))
        self.target_roles = config.get("target_roles", [])
        self.time_range = config.get("time_range", {})
    
    def check_condition(self, user_input: str, emotion_analysis: dict,
                       current_time: str, context: dict = None) -> Optional[Dict]:
        # 解析时间范围
        start_str = self.time_range.get("start", "00:00")
        end_str = self.time_range.get("end", "23:59")
        
        try:
            # 解析当前时间（支持多种格式）
            # 格式1: "2025-11-05 16:49:07" -> 提取 "16:49"
            # 格式2: "16:49:07" -> 提取 "16:49"
            # 格式3: "16:49" -> 直接使用
            
            # 先尝试提取时间部分
            if " " in current_time:
                # 包含日期的时间格式 "2025-11-05 16:49:07"
                time_part = current_time.split(" ")[1]  # 获取 "16:49:07"
                current_hour, current_minute = map(int, time_part.split(":")[:2])
            elif ":" in current_time:
                # 只有时间格式 "16:49:07" 或 "16:49"
                parts = current_time.split(":")
                current_hour = int(parts[0])
                current_minute = int(parts[1]) if len(parts) > 1 else 0
            else:
                # 无法解析，返回None
                return None
            
            start_hour, start_minute = map(int, start_str.split(":"))
            end_hour, end_minute = map(int, end_str.split(":"))
            
            current_minutes = current_hour * 60 + current_minute
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute
            
            # 处理跨天的情况（如22:00-02:00）
            if end_minutes < start_minutes:
                if current_minutes >= start_minutes or current_minutes <= end_minutes:
                    return {
                        "target_roles": self.target_roles,
                        "priority": self.priority,
                        "reason": f"场景匹配: {start_str}-{end_str}"
                    }
            else:
                if start_minutes <= current_minutes <= end_minutes:
                    return {
                        "target_roles": self.target_roles,
                        "priority": self.priority,
                        "reason": f"场景匹配: {start_str}-{end_str}"
                    }
        except Exception as e:
            print(f"⚠ 场景规则检查失败: {e}")
        
        return None

class RoleSwitchingEngine:
    """角色切换规则引擎"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or RULES_CONFIG_PATH
        self.rules_config = {}
        self.plugins: List[RulePlugin] = []
        self.custom_plugins: List[RulePlugin] = []
        self.load_rules()
    
    def load_rules(self):
        """从配置文件加载规则"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.rules_config = json.load(f)
                
                # 加载内置规则插件
                self._load_builtin_plugins()
                
                # 加载自定义插件
                self._load_custom_plugins()
                
                print(f"✓ 角色切换规则加载成功: {len(self.plugins)} 个规则")
            else:
                print(f"⚠ 规则配置文件不存在: {self.config_path}")
                # 使用默认规则
                self._load_default_rules()
        except Exception as e:
            print(f"⚠ 加载规则配置失败: {e}")
            self._load_default_rules()
    
    def _load_builtin_plugins(self):
        """加载内置规则插件"""
        rules = self.rules_config.get("rules", {})
        
        # 加载情绪驱动规则
        emotion_rules = rules.get("emotion_driven", {})
        for rule_name, rule_config in emotion_rules.items():
            plugin = EmotionRulePlugin({**rule_config, "name": rule_name})
            self.plugins.append(plugin)
        
        # 加载主题驱动规则
        topic_rules = rules.get("topic_driven", {})
        for rule_name, rule_config in topic_rules.items():
            plugin = TopicRulePlugin({**rule_config, "name": rule_name})
            self.plugins.append(plugin)
        
        # 加载场景驱动规则
        scenario_rules = rules.get("scenario_driven", {})
        for rule_name, rule_config in scenario_rules.items():
            plugin = ScenarioRulePlugin({**rule_config, "name": rule_name})
            self.plugins.append(plugin)
    
    def _load_custom_plugins(self):
        """加载自定义插件"""
        plugins_config = self.rules_config.get("plugins", {})
        enabled_plugins = plugins_config.get("enabled", [])
        custom_rules = plugins_config.get("custom_rules", [])
        
        for custom_rule in custom_rules:
            if custom_rule.get("name") in enabled_plugins:
                # 根据类型创建插件
                rule_type = custom_rule.get("type", "topic")
                if rule_type == "emotion":
                    plugin = EmotionRulePlugin(custom_rule)
                elif rule_type == "scenario":
                    plugin = ScenarioRulePlugin(custom_rule)
                else:
                    plugin = TopicRulePlugin(custom_rule)
                
                self.custom_plugins.append(plugin)
                self.plugins.append(plugin)
    
    def _load_default_rules(self):
        """加载默认规则（降级方案）"""
        # 这里可以定义一些基本的默认规则
        pass
    
    def register_plugin(self, plugin: RulePlugin):
        """注册自定义插件"""
        self.custom_plugins.append(plugin)
        self.plugins.append(plugin)
        print(f"✓ 注册自定义规则插件: {plugin.name}")
    
    def evaluate_switching(self, user_input: str, emotion_analysis: dict,
                          current_time: str, context: dict = None) -> Optional[Dict]:
        """
        评估是否需要切换角色
        
        Returns:
            None: 不需要切换
            Dict: 需要切换，包含目标角色和原因
        """
        matched_rules = []
        
        # 评估所有规则插件
        for plugin in self.plugins:
            result = plugin.check_condition(user_input, emotion_analysis, current_time, context)
            if result:
                matched_rules.append((plugin.priority, result))
        
        if not matched_rules:
            return None
        
        # 按优先级排序，选择优先级最高的规则
        matched_rules.sort(key=lambda x: x[0], reverse=True)
        best_match = matched_rules[0][1]
        
        # 选择目标角色（如果有多个，选择第一个）
        target_role = best_match["target_roles"][0] if best_match["target_roles"] else None
        
        return {
            "should_switch": True,
            "target_role": target_role,
            "target_roles": best_match["target_roles"],
            "priority": best_match.get("priority", 0.5),
            "reason": best_match.get("reason", "规则触发"),
            "matched_rules": [r[1] for r in matched_rules]
        }

# 全局规则引擎实例
_switching_engine = None

def get_switching_engine() -> RoleSwitchingEngine:
    """获取角色切换引擎实例（单例）"""
    global _switching_engine
    if _switching_engine is None:
        _switching_engine = RoleSwitchingEngine()
    return _switching_engine

