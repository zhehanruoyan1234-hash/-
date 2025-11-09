"""
角色偏好分析模块（优化版）
使用 Gemini API + LangGraph状态机 + 动态切换机制
改进：
1. 使用可配置的规则引擎（role_switching_engine）
2. 使用SQLite数据库存储历史（role_history_db）
3. 使用 Gemini API 进行角色分析（已删除 DeBERTa-v3 方案，避免 SentencePiece 依赖问题）
4. 记录详细的切换日志（role_switching_logger）
"""
import json
import re
from typing import List, Dict, Optional, TypedDict
from datetime import datetime, timedelta
# 已删除：import torch, from transformers import pipeline, import numpy as np
# 原因：不再使用 DeBERTa-v3 模型，直接使用 Gemini API

# 导入新模块
from role_switching_engine import get_switching_engine
from role_history_db import save_role_history, get_role_history, get_role_history_cached
from role_switching_logger import log_role_switch, log_role_analysis
# 已删除：from batch_role_classifier import batch_classify_batch, get_model_and_tokenizer
# 已删除：from model_utils import setup_transformers_cache, get_device_id
# 原因：DeBERTa-v3 需要 SentencePiece 库，环境不支持，直接使用 Gemini API

# LangGraph 导入
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠ LangGraph未安装，将使用传统角色分析")

# 角色定义
VALID_ROLES = ["朋友", "导师", "倾听者", "玩伴", "安慰者", "伙伴", "陪伴者"]
VALID_STYLES = ["轻松幽默", "温和关怀", "专业理性", "活泼热情", "平静内敛", "随性自然"]
VALID_TONES = ["随和", "正式", "亲密", "中性"]

# 已删除 get_role_classifier 函数
# 原因：DeBERTa-v3 需要 SentencePiece 库，环境不支持，直接使用 Gemini API

# LangGraph 角色状态定义
class RoleState(TypedDict):
    """角色分析状态"""
    user_id: str
    user_input: str
    emotion_analysis: dict
    conversation_history: List[dict]
    current_role: Optional[dict]
    role_history: List[dict]
    context_window: int  # 上下文窗口大小
    should_switch: bool  # 是否应该切换角色
    switch_reason: str  # 切换原因

# LangGraph 角色工作流
class RoleAnalysisWorkflow:
    """基于LangGraph的角色分析工作流（带动态切换）"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph = None
        self.checkpointer = None
        if LANGGRAPH_AVAILABLE:
            self._build_workflow()
    
    def _build_workflow(self):
        """构建LangGraph角色分析工作流"""
        try:
            # 创建检查点保存器（用于持久化角色历史）
            self.checkpointer = MemorySaver()
            
            # 创建状态图
            workflow = StateGraph(RoleState)
            
            # 添加节点
            workflow.add_node("context_analysis", self._context_analysis_node)
            workflow.add_node("role_classification", self._role_classification_node)
            workflow.add_node("consistency_check", self._consistency_check_node)
            workflow.add_node("dynamic_switch", self._dynamic_switch_node)
            workflow.add_node("update_history", self._update_history_node)
            
            # 定义流程
            workflow.set_entry_point("context_analysis")
            workflow.add_edge("context_analysis", "role_classification")
            workflow.add_edge("role_classification", "consistency_check")
            workflow.add_conditional_edges(
                "consistency_check",
                self._should_switch_role,
                {
                    "switch": "dynamic_switch",
                    "keep": "update_history"
                }
            )
            workflow.add_edge("dynamic_switch", "update_history")
            workflow.add_edge("update_history", END)
            
            # 编译图
            self.graph = workflow.compile(checkpointer=self.checkpointer)
            print("✓ LangGraph角色分析工作流构建成功")
        except Exception as e:
            print(f"⚠ 构建LangGraph角色工作流失败: {e}")
            self.graph = None
    
    def _context_analysis_node(self, state: RoleState) -> RoleState:
        """上下文分析节点：整合最近N轮对话"""
        conversation_history = state.get("conversation_history", [])
        context_window = state.get("context_window", 3)
        
        # 整合最近N轮对话的文本
        recent_texts = []
        for conv in conversation_history[-context_window:]:
            user_input = conv.get("user_input", "")
            if user_input:
                recent_texts.append(user_input)
        
        # 如果当前输入不为空，添加进去
        current_input = state.get("user_input", "")
        if current_input:
            recent_texts.append(current_input)
        
        # 合并为上下文文本
        context_text = " ".join(recent_texts) if recent_texts else current_input
        state["context_text"] = context_text
        state["recent_texts"] = recent_texts
        
        return state
    
    def _role_classification_node(self, state: RoleState) -> RoleState:
        """角色分类节点（直接使用 Gemini API，已删除 DeBERTa-v3 方案）"""
        recent_texts = state.get("recent_texts", [])
        context_text = state.get("context_text", "")
        emotion_analysis = state.get("emotion_analysis", {})
        
        # 直接使用 Gemini API（已删除 DeBERTa-v3 批量推理和 pipeline 方案）
        try:
            state["current_role"] = self._fallback_role_analysis(context_text, emotion_analysis)
        except Exception as e:
            print(f"角色分类失败: {e}，使用默认值")
            state["current_role"] = {
                "role_type": "朋友",
                "communication_style": "温和关怀",
                "tone": "随和",
                "confidence": {"role": 0.5, "style": 0.5, "tone": 0.5}
            }
        
        return state
    
    def _consistency_check_node(self, state: RoleState) -> RoleState:
        """一致性检查节点：比较当前角色与历史角色分布"""
        current_role = state.get("current_role", {})
        role_history = state.get("role_history", [])
        
        if not role_history:
            state["should_switch"] = False
            return state
        
        # 计算历史角色分布
        role_counts = {}
        style_counts = {}
        tone_counts = {}
        
        for historical_role in role_history[-10:]:  # 最近10次
            role_type = historical_role.get("role_type", "朋友")
            style = historical_role.get("communication_style", "温和关怀")
            tone = historical_role.get("tone", "随和")
            
            role_counts[role_type] = role_counts.get(role_type, 0) + 1
            style_counts[style] = style_counts.get(style, 0) + 1
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
        
        # 计算当前角色与历史分布的差异
        current_role_type = current_role.get("role_type", "朋友")
        current_style = current_role.get("communication_style", "温和关怀")
        current_tone = current_role.get("tone", "随和")
        
        total = len(role_history[-10:])
        if total == 0:
            state["should_switch"] = False
            return state
        
        # 计算偏差度（0-1，越大越偏离）
        role_deviation = 1.0 - (role_counts.get(current_role_type, 0) / total)
        style_deviation = 1.0 - (style_counts.get(current_style, 0) / total)
        tone_deviation = 1.0 - (tone_counts.get(current_tone, 0) / total)
        
        avg_deviation = (role_deviation + style_deviation + tone_deviation) / 3.0
        
        # 如果平均偏差超过阈值（0.5），认为需要切换
        should_switch = avg_deviation > 0.5
        state["should_switch"] = should_switch
        state["deviation_scores"] = {
            "role": role_deviation,
            "style": style_deviation,
            "tone": tone_deviation,
            "average": avg_deviation
        }
        
        return state
    
    def _should_switch_role(self, state: RoleState) -> str:
        """判断是否应该切换角色（使用规则引擎）"""
        should_switch = state.get("should_switch", False)
        
        # 使用规则引擎评估切换
        user_input = state.get("user_input", "")
        emotion_analysis = state.get("emotion_analysis", {})
        
        # 获取当前时间
        from app import get_beijing_time
        current_time = get_beijing_time()
        
        # 使用规则引擎评估
        switching_engine = get_switching_engine()
        switch_result = switching_engine.evaluate_switching(
            user_input, emotion_analysis, current_time, {"state": state}
        )
        
        if switch_result and switch_result.get("should_switch"):
            state["switch_reason"] = switch_result.get("reason", "规则触发")
            state["switch_result"] = switch_result
            return "switch"
        
        # 如果一致性检查也建议切换
        if should_switch:
            state["switch_reason"] = "一致性偏差触发"
            return "switch"
        
        return "keep"
    
    def _dynamic_switch_node(self, state: RoleState) -> RoleState:
        """动态切换节点：根据规则引擎结果调整角色"""
        current_role = state.get("current_role", {})
        switch_result = state.get("switch_result", {})
        
        # 保存旧角色（用于日志）
        old_role = current_role.copy()
        
        # 根据规则引擎结果调整角色
        if switch_result:
            target_role = switch_result.get("target_role")
            target_roles = switch_result.get("target_roles", [])
            
            if target_role:
                current_role["role_type"] = target_role
                # 根据角色类型设置默认风格
                role_style_map = {
                    "安慰者": "温和关怀",
                    "倾听者": "平静内敛",
                    "玩伴": "活泼热情",
                    "导师": "专业理性",
                    "朋友": "轻松幽默",
                    "伙伴": "随性自然",
                    "陪伴者": "温和关怀"
                }
                current_role["communication_style"] = role_style_map.get(target_role, "温和关怀")
        
        state["current_role"] = current_role
        state["switch_applied"] = True
        
        # 记录角色切换日志
        user_id = state.get("user_id", "")
        switch_reason = state.get("switch_reason", "")
        emotion_analysis = state.get("emotion_analysis", {})
        user_input = state.get("user_input", "")
        
        if old_role.get("role_type") != current_role.get("role_type"):
            log_role_switch(
                user_id, old_role, current_role, switch_reason,
                current_role.get("confidence", {}), emotion_analysis, user_input
            )
        
        return state
    
    def _update_history_node(self, state: RoleState) -> RoleState:
        """更新历史节点：记录角色变化趋势（使用SQLite数据库）"""
        current_role = state.get("current_role", {})
        user_id = state.get("user_id", "")
        emotion_analysis = state.get("emotion_analysis", {})
        user_input = state.get("user_input", "")
        switch_reason = state.get("switch_reason", "")
        
        # 保存到SQLite数据库
        role_data = {
            **current_role,
            "user_input": user_input[:200],  # 限制长度
            "emotion_type": emotion_analysis.get("emotion_type", ""),
            "emotion_intensity": emotion_analysis.get("emotion_intensity", ""),
            "switch_reason": switch_reason
        }
        save_role_history(user_id, role_data)
        
        # 从数据库获取历史（带LRU缓存）
        role_history = get_role_history_cached(user_id, limit=100)
        state["role_history"] = role_history
        
        # 记录角色分析日志（非切换情况）
        log_role_analysis(user_id, current_role, emotion_analysis)
        
        return state
    
    def _fallback_role_analysis(self, user_input: str, emotion_analysis: dict) -> dict:
        """角色分析：使用 Gemini API（主要方案，已删除 DeBERTa-v3 降级方案）"""
        try:
            from app import client, get_beijing_time
            current_time = get_beijing_time()
            
            role_prompt = f"""分析用户输入，判断用户期望的AI角色类型。

当前时间：{current_time}
用户输入：{user_input}
情绪状态：{emotion_analysis.get('emotion_type', 'neutral')}

请以JSON格式输出：
{{
    "role_type": "朋友/导师/倾听者/玩伴/安慰者/伙伴/陪伴者",
    "communication_style": "轻松幽默/温和关怀/专业理性/活泼热情/平静内敛/随性自然",
    "tone": "随和/正式/亲密/中性"
}}"""

            response = client.models.generate_content(model="gemini-2.5-flash", contents=role_prompt)
            response_text = response.text.strip()
            
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                role_data = json.loads(json_match.group())
            else:
                role_data = json.loads(response_text)
            
            return {
                "role_type": role_data.get("role_type", "朋友"),
                "communication_style": role_data.get("communication_style", "温和关怀"),
                "tone": role_data.get("tone", "随和"),
                "confidence": {"role": 0.7, "style": 0.7, "tone": 0.7}
            }
        except Exception as e:
            print(f"降级角色分析失败: {e}")
            return {
                "role_type": "朋友",
                "communication_style": "温和关怀",
                "tone": "随和",
                "confidence": {"role": 0.5, "style": 0.5, "tone": 0.5}
            }
    
    def analyze_role(self, user_input: str, emotion_analysis: dict, 
                     conversation_history: List[dict] = None) -> dict:
        """分析角色偏好（使用LangGraph工作流）"""
        if not self.graph:
            # 降级到传统方法
            return self._traditional_role_analysis(user_input, emotion_analysis, conversation_history)
        
        try:
            user_id = self.user_id
            config = {"configurable": {"thread_id": user_id}}
            
            # 从SQLite数据库恢复历史（带LRU缓存）
            role_history = get_role_history_cached(user_id, limit=100)
            
            initial_state = {
                "user_id": user_id,
                "user_input": user_input,
                "emotion_analysis": emotion_analysis,
                "conversation_history": conversation_history or [],
                "current_role": None,
                "role_history": role_history,
                "context_window": 3,  # 整合最近3轮对话
                "should_switch": False,
                "switch_reason": ""
            }
            
            result = self.graph.invoke(initial_state, config=config)
            return result.get("current_role", {
                "role_type": "朋友",
                "communication_style": "温和关怀",
                "tone": "随和"
            })
        except Exception as e:
            print(f"LangGraph角色分析失败: {e}，使用传统方法")
            return self._traditional_role_analysis(user_input, emotion_analysis, conversation_history)
    
    def _traditional_role_analysis(self, user_input: str, emotion_analysis: dict,
                                  conversation_history: List[dict] = None) -> dict:
        """传统角色分析方法（直接使用 Gemini API，已删除 DeBERTa-v3 方案）"""
        # 整合上下文
        context_texts = []
        if conversation_history:
            for conv in conversation_history[-3:]:
                context_texts.append(conv.get("user_input", ""))
        context_texts.append(user_input)
        combined_text = " ".join(context_texts)
        
        # 直接使用 Gemini API
        return self._fallback_role_analysis(combined_text, emotion_analysis)

# 全局工作流实例（按用户ID）
_role_workflows = {}

def get_role_workflow(user_id: str) -> RoleAnalysisWorkflow:
    """获取或创建角色分析工作流"""
    if user_id not in _role_workflows:
        _role_workflows[user_id] = RoleAnalysisWorkflow(user_id)
    return _role_workflows[user_id]

def analyze_role_preference_optimized(user_input: str, current_time: str, 
                                      emotion_analysis: dict,
                                      conversation_history: List[dict] = None,
                                      user_id: str = "default_user") -> dict:
    """
    优化的角色偏好分析（使用LangGraph + Gemini API）
    已删除 DeBERTa-v3 方案，直接使用 Gemini API 避免依赖问题
    
    Args:
        user_input: 用户输入
        current_time: 当前时间
        emotion_analysis: 情绪分析结果
        conversation_history: 对话历史
        user_id: 用户ID
    
    Returns:
        角色偏好字典
    """
    workflow = get_role_workflow(user_id)
    result = workflow.analyze_role(user_input, emotion_analysis, conversation_history)
    
    # 添加一致性信息（从数据库获取）
    role_history = get_role_history_cached(user_id, limit=100)
    if len(role_history) > 1:
        recent_roles = [r.get("role_type") for r in role_history[-5:]]
        unique_roles = len(set(recent_roles))
        result["preference_consistency"] = unique_roles <= 2
    else:
        result["preference_consistency"] = True
    
    return result

