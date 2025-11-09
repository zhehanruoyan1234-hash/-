"""
批量角色分类优化模块
使用真正的批量推理（model(**batch_inputs)），并行tokenization，共享模型实例
优化：使用公共工具模块、结果缓存、批量处理优化
"""
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForSequenceClassification
import threading
import hashlib
import json

# 导入公共工具模块
from model_utils import setup_transformers_cache, get_device

# 方案1：直接使用 DebertaV3Tokenizer（避免 AutoTokenizer 的自动转换问题）
# 注意：transformers 库中可能没有 DebertaV3Tokenizer，需要尝试不同的导入方式
try:
    # 尝试直接导入 DebertaV3Tokenizer
    from transformers import DebertaV3Tokenizer
    DEBERTA_V3_TOKENIZER_AVAILABLE = True
except ImportError:
    try:
        # 尝试从 deberta 模块导入
        from transformers.models.deberta_v2 import DebertaV2Tokenizer
        # DeBERTa-v3 使用与 v2 相同的 tokenizer 结构
        DebertaV3Tokenizer = DebertaV2Tokenizer
        DEBERTA_V3_TOKENIZER_AVAILABLE = True
        print("✓ 使用 DebertaV2Tokenizer 作为 DeBERTa-v3 的 tokenizer")
    except ImportError:
        # 如果都不可用，回退到 AutoTokenizer with use_fast=False
        from transformers import AutoTokenizer
        DebertaV3Tokenizer = AutoTokenizer
        DEBERTA_V3_TOKENIZER_AVAILABLE = False
        print("⚠ DebertaV3Tokenizer 不可用，使用 AutoTokenizer 作为备选")

# 全局模型和tokenizer（共享实例）
_model_instance = None
_tokenizer_instance = None
_model_lock = threading.Lock()

# 结果缓存（优化4：性能优化）
_result_cache: Dict[str, List[Dict]] = {}
_cache_max_size = 1000  # 最大缓存条目数
_cache_lock = threading.Lock()

def _get_cache_key(texts: List[str], labels: List[str]) -> str:
    """生成缓存键"""
    content = json.dumps({"texts": texts, "labels": labels}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_model_and_tokenizer():
    """获取共享的模型和tokenizer实例（懒加载）
    优化1：使用公共工具模块设置缓存
    优化2：与pipeline共享模型实例（通过返回tokenizer供pipeline使用）
    """
    global _model_instance, _tokenizer_instance
    
    with _model_lock:
        if _model_instance is None or _tokenizer_instance is None:
            try:
                model_name = "microsoft/deberta-v3-base"
                device = get_device()
                
                # 优化1：使用公共工具模块设置缓存
                cache_dir = setup_transformers_cache()
                
                # 方案1：直接使用 DebertaV3Tokenizer，避免 AutoTokenizer 的自动转换问题
                if DEBERTA_V3_TOKENIZER_AVAILABLE:
                    # 直接使用 DebertaV3Tokenizer（或 DebertaV2Tokenizer），不尝试转换
                    _tokenizer_instance = DebertaV3Tokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        trust_remote_code=False  # 不需要远程代码
                    )
                else:
                    # 备选方案：使用 AutoTokenizer，但强制使用慢速 tokenizer
                    _tokenizer_instance = DebertaV3Tokenizer.from_pretrained(
                        model_name,
                        use_fast=False,  # 强制使用慢速 tokenizer
                        cache_dir=cache_dir,
                        trust_remote_code=False
                    )
                
                # 加载model
                _model_instance = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,  # 二元分类（用于zero-shot）
                    cache_dir=cache_dir
                )
                _model_instance.to(device)
                _model_instance.eval()
                
                print(f"✓ 批量角色分类模型加载成功 (DeBERTa-v3-base on {device}, 缓存目录: {cache_dir})")
            except Exception as e:
                print(f"⚠ 加载批量角色分类模型失败: {e}")
                return None, None
        
        return _model_instance, _tokenizer_instance

def batch_tokenize(texts: List[str], labels: List[str]) -> Dict:
    """
    批量tokenize文本和标签
    
    Returns:
        tokenized inputs字典
    """
    model, tokenizer = get_model_and_tokenizer()
    if model is None or tokenizer is None:
        return None
    
    # 构建候选标签对（每个文本对每个标签）
    # zero-shot分类需要构建 "文本 [SEP] 标签" 的格式
    batch_texts = []
    batch_labels = []
    
    for text in texts:
        for label in labels:
            # 构建zero-shot输入格式
            # 格式: "文本。这个文本是关于 [标签] 的吗？"
            candidate_text = f"{text}. This text is about {label}."
            batch_texts.append(candidate_text)
            batch_labels.append(label)
    
    # 批量tokenize
    try:
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 移动到设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return {
            "inputs": inputs,
            "batch_labels": batch_labels,
            "texts_per_label": len(labels),
            "total_texts": len(texts)
        }
    except Exception as e:
        print(f"⚠ 批量tokenize失败: {e}")
        return None

def batch_classify_batch(texts: List[str], labels: List[str]) -> Optional[List[Dict]]:
    """
    真正的批量分类（使用model(**batch_inputs)）
    优化4：添加结果缓存，避免重复计算
    
    Args:
        texts: 待分类文本列表
        labels: 标签列表
    
    Returns:
        每个文本的分类结果列表，失败返回None
    """
    # 优化4：检查缓存
    cache_key = _get_cache_key(texts, labels)
    with _cache_lock:
        if cache_key in _result_cache:
            return _result_cache[cache_key]
    
    model, tokenizer = get_model_and_tokenizer()
    if model is None or tokenizer is None:
        return None
    
    try:
        # 批量tokenize
        tokenized = batch_tokenize(texts, labels)
        if tokenized is None:
            return None
        
        inputs = tokenized["inputs"]
        texts_per_label = tokenized["texts_per_label"]
        total_texts = tokenized["total_texts"]
        
        # 批量推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # 获取预测概率（使用softmax）
        probs = torch.softmax(logits, dim=-1)
        # 对于zero-shot，我们使用正类概率
        scores = probs[:, 1].cpu().tolist()
        
        # 重组结果：每个文本对应所有标签的分数
        results = []
        for i in range(total_texts):
            text_scores = []
            for j in range(len(labels)):
                idx = i * len(labels) + j
                text_scores.append((labels[j], scores[idx]))
            
            # 按分数排序
            text_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 转换为dict格式（兼容pipeline输出）
            results.append({
                "labels": [label for label, _ in text_scores],
                "scores": [score for _, score in text_scores]
            })
        
        # 优化4：保存到缓存
        with _cache_lock:
            if len(_result_cache) >= _cache_max_size:
                # 简单的FIFO策略：删除最旧的条目
                oldest_key = next(iter(_result_cache))
                del _result_cache[oldest_key]
            _result_cache[cache_key] = results
        
        return results
    
    except Exception as e:
        print(f"⚠ 批量分类失败: {e}")
        return None

def parallel_tokenize(texts: List[str], labels: List[str], max_workers: int = 4) -> List[Dict]:
    """
    并行tokenization（使用线程池）
    注意：由于GIL限制，CPU-bound任务并行化效果有限
    但可以用于I/O密集型任务
    """
    # 实际上，tokenization主要是CPU-bound，线程池效果有限
    # 这里保留接口，但实际使用批量tokenize
    return batch_classify_batch(texts, labels)

# 降级方案：使用pipeline（如果批量推理失败）
def fallback_pipeline_classify(texts: List[str], labels: List[str], classifier) -> List[Dict]:
    """使用pipeline进行分类（降级方案）"""
    if classifier is None:
        return None
    
    results = []
    for text in texts:
        try:
            result = classifier(text, labels)
            results.append(result)
        except Exception as e:
            print(f"⚠ Pipeline分类失败: {e}")
            # 返回默认结果
            results.append({
                "labels": labels,
                "scores": [1.0 / len(labels)] * len(labels)
            })
    
    return results

