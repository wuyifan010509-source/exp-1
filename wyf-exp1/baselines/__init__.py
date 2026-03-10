"""
基线方法模块
提供人工描述、IntentGPT、随机压缩等基线
"""
from .random_compress import (
    RandomCompression,
    load_text_baseline,
    load_manual_baseline,
    load_intentgpt_baseline,
    load_unstructured_baseline,
    text_to_profile,
    # 向后兼容的别名
    load_manual,
    load_intentgpt,
    load_compressed
)

__all__ = [
    'RandomCompression',
    'load_text_baseline',
    'load_manual_baseline',
    'load_intentgpt_baseline',
    'load_unstructured_baseline',
    'text_to_profile',
    'load_manual',
    'load_intentgpt',
    'load_compressed'
]
