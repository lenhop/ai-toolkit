"""
Models Module - Simplified model management.

Usage:
    >>> from ai_toolkit.models import ModelManager
    >>> 
    >>> manager = ModelManager()
    >>> model = manager.create_model("deepseek")
    >>> response = model.invoke("Hello!")

    >>> from ai_toolkit.models import LocalQwenEmbeddings
    >>> embeddings = LocalQwenEmbeddings("/path/to/Qwen3-VL-Embedding-2B")
"""

from .model_manager import ModelManager
from .qwen_embeddings import LocalQwenEmbeddings

__all__ = ['ModelManager', 'LocalQwenEmbeddings']
