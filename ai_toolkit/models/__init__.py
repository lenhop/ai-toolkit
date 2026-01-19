"""
Models Module - Simplified model management.

Usage:
    >>> from ai_toolkit.models import ModelManager
    >>> 
    >>> manager = ModelManager()
    >>> model = manager.create_model("deepseek")
    >>> response = model.invoke("Hello!")
"""

from .model_manager import ModelManager

__all__ = ['ModelManager']
