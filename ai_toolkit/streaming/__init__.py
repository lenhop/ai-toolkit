"""
Streaming Toolkit

Simple utilities for handling streaming output from AI models.
Based on official LangChain streaming patterns.

Official Documentation:
    https://docs.langchain.com/oss/python/langchain-streaming

Key Concepts:
    - Use model.stream() for direct streaming
    - Use StreamCallback for custom token processing
    - LangChain handles the complexity internally

Example:
    >>> from ai_toolkit.streaming import StreamCallback
    >>> from ai_toolkit.models import ModelManager
    >>> 
    >>> manager = ModelManager()
    >>> model = manager.create_model("deepseek")
    >>> 
    >>> # Simple streaming
    >>> for chunk in model.stream("Tell me a joke"):
    ...     print(chunk.content, end="", flush=True)
    >>> 
    >>> # With callback
    >>> callback = StreamCallback(verbose=True)
    >>> for chunk in model.stream("Hello", config={"callbacks": [callback]}):
    ...     pass
    >>> print(callback.get_accumulated_text())
"""

from .stream_callback import StreamCallback

__all__ = ['StreamCallback']
