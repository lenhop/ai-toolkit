"""
Streaming Toolkit

Simple utilities for handling streaming output from AI models.
Based on official LangChain streaming patterns.

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/streaming/overview

Components:
    - StreamHandler: Convenience wrapper for model.stream()

Quick Start:
    >>> from ai_toolkit.streaming import StreamHandler
    >>> from ai_toolkit.models import ModelManager
    >>> 
    >>> manager = ModelManager()
    >>> model = manager.create_model("deepseek")
    >>> handler = StreamHandler(model)
    >>> 
    >>> # Simple token streaming
    >>> for token in handler.stream_tokens("Tell me a joke"):
    ...     print(token, end="", flush=True)
    >>> 
    >>> # Or use model.stream() directly (even simpler!)
    >>> for chunk in model.stream("Tell me a joke"):
    ...     print(chunk.content, end="", flush=True)
    >>> 
    >>> # Need to accumulate? Just use a list
    >>> chunks = []
    >>> for token in handler.stream_tokens("Hello!"):
    ...     print(token, end="", flush=True)
    ...     chunks.append(token)
    >>> final = "".join(chunks)
    >>> 
    >>> # Need callbacks? Use LangChain's BaseCallbackHandler
    >>> from langchain_core.callbacks import BaseCallbackHandler
    >>> 
    >>> class MyCallback(BaseCallbackHandler):
    ...     def on_llm_new_token(self, token, **kwargs):
    ...         print(f"[{token}]", end="")
    >>> 
    >>> for chunk in model.stream("Hi!", config={"callbacks": [MyCallback()]}):
    ...     pass

Design Principles:
    - Keep it simple: Use model.stream() directly when possible
    - StreamHandler is optional: Just a convenience wrapper
    - Use standard Python: Lists for accumulation, LangChain callbacks for custom logic
"""

from .stream_handler import StreamHandler

__all__ = ['StreamHandler']
