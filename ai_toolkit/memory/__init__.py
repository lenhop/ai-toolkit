"""
Memory Management Toolkit for LangChain Agents

This module provides utilities for managing agent memory and conversation history.

Overview:
    The Memory Toolkit provides components for managing short-term memory,
    message trimming, summarization, and custom state management for LangChain agents.

Components:
    - MemoryManager: Main class for memory operations
    - MessageTrimmer: Trim messages to fit context window
    - MessageSummarizer: Summarize long conversations
    - CheckpointerFactory: Create different types of checkpointers

Features:
    - Multiple checkpointer types (InMemory, PostgreSQL)
    - Message trimming strategies
    - Automatic summarization
    - Custom state management
    - Message deletion utilities
    - Dynamic prompt generation

Use Cases:
    - Managing conversation history
    - Preventing context overflow
    - Persistent memory storage
    - Custom agent state
    - Message cleanup

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/short-term-memory

Author: AI Toolkit Team
Version: 1.0.0
"""

from .memory_manager import (
    MemoryManager,
    CheckpointerFactory,
    MessageTrimmer,
    MessageSummarizer,
    create_trimming_middleware,
    create_deletion_middleware,
    create_summarization_middleware,
    create_dynamic_prompt_middleware
)

__all__ = [
    'MemoryManager',
    'CheckpointerFactory',
    'MessageTrimmer',
    'MessageSummarizer',
    'create_trimming_middleware',
    'create_deletion_middleware',
    'create_summarization_middleware',
    'create_dynamic_prompt_middleware',
]
