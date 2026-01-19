"""
Message Toolkit for LangChain

This module provides utilities for working with LangChain message types:
- SystemMessage: Initial instructions that prime the model's behavior
- HumanMessage: User input and interactions
- AIMessage: Model output (automatically created by model)
- ToolMessage: Tool execution results

Overview:
    The Message Toolkit provides MessageBuilder for constructing message lists
    with a clean, fluent API. For validation and conversion, use LangChain's
    native capabilities directly.

Components:
    - MessageBuilder: Fluent interface for building message lists

Features:
    - Clean, fluent API for message construction
    - Support for all LangChain message types
    - Method chaining for readable code
    - Conversation building utilities

Use Cases:
    - Building agent conversations
    - Constructing message histories
    - Creating multi-turn dialogues

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/messages

Example:
    See examples/4.message_toolkit_guide.py for comprehensive demonstrations.

Author: AI Toolkit Team
Version: 2.0.0
"""

from .message_builder import MessageBuilder

__all__ = [
    'MessageBuilder',
]
