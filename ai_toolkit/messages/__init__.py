"""
Message Toolkit for LangChain

This module provides utilities for working with LangChain message types:
- SystemMessage: Initial instructions that prime the model's behavior
- HumanMessage: User input and interactions
- AIMessage: Model output (automatically created by model)
- ToolMessage: Tool execution results

Overview:
    The Message Toolkit provides four main components for managing LangChain
    messages: MessageBuilder for construction, MessageFormatter for display,
    MessageValidator for validation, and MessageConverter for format conversion.

Components:
    - MessageBuilder: Fluent interface for building message lists
    - MessageFormatter: Display and format messages in various styles
    - MessageValidator: Validate message structures and flows
    - MessageConverter: Convert between different message formats

Features:
    - Clean, fluent API for message construction
    - Multiple display styles with emoji icons
    - Comprehensive validation with error messages
    - Format conversion (dict, OpenAI, string)
    - Message statistics and analysis
    - Type-safe operations
    - Full support for all LangChain message types

Use Cases:
    - Building agent conversations
    - Validating user input
    - Converting for API calls
    - Debugging message flows
    - Analyzing conversation statistics
    - Format transformation

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/messages

Example:
    See examples/4.message_toolkit_guide.py for comprehensive demonstrations.

Author: AI Toolkit Team
Version: 1.0.0
"""

from .message_builder import MessageBuilder
from .message_formatter import MessageFormatter
from .message_validator import MessageValidator
from .message_converter import MessageConverter

__all__ = [
    'MessageBuilder',
    'MessageFormatter',
    'MessageValidator',
    'MessageConverter',
]
