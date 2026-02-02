"""
AI Toolkit - Comprehensive LangChain utilities for building AI agents

This toolkit provides reusable components for building LangChain-based AI applications.

Modules:
    - models: Model management and creation
    - agents: Agent creation helpers and middleware
    - tools: Common tool patterns and factories
    - memory: Memory management and checkpointers
    - messages: Message building utilities
    - streaming: Streaming utilities
    - parsers: Output parsing and structured output
    - rag: RAG (Retrieval Augmented Generation) utilities
    - chroma: Chroma vector database toolkit
    - prompts: Prompt management
    - tokens: Token counting and optimization
    - errors: Error handling and retry logic
    - config: Configuration management
    - utils: General utilities

Quick Start:
    >>> from ai_toolkit.models import ModelManager
    >>> from ai_toolkit.agents import create_agent_with_tools
    >>> from ai_toolkit.tools import create_search_tool
    >>> 
    >>> # Create model
    >>> manager = ModelManager()
    >>> model = manager.create_model("deepseek")
    >>> 
    >>> # Create tools
    >>> search = create_search_tool()
    >>> 
    >>> # Create agent
    >>> agent = create_agent_with_tools(model, tools=[search])
    >>> 
    >>> # Use agent
    >>> result = agent.invoke({
    ...     "messages": [{"role": "user", "content": "Search for AI news"}]
    ... })

LangChain Version: 1.0
Python Version: 3.11

Author: AI Toolkit Team
Version: 1.0.0
"""

__version__ = "1.0.0"

# Core modules
from . import models
from . import agents
from . import tools
from . import memory
from . import messages
from . import streaming
from . import parsers
from . import rag
from . import chroma
from . import prompts
from . import tokens
from . import errors
from . import config
from . import utils

# Commonly used imports
from .models import ModelManager
from .memory import MemoryManager
from .messages import MessageBuilder

__all__ = [
    # Modules
    'models',
    'agents',
    'tools',
    'memory',
    'messages',
    'streaming',
    'parsers',
    'rag',
    'chroma',
    'prompts',
    'tokens',
    'errors',
    'config',
    'utils',
    # Common classes
    'ModelManager',
    'MemoryManager',
    'MessageBuilder',
]
