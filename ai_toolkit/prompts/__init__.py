"""
Prompt management module for AI Toolkit.

This module provides classes and utilities for managing prompt templates
including loading, rendering, and organizing prompts.
"""

from .prompt_templates import (
    BasePromptTemplate,
    ChatPromptTemplate,
    SystemPromptTemplate,
    FewShotPromptTemplate,
    SimplePromptTemplate,
    create_template,
    detect_template_type
)
from .prompt_loader import PromptLoader
from .prompt_manager import PromptManager

__all__ = [
    # Template classes
    'BasePromptTemplate',
    'ChatPromptTemplate',
    'SystemPromptTemplate',
    'FewShotPromptTemplate',
    'SimplePromptTemplate',
    'create_template',
    'detect_template_type',
    
    # Loader class
    'PromptLoader',
    
    # Manager class
    'PromptManager',
]