"""
Tools Module - Reusable tool utilities for agents

This module provides common tool patterns and utilities.

Key Functions:
    - create_search_tool(): Generic search tool factory
    - create_weather_tool(): Weather lookup tool factory
    - create_calculator_tool(): Math calculation tool
    - create_memory_access_tool(): Tool to read from agent memory
    - create_memory_update_tool(): Tool to write to agent memory

Author: AI Toolkit Team
Version: 1.0.0
"""

from .tool_utils import (
    create_search_tool,
    create_weather_tool,
    create_calculator_tool,
    create_memory_access_tool,
    create_memory_update_tool,
    wrap_tool_with_error_handler,
)

__all__ = [
    'create_search_tool',
    'create_weather_tool',
    'create_calculator_tool',
    'create_memory_access_tool',
    'create_memory_update_tool',
    'wrap_tool_with_error_handler',
]
