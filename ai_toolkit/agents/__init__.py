"""
Agents Module - Agent creation and management utilities

This module provides helper functions for creating and configuring LangChain agents.

Key Functions:
    - create_agent_with_tools(): Simplified agent creation
    - create_agent_with_memory(): Agent with memory pre-configured
    - create_streaming_agent(): Agent optimized for streaming
    - create_structured_output_agent(): Agent with structured output

Author: AI Toolkit Team
Version: 1.0.0
"""

from .agent_helpers import (
    create_agent_with_tools,
    create_agent_with_memory,
    create_streaming_agent,
    create_structured_output_agent,
)

from .middleware_utils import (
    # Custom middleware
    create_dynamic_model_selector,
    create_tool_error_handler,
    create_context_based_prompt,
    # Built-in middleware wrappers
    create_summarization_middleware,
    create_human_in_loop_middleware,
    create_model_call_limit_middleware,
    create_tool_call_limit_middleware,
    create_model_fallback_middleware,
    create_pii_middleware,
    create_todo_list_middleware,
    create_llm_tool_selector_middleware,
    create_tool_retry_middleware,
    create_model_retry_middleware,
    create_llm_tool_emulator,
    create_context_editing_middleware,
    create_shell_tool_middleware,
    create_filesystem_search_middleware,
)

__all__ = [
    # Agent helpers
    'create_agent_with_tools',
    'create_agent_with_memory',
    'create_streaming_agent',
    'create_structured_output_agent',
    # Custom middleware
    'create_dynamic_model_selector',
    'create_tool_error_handler',
    'create_context_based_prompt',
    # Built-in middleware wrappers
    'create_summarization_middleware',
    'create_human_in_loop_middleware',
    'create_model_call_limit_middleware',
    'create_tool_call_limit_middleware',
    'create_model_fallback_middleware',
    'create_pii_middleware',
    'create_todo_list_middleware',
    'create_llm_tool_selector_middleware',
    'create_tool_retry_middleware',
    'create_model_retry_middleware',
    'create_llm_tool_emulator',
    'create_context_editing_middleware',
    'create_shell_tool_middleware',
    'create_filesystem_search_middleware',
]
