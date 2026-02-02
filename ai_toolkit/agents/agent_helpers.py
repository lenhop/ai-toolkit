"""
Agent Helper Functions - Simplified agent creation

This module provides convenience functions for creating LangChain agents
with common configurations based on the practice examples.

Functions:
    - create_agent_with_tools(): Create agent with tools and common defaults
    - create_agent_with_memory(): Create agent with memory/checkpointer
    - create_streaming_agent(): Create agent optimized for streaming
    - create_structured_output_agent(): Create agent with structured output

Based on: examples/practice/13_agent_base.py, 14_agent_advanced.py
LangChain Version: 1.0
Python Version: 3.11

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Optional, Any, Dict, Union
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


def create_agent_with_tools(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    middleware: Optional[List[Any]] = None,
    **kwargs
) -> Any:
    """
    Create an agent with tools and common defaults.
    
    Simplified wrapper around create_agent() with sensible defaults.
    Based on examples/practice/13_agent_base.py
    
    Args:
        model: LangChain chat model
        tools: List of tools for the agent
        system_prompt: Optional system prompt (default: helpful assistant)
        middleware: Optional list of middleware functions
        **kwargs: Additional arguments passed to create_agent()
    
    Returns:
        Configured agent
    
    Example:
        >>> from ai_toolkit.models import ModelManager
        >>> from langchain.tools import tool
        >>> 
        >>> @tool
        >>> def search(query: str) -> str:
        ...     '''Search for information.'''
        ...     return f"Results for: {query}"
        >>> 
        >>> manager = ModelManager()
        >>> model = manager.create_model("deepseek")
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search],
        ...     system_prompt="You are a helpful research assistant."
        ... )
        >>> 
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "Search for AI news"}]
        ... })
    
    Note:
        - Default system prompt: "You are a helpful assistant. Be concise and accurate."
        - Automatically handles tool errors with graceful fallback
        - Compatible with LangChain 1.0 agent API
    """
    # Default system prompt
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Be concise and accurate."
    
    # Create agent with provided configuration
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        **kwargs
    )
    
    return agent


def create_agent_with_memory(
    model: BaseChatModel,
    tools: List[BaseTool],
    checkpointer_type: str = "inmemory",
    db_uri: Optional[str] = None,
    system_prompt: Optional[str] = None,
    middleware: Optional[List[Any]] = None,
    **kwargs
) -> Any:
    """
    Create an agent with memory/checkpointer pre-configured.
    
    Automatically sets up checkpointer for conversation persistence.
    Based on examples/practice/17_memory_base.py
    
    Args:
        model: LangChain chat model
        tools: List of tools for the agent
        checkpointer_type: Type of checkpointer ("inmemory" or "postgres")
        db_uri: Database URI (required for postgres)
        system_prompt: Optional system prompt
        middleware: Optional list of middleware functions
        **kwargs: Additional arguments passed to create_agent()
    
    Returns:
        Configured agent with memory
    
    Example:
        >>> # In-memory checkpointer (development)
        >>> agent = create_agent_with_memory(
        ...     model=model,
        ...     tools=[search_tool],
        ...     checkpointer_type="inmemory"
        ... )
        >>> 
        >>> # PostgreSQL checkpointer (production)
        >>> agent = create_agent_with_memory(
        ...     model=model,
        ...     tools=[search_tool],
        ...     checkpointer_type="postgres",
        ...     db_uri="postgresql://user:pass@localhost:5432/db"
        ... )
        >>> 
        >>> # Use with thread_id for conversation isolation
        >>> config = {"configurable": {"thread_id": "user-123"}}
        >>> result = agent.invoke(
        ...     {"messages": [{"role": "user", "content": "Hi, I'm Alice"}]},
        ...     config
        ... )
    
    Note:
        - In-memory: Lost on restart, good for development
        - PostgreSQL: Persistent, good for production
        - Requires thread_id in config for conversation isolation
    """
    from ai_toolkit.memory import MemoryManager
    
    # Create checkpointer
    memory_manager = MemoryManager()
    checkpointer = memory_manager.create_checkpointer(
        type=checkpointer_type,
        db_uri=db_uri
    )
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Be concise and accurate."
    
    # Create agent with checkpointer
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=middleware or [],
        **kwargs
    )
    
    return agent


def create_streaming_agent(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    middleware: Optional[List[Any]] = None,
    **kwargs
) -> Any:
    """
    Create an agent optimized for streaming responses.
    
    Pre-configured for streaming with common patterns.
    Based on examples/practice/18_streaming_base.py
    
    Args:
        model: LangChain chat model (should support streaming)
        tools: List of tools for the agent
        system_prompt: Optional system prompt
        middleware: Optional list of middleware functions
        **kwargs: Additional arguments passed to create_agent()
    
    Returns:
        Configured streaming agent
    
    Example:
        >>> agent = create_streaming_agent(
        ...     model=model,
        ...     tools=[get_weather],
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> 
        >>> # Stream agent progress
        >>> for chunk in agent.stream(
        ...     {"messages": [{"role": "user", "content": "What's the weather?"}]},
        ...     stream_mode="updates"
        ... ):
        ...     for step, data in chunk.items():
        ...         print(f"Step: {step}")
        ...         print(f"Content: {data['messages'][-1].content_blocks}")
        >>> 
        >>> # Stream LLM tokens
        >>> for token, metadata in agent.stream(
        ...     {"messages": [{"role": "user", "content": "Hello!"}]},
        ...     stream_mode="messages"
        ... ):
        ...     print(token.content_blocks, end="", flush=True)
    
    Note:
        - Model must support streaming (check model.profile)
        - Use stream_mode="updates" for agent progress
        - Use stream_mode="messages" for LLM tokens
        - Use stream_mode="custom" for custom updates
    """
    # Default system prompt
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Be concise and accurate."
    
    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        **kwargs
    )
    
    return agent


def create_structured_output_agent(
    model: BaseChatModel,
    tools: List[BaseTool],
    schema: Union[BaseModel, Dict[str, Any]],
    strategy: str = "tool",
    system_prompt: Optional[str] = None,
    middleware: Optional[List[Any]] = None,
    **kwargs
) -> Any:
    """
    Create an agent with structured output pre-configured.
    
    Automatically sets up response_format for structured output.
    Based on examples/practice/14_agent_advanced.py, 19_output_structure.py
    
    Args:
        model: LangChain chat model
        tools: List of tools for the agent
        schema: Pydantic model or dict schema for output structure
        strategy: Output strategy ("tool" or "provider")
                 - "tool": ToolStrategy (works with all models)
                 - "provider": ProviderStrategy (uses native JSON mode)
        system_prompt: Optional system prompt
        middleware: Optional list of middleware functions
        **kwargs: Additional arguments passed to create_agent()
    
    Returns:
        Configured agent with structured output
    
    Example:
        >>> from pydantic import BaseModel, Field
        >>> 
        >>> class ContactInfo(BaseModel):
        ...     name: str = Field(description="Person's name")
        ...     email: str = Field(description="Email address")
        ...     phone: str = Field(description="Phone number")
        >>> 
        >>> # Using ToolStrategy (works with all models)
        >>> agent = create_structured_output_agent(
        ...     model=model,
        ...     tools=[search],
        ...     schema=ContactInfo,
        ...     strategy="tool"
        ... )
        >>> 
        >>> # Using ProviderStrategy (native JSON mode)
        >>> agent = create_structured_output_agent(
        ...     model=model,
        ...     tools=[search],
        ...     schema=ContactInfo,
        ...     strategy="provider",
        ...     system_prompt="Respond in JSON format."
        ... )
        >>> 
        >>> result = agent.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": "Extract: John Doe, john@example.com, (555) 123-4567"
        ...     }]
        ... })
        >>> print(result["structured_response"])
        >>> # ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
    
    Note:
        - ToolStrategy: Works with all models, uses tool calling
        - ProviderStrategy: Requires model with native JSON mode (e.g., Qwen)
        - For Qwen with ProviderStrategy, include "json" in system prompt
        - Access structured output via result["structured_response"]
    """
    from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
    
    # Create response format based on strategy
    if strategy == "tool":
        response_format = ToolStrategy(schema)
    elif strategy == "provider":
        response_format = ProviderStrategy(schema)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'tool' or 'provider'.")
    
    # Default system prompt
    if system_prompt is None:
        if strategy == "provider":
            # For provider strategy, hint at JSON format
            system_prompt = "You are a helpful assistant that responds in JSON format."
        else:
            system_prompt = "You are a helpful assistant. Be concise and accurate."
    
    # Create agent with structured output
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        response_format=response_format,
        middleware=middleware or [],
        **kwargs
    )
    
    return agent
