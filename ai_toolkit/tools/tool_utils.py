"""
Tool Utilities - Common tool patterns and factories

This module provides factory functions for creating common tools
used in the practice examples.

Functions:
    - create_search_tool(): Generic search tool
    - create_weather_tool(): Weather lookup tool
    - create_calculator_tool(): Math calculation tool
    - create_memory_access_tool(): Read from agent memory
    - create_memory_update_tool(): Write to agent memory
    - wrap_tool_with_error_handler(): Add error handling to any tool

Based on: examples/practice/13_agent_base.py, 16_tool_base.py
LangChain Version: 1.0
Python Version: 3.11

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import Callable, Optional, Any, Dict
from langchain.tools import tool, BaseTool, ToolRuntime
from pydantic import BaseModel, Field
from typing import Literal


def create_search_tool(
    search_function: Optional[Callable[[str], str]] = None,
    name: str = "search",
    description: str = "Search for information."
) -> BaseTool:
    """
    Create a generic search tool.
    
    Based on examples/practice/13_agent_base.py
    
    Args:
        search_function: Optional custom search function
                        Signature: (query: str) -> str
                        If None, returns mock results
        name: Tool name (default: "search")
        description: Tool description for the agent
    
    Returns:
        Search tool
    
    Example:
        >>> # Mock search tool (for testing)
        >>> search = create_search_tool()
        >>> 
        >>> # Custom search tool
        >>> def my_search(query: str) -> str:
        ...     # Your search implementation
        ...     return f"Results for: {query}"
        >>> 
        >>> search = create_search_tool(
        ...     search_function=my_search,
        ...     description="Search the knowledge base."
        ... )
        >>> 
        >>> # Use in agent
        >>> agent = create_agent(model, tools=[search])
    
    Note:
        - Default implementation returns mock results
        - Provide custom search_function for real search
        - Tool name must be unique within agent
    """
    if search_function is None:
        # Default mock implementation
        def default_search(query: str) -> str:
            return f"Results for: {query}"
        search_function = default_search
    
    @tool(name, description=description)
    def search_tool(query: str) -> str:
        """Search for information."""
        return search_function(query)
    
    return search_tool


def create_weather_tool(
    weather_function: Optional[Callable[[str], str]] = None,
    name: str = "get_weather",
    description: str = "Get weather information for a location."
) -> BaseTool:
    """
    Create a weather lookup tool.
    
    Based on examples/practice/13_agent_base.py
    
    Args:
        weather_function: Optional custom weather function
                         Signature: (location: str) -> str
                         If None, returns mock weather
        name: Tool name (default: "get_weather")
        description: Tool description for the agent
    
    Returns:
        Weather tool
    
    Example:
        >>> # Mock weather tool (for testing)
        >>> weather = create_weather_tool()
        >>> 
        >>> # Custom weather tool
        >>> def my_weather(location: str) -> str:
        ...     # Your weather API implementation
        ...     return f"Weather in {location}: Sunny, 72°F"
        >>> 
        >>> weather = create_weather_tool(
        ...     weather_function=my_weather,
        ...     description="Get current weather for any city."
        ... )
        >>> 
        >>> # Use in agent
        >>> agent = create_agent(model, tools=[weather])
    
    Note:
        - Default implementation returns mock weather
        - Provide custom weather_function for real weather API
        - Consider adding parameters for units, forecast, etc.
    """
    if weather_function is None:
        # Default mock implementation
        def default_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 72°F"
        weather_function = default_weather
    
    @tool(name, description=description)
    def weather_tool(location: str) -> str:
        """Get weather information for a location."""
        return weather_function(location)
    
    return weather_tool


def create_calculator_tool(
    name: str = "calculator",
    description: str = "Performs arithmetic calculations. Use this for any math problems.",
    safe_mode: bool = True
) -> BaseTool:
    """
    Create a calculator tool for math operations.
    
    Based on examples/practice/16_tool_base.py
    
    Args:
        name: Tool name (default: "calculator")
        description: Tool description for the agent
        safe_mode: If True, restricts to safe operations only (default: True)
    
    Returns:
        Calculator tool
    
    Example:
        >>> # Safe calculator (recommended)
        >>> calc = create_calculator_tool(safe_mode=True)
        >>> 
        >>> # Unrestricted calculator (use with caution)
        >>> calc = create_calculator_tool(safe_mode=False)
        >>> 
        >>> # Use in agent
        >>> agent = create_agent(model, tools=[calc])
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "What is 25 * 4?"}]
        ... })
    
    Warning:
        - safe_mode=False uses eval() which can be dangerous
        - Only disable safe_mode in trusted environments
        - Consider using a proper math parser for production
    
    Note:
        - Safe mode restricts to basic arithmetic operations
        - Supports: +, -, *, /, **, (), numbers
        - Does not support: functions, imports, variables
    """
    @tool(name, description=description)
    def calculator(expression: str) -> str:
        """Evaluate mathematical expressions."""
        try:
            if safe_mode:
                # Restrict to safe characters only
                import re
                if not re.match(r'^[\d\s+\-*/().]+$', expression):
                    return "Error: Expression contains unsafe characters. Use only numbers and operators (+, -, *, /, **, ())."
            
            # Evaluate expression
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return calculator


def create_memory_access_tool(
    memory_key: str = "user_info",
    name: str = "get_memory",
    description: str = "Look up information from memory."
) -> BaseTool:
    """
    Create a tool to read from agent memory/store.
    
    Based on examples/practice/16_tool_base.py
    
    Args:
        memory_key: Key to access in memory store
        name: Tool name (default: "get_memory")
        description: Tool description for the agent
    
    Returns:
        Memory access tool
    
    Example:
        >>> from langgraph.store.memory import InMemoryStore
        >>> from langchain.agents import create_agent
        >>> 
        >>> # Create memory access tool
        >>> get_memory = create_memory_access_tool(
        ...     memory_key="users",
        ...     description="Look up user info by user_id."
        ... )
        >>> 
        >>> # Create store
        >>> store = InMemoryStore()
        >>> 
        >>> # Create agent with store
        >>> agent = create_agent(
        ...     model=model,
        ...     tools=[get_memory],
        ...     store=store
        ... )
        >>> 
        >>> # Agent can now access memory
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "Get user info for user_123"}]
        ... })
    
    Note:
        - Requires agent to be created with store parameter
        - Memory is accessed via runtime.store in tool
        - Use with InMemoryStore or persistent store
    """
    @tool(name, description=description)
    def get_memory(key: str, runtime: ToolRuntime) -> str:
        """Look up information from memory."""
        store = runtime.store
        item = store.get((memory_key,), key)
        return str(item.value) if item else f"No data found for key: {key}"
    
    return get_memory


def create_memory_update_tool(
    memory_key: str = "user_info",
    name: str = "save_memory",
    description: str = "Save information to memory."
) -> BaseTool:
    """
    Create a tool to write to agent memory/store.
    
    Based on examples/practice/16_tool_base.py
    
    Args:
        memory_key: Key to store in memory
        name: Tool name (default: "save_memory")
        description: Tool description for the agent
    
    Returns:
        Memory update tool
    
    Example:
        >>> from langgraph.store.memory import InMemoryStore
        >>> from langchain.agents import create_agent
        >>> 
        >>> # Create memory update tool
        >>> save_memory = create_memory_update_tool(
        ...     memory_key="users",
        ...     description="Save user info with user_id as key."
        ... )
        >>> 
        >>> # Create store
        >>> store = InMemoryStore()
        >>> 
        >>> # Create agent with store
        >>> agent = create_agent(
        ...     model=model,
        ...     tools=[save_memory],
        ...     store=store
        ... )
        >>> 
        >>> # Agent can now save to memory
        >>> result = agent.invoke({
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": "Save user: id=user_123, name=Alice, age=25"
        ...     }]
        ... })
    
    Note:
        - Requires agent to be created with store parameter
        - Memory is updated via runtime.store in tool
        - Data persists for the lifetime of the store
    """
    @tool(name, description=description)
    def save_memory(key: str, value: Dict[str, Any], runtime: ToolRuntime) -> str:
        """Save information to memory."""
        store = runtime.store
        store.put((memory_key,), key, value)
        return f"Successfully saved data for key: {key}"
    
    return save_memory


def wrap_tool_with_error_handler(
    tool: BaseTool,
    error_message: str = "Tool execution failed. Please try again."
) -> BaseTool:
    """
    Wrap a tool with error handling.
    
    Catches exceptions and returns user-friendly error messages.
    
    Args:
        tool: Tool to wrap
        error_message: Error message to return on failure
    
    Returns:
        Wrapped tool with error handling
    
    Example:
        >>> # Create a tool that might fail
        >>> @tool
        >>> def risky_tool(input: str) -> str:
        ...     '''A tool that might fail.'''
        ...     if not input:
        ...         raise ValueError("Input required")
        ...     return f"Processed: {input}"
        >>> 
        >>> # Wrap with error handler
        >>> safe_tool = wrap_tool_with_error_handler(
        ...     risky_tool,
        ...     error_message="Tool failed. Please provide valid input."
        ... )
        >>> 
        >>> # Use in agent
        >>> agent = create_agent(model, tools=[safe_tool])
    
    Note:
        - Prevents agent crashes from tool failures
        - Returns error message instead of raising exception
        - Allows agent to recover and try alternatives
    """
    original_func = tool.func
    
    def wrapped_func(*args, **kwargs):
        try:
            return original_func(*args, **kwargs)
        except Exception as e:
            return f"{error_message} Error: {str(e)}"
    
    # Create new tool with wrapped function
    wrapped_tool = tool.__class__(
        name=tool.name,
        description=tool.description,
        func=wrapped_func,
        args_schema=tool.args_schema
    )
    
    return wrapped_tool
