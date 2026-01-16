"""
Message Builder - Create LangChain messages easily

This module provides a fluent interface for building LangChain messages.

Overview:
    The MessageBuilder class provides a clean, chainable API for constructing
    message lists. It supports all four LangChain message types: SystemMessage,
    HumanMessage, AIMessage, and ToolMessage.

Key Classes:
    - MessageBuilder: Fluent interface for building message lists
    
Key Functions:
    - create_system_message(): Create a SystemMessage
    - create_human_message(): Create a HumanMessage
    - create_ai_message(): Create an AIMessage
    - create_tool_message(): Create a ToolMessage

Features:
    - Fluent interface with method chaining
    - Support for all LangChain message types
    - Convenient conversation building
    - Utility methods for message management
    - Type-safe message construction

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/messages

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage
)


class MessageBuilder:
    """
    Builder class for creating LangChain messages.
    
    Provides a fluent interface for constructing message lists.
    
    Example:
        >>> builder = MessageBuilder()
        >>> messages = (builder
        ...     .add_system("You are a helpful assistant")
        ...     .add_human("Hello!")
        ...     .add_ai("Hi! How can I help?")
        ...     .build())
    """
    
    def __init__(self):
        """Initialize the message builder."""
        self._messages: List[BaseMessage] = []
    
    def add_system(
        self,
        content: str,
        name: Optional[str] = None,
        **kwargs
    ) -> 'MessageBuilder':
        """
        Add a SystemMessage to the message list.
        
        SystemMessage sets the agent's behavior and personality.
        Use at the start of conversation to define capabilities.
        
        Args:
            content: The system message content
            name: Optional name for the message
            **kwargs: Additional message attributes
        
        Returns:
            Self for method chaining
        
        Example:
            >>> builder.add_system("You are a helpful AI assistant")
        """
        message = SystemMessage(content=content, name=name, **kwargs)
        self._messages.append(message)
        return self
    
    def add_human(
        self,
        content: str,
        name: Optional[str] = None,
        **kwargs
    ) -> 'MessageBuilder':
        """
        Add a HumanMessage to the message list.
        
        HumanMessage represents user input.
        Use for every user question or request.
        
        Args:
            content: The user message content
            name: Optional name for the message (e.g., username)
            **kwargs: Additional message attributes
        
        Returns:
            Self for method chaining
        
        Example:
            >>> builder.add_human("What is 2 + 2?", name="Alice")
        """
        message = HumanMessage(content=content, name=name, **kwargs)
        self._messages.append(message)
        return self
    
    def add_ai(
        self,
        content: str,
        name: Optional[str] = None,
        **kwargs
    ) -> 'MessageBuilder':
        """
        Add an AIMessage to the message list.
        
        AIMessage represents model output.
        Usually created automatically by the model, but can be added manually
        for conversation history or testing.
        
        Args:
            content: The AI response content
            name: Optional name for the message
            **kwargs: Additional message attributes
        
        Returns:
            Self for method chaining
        
        Example:
            >>> builder.add_ai("2 + 2 equals 4")
        """
        message = AIMessage(content=content, name=name, **kwargs)
        self._messages.append(message)
        return self
    
    def add_tool(
        self,
        content: str,
        tool_call_id: str,
        name: Optional[str] = None,
        **kwargs
    ) -> 'MessageBuilder':
        """
        Add a ToolMessage to the message list.
        
        ToolMessage represents tool execution results.
        Used to provide tool output back to the agent.
        
        Args:
            content: The tool execution result
            tool_call_id: ID of the tool call this responds to
            name: Optional tool name
            **kwargs: Additional message attributes
        
        Returns:
            Self for method chaining
        
        Example:
            >>> builder.add_tool("Result: 42", tool_call_id="call_123", name="calculator")
        """
        message = ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            **kwargs
        )
        self._messages.append(message)
        return self
    
    def add_message(self, message: BaseMessage) -> 'MessageBuilder':
        """
        Add a pre-constructed message to the list.
        
        Args:
            message: A LangChain message object
        
        Returns:
            Self for method chaining
        
        Example:
            >>> msg = SystemMessage(content="You are helpful")
            >>> builder.add_message(msg)
        """
        self._messages.append(message)
        return self
    
    def add_messages(self, messages: List[BaseMessage]) -> 'MessageBuilder':
        """
        Add multiple pre-constructed messages to the list.
        
        Args:
            messages: List of LangChain message objects
        
        Returns:
            Self for method chaining
        
        Example:
            >>> msgs = [SystemMessage(content="..."), HumanMessage(content="...")]
            >>> builder.add_messages(msgs)
        """
        self._messages.extend(messages)
        return self
    
    def add_conversation(
        self,
        exchanges: List[tuple[str, str]]
    ) -> 'MessageBuilder':
        """
        Add multiple human-AI message pairs.
        
        Convenient for adding conversation history.
        
        Args:
            exchanges: List of (human_message, ai_message) tuples
        
        Returns:
            Self for method chaining
        
        Example:
            >>> builder.add_conversation([
            ...     ("Hello!", "Hi! How can I help?"),
            ...     ("What is 2+2?", "2+2 equals 4")
            ... ])
        """
        for human_msg, ai_msg in exchanges:
            self.add_human(human_msg)
            self.add_ai(ai_msg)
        return self
    
    def clear(self) -> 'MessageBuilder':
        """
        Clear all messages from the builder.
        
        Returns:
            Self for method chaining
        """
        self._messages.clear()
        return self
    
    def build(self) -> List[BaseMessage]:
        """
        Build and return the message list.
        
        Returns:
            List of LangChain messages
        
        Example:
            >>> messages = builder.build()
        """
        return self._messages.copy()
    
    def count(self) -> int:
        """
        Get the number of messages in the builder.
        
        Returns:
            Number of messages
        """
        return len(self._messages)
    
    def get_last(self) -> Optional[BaseMessage]:
        """
        Get the last message in the builder.
        
        Returns:
            Last message or None if empty
        """
        return self._messages[-1] if self._messages else None
    
    def __len__(self) -> int:
        """Get the number of messages."""
        return len(self._messages)
    
    def __repr__(self) -> str:
        """String representation of the builder."""
        return f"MessageBuilder(messages={len(self._messages)})"


def create_system_message(content: str, **kwargs) -> SystemMessage:
    """
    Create a SystemMessage.
    
    Args:
        content: The system message content
        **kwargs: Additional message attributes
    
    Returns:
        SystemMessage instance
    
    Example:
        >>> msg = create_system_message("You are a helpful assistant")
    """
    return SystemMessage(content=content, **kwargs)


def create_human_message(content: str, **kwargs) -> HumanMessage:
    """
    Create a HumanMessage.
    
    Args:
        content: The user message content
        **kwargs: Additional message attributes
    
    Returns:
        HumanMessage instance
    
    Example:
        >>> msg = create_human_message("What is 2 + 2?")
    """
    return HumanMessage(content=content, **kwargs)


def create_ai_message(content: str, **kwargs) -> AIMessage:
    """
    Create an AIMessage.
    
    Args:
        content: The AI response content
        **kwargs: Additional message attributes
    
    Returns:
        AIMessage instance
    
    Example:
        >>> msg = create_ai_message("2 + 2 equals 4")
    """
    return AIMessage(content=content, **kwargs)


def create_tool_message(
    content: str,
    tool_call_id: str,
    **kwargs
) -> ToolMessage:
    """
    Create a ToolMessage.
    
    Args:
        content: The tool execution result
        tool_call_id: ID of the tool call
        **kwargs: Additional message attributes
    
    Returns:
        ToolMessage instance
    
    Example:
        >>> msg = create_tool_message("Result: 42", tool_call_id="call_123")
    """
    return ToolMessage(content=content, tool_call_id=tool_call_id, **kwargs)
