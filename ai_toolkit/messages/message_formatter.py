"""
Message Formatter - Format and display LangChain messages

This module provides utilities for formatting messages for display.
"""

from typing import List, Optional
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage
)


class MessageFormatter:
    """
    Formatter for displaying LangChain messages.
    
    Provides various formatting options for message display.
    """
    
    @staticmethod
    def format_message(
        message: BaseMessage,
        include_type: bool = True,
        include_name: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """
        Format a single message for display.
        
        Args:
            message: The message to format
            include_type: Whether to include message type
            include_name: Whether to include message name
            max_length: Maximum content length (truncate if longer)
        
        Returns:
            Formatted message string
        
        Example:
            >>> msg = HumanMessage(content="Hello!")
            >>> print(MessageFormatter.format_message(msg))
            HumanMessage: Hello!
        """
        msg_type = type(message).__name__
        content = message.content
        
        # Truncate if needed
        if max_length and len(content) > max_length:
            content = content[:max_length] + "..."
        
        # Build formatted string
        parts = []
        if include_type:
            parts.append(msg_type)
        
        if include_name and hasattr(message, 'name') and message.name:
            parts.append(f"[{message.name}]")
        
        if parts:
            return f"{' '.join(parts)}: {content}"
        else:
            return content
    
    @staticmethod
    def format_messages(
        messages: List[BaseMessage],
        numbered: bool = True,
        include_type: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """
        Format multiple messages for display.
        
        Args:
            messages: List of messages to format
            numbered: Whether to number the messages
            include_type: Whether to include message types
            max_length: Maximum content length per message
        
        Returns:
            Formatted messages string
        
        Example:
            >>> messages = [
            ...     SystemMessage(content="You are helpful"),
            ...     HumanMessage(content="Hello!")
            ... ]
            >>> print(MessageFormatter.format_messages(messages))
        """
        lines = []
        for i, msg in enumerate(messages, 1):
            formatted = MessageFormatter.format_message(
                msg,
                include_type=include_type,
                max_length=max_length
            )
            if numbered:
                lines.append(f"{i}. {formatted}")
            else:
                lines.append(formatted)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_conversation(
        messages: List[BaseMessage],
        show_system: bool = True
    ) -> str:
        """
        Format messages as a conversation.
        
        Uses emoji icons for different message types.
        
        Args:
            messages: List of messages to format
            show_system: Whether to show system messages
        
        Returns:
            Formatted conversation string
        
        Example:
            >>> print(MessageFormatter.format_conversation(messages))
            🔧 System: You are a helpful assistant
            👤 User: Hello!
            🤖 AI: Hi! How can I help?
        """
        icons = {
            'SystemMessage': '🔧 System',
            'HumanMessage': '👤 User',
            'AIMessage': '🤖 AI',
            'ToolMessage': '🔨 Tool'
        }
        
        lines = []
        for msg in messages:
            msg_type = type(msg).__name__
            
            # Skip system messages if requested
            if not show_system and msg_type == 'SystemMessage':
                continue
            
            icon = icons.get(msg_type, '💬')
            lines.append(f"{icon}: {msg.content}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_as_dict(message: BaseMessage) -> dict:
        """
        Format a message as a dictionary.
        
        Args:
            message: The message to format
        
        Returns:
            Dictionary representation
        
        Example:
            >>> msg = HumanMessage(content="Hello!")
            >>> MessageFormatter.format_as_dict(msg)
            {'type': 'HumanMessage', 'content': 'Hello!', 'role': 'user'}
        """
        result = {
            'type': type(message).__name__,
            'content': message.content,
        }
        
        # Add role
        if isinstance(message, SystemMessage):
            result['role'] = 'system'
        elif isinstance(message, HumanMessage):
            result['role'] = 'user'
        elif isinstance(message, AIMessage):
            result['role'] = 'assistant'
        elif isinstance(message, ToolMessage):
            result['role'] = 'tool'
        
        # Add optional fields
        if hasattr(message, 'name') and message.name:
            result['name'] = message.name
        
        if isinstance(message, ToolMessage):
            result['tool_call_id'] = message.tool_call_id
        
        return result
    
    @staticmethod
    def format_as_dicts(messages: List[BaseMessage]) -> List[dict]:
        """
        Format multiple messages as dictionaries.
        
        Args:
            messages: List of messages to format
        
        Returns:
            List of dictionary representations
        
        Example:
            >>> dicts = MessageFormatter.format_as_dicts(messages)
        """
        return [MessageFormatter.format_as_dict(msg) for msg in messages]
    
    @staticmethod
    def get_message_stats(messages: List[BaseMessage]) -> dict:
        """
        Get statistics about a message list.
        
        Args:
            messages: List of messages
        
        Returns:
            Dictionary with statistics
        
        Example:
            >>> stats = MessageFormatter.get_message_stats(messages)
            >>> print(f"Total: {stats['total']}, Human: {stats['human']}")
        """
        stats = {
            'total': len(messages),
            'system': 0,
            'human': 0,
            'ai': 0,
            'tool': 0,
            'total_chars': 0
        }
        
        for msg in messages:
            msg_type = type(msg).__name__
            stats['total_chars'] += len(msg.content)
            
            if msg_type == 'SystemMessage':
                stats['system'] += 1
            elif msg_type == 'HumanMessage':
                stats['human'] += 1
            elif msg_type == 'AIMessage':
                stats['ai'] += 1
            elif msg_type == 'ToolMessage':
                stats['tool'] += 1
        
        return stats
    
    @staticmethod
    def print_messages(
        messages: List[BaseMessage],
        style: str = 'conversation'
    ):
        """
        Print messages to console.
        
        Args:
            messages: List of messages to print
            style: Display style ('conversation', 'numbered', 'simple')
        
        Example:
            >>> MessageFormatter.print_messages(messages, style='conversation')
        """
        if style == 'conversation':
            print(MessageFormatter.format_conversation(messages))
        elif style == 'numbered':
            print(MessageFormatter.format_messages(messages, numbered=True))
        elif style == 'simple':
            print(MessageFormatter.format_messages(messages, numbered=False, include_type=False))
        else:
            print(MessageFormatter.format_messages(messages))
