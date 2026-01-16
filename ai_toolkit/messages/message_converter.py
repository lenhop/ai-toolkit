"""
Message Converter - Convert between message formats

This module provides utilities for converting messages between different formats.

Overview:
    The MessageConverter class provides methods for transforming LangChain
    messages to and from various formats including dictionaries, OpenAI format,
    strings, and more. It also supports message merging and cloning.

Key Classes:
    - MessageConverter: Converter for transforming messages between formats

Key Methods:
    - to_dict(): Convert message to dictionary
    - from_dict(): Convert dictionary to message
    - to_openai_format(): Convert to OpenAI API format
    - from_openai_format(): Convert from OpenAI API format
    - extract_content(): Extract content strings only
    - merge_messages(): Merge multiple message lists
    - clone_message(): Create a copy of a message
    - to_string(): Convert messages to single string

Usage Example:
    >>> from ai_toolkit.messages import MessageConverter
    >>> from langchain_core.messages import HumanMessage
    >>> 
    >>> # Convert to dictionary
    >>> msg = HumanMessage(content="Hello!", name="Alice")
    >>> dict_msg = MessageConverter.to_dict(msg)
    >>> print(dict_msg)
    {'content': 'Hello!', 'role': 'user', 'name': 'Alice'}
    >>> 
    >>> # Convert from dictionary
    >>> data = {'role': 'user', 'content': 'Hello!'}
    >>> message = MessageConverter.from_dict(data)
    >>> print(type(message).__name__)
    HumanMessage
    >>> 
    >>> # Merge message lists
    >>> history = [SystemMessage(content="You are helpful")]
    >>> new_msgs = [HumanMessage(content="Hello!")]
    >>> merged = MessageConverter.merge_messages(history, new_msgs)
    >>> print(len(merged))
    2

Features:
    - Dictionary conversion (to/from)
    - OpenAI API format conversion
    - Content extraction
    - Message list merging
    - Message cloning (deep copy)
    - String concatenation
    - Format transformation

Supported Formats:
    - Dictionary: {'role': 'user', 'content': '...'}
    - OpenAI: Standard OpenAI API message format
    - String: Concatenated text with separators
    - Content list: List of content strings only

Role Mapping:
    - SystemMessage <-> 'system'
    - HumanMessage <-> 'user'
    - AIMessage <-> 'assistant'
    - ToolMessage <-> 'tool'

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/messages

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Dict, Any
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage
)


class MessageConverter:
    """
    Converter for transforming messages between formats.
    
    Supports conversion to/from dictionaries, OpenAI format, etc.
    """
    
    @staticmethod
    def to_dict(message: BaseMessage) -> Dict[str, Any]:
        """
        Convert a message to dictionary format.
        
        Args:
            message: The message to convert
        
        Returns:
            Dictionary representation
        
        Example:
            >>> msg = HumanMessage(content="Hello!")
            >>> dict_msg = MessageConverter.to_dict(msg)
            >>> print(dict_msg)
            {'role': 'user', 'content': 'Hello!'}
        """
        result = {'content': message.content}
        
        # Add role
        if isinstance(message, SystemMessage):
            result['role'] = 'system'
        elif isinstance(message, HumanMessage):
            result['role'] = 'user'
        elif isinstance(message, AIMessage):
            result['role'] = 'assistant'
        elif isinstance(message, ToolMessage):
            result['role'] = 'tool'
            result['tool_call_id'] = message.tool_call_id
        
        # Add optional fields
        if hasattr(message, 'name') and message.name:
            result['name'] = message.name
        
        return result
    
    @staticmethod
    def to_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Convert multiple messages to dictionary format.
        
        Args:
            messages: List of messages to convert
        
        Returns:
            List of dictionary representations
        
        Example:
            >>> dicts = MessageConverter.to_dicts(messages)
        """
        return [MessageConverter.to_dict(msg) for msg in messages]
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> BaseMessage:
        """
        Convert a dictionary to a message.
        
        Args:
            data: Dictionary with 'role' and 'content' keys
        
        Returns:
            LangChain message object
        
        Example:
            >>> data = {'role': 'user', 'content': 'Hello!'}
            >>> msg = MessageConverter.from_dict(data)
        """
        role = data.get('role', '').lower()
        content = data.get('content', '')
        name = data.get('name')
        
        if role == 'system':
            return SystemMessage(content=content, name=name)
        elif role == 'user':
            return HumanMessage(content=content, name=name)
        elif role == 'assistant':
            return AIMessage(content=content, name=name)
        elif role == 'tool':
            tool_call_id = data.get('tool_call_id', '')
            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=name
            )
        else:
            raise ValueError(f"Unknown role: {role}")
    
    @staticmethod
    def from_dicts(data_list: List[Dict[str, Any]]) -> List[BaseMessage]:
        """
        Convert multiple dictionaries to messages.
        
        Args:
            data_list: List of dictionaries
        
        Returns:
            List of LangChain messages
        
        Example:
            >>> data = [
            ...     {'role': 'system', 'content': 'You are helpful'},
            ...     {'role': 'user', 'content': 'Hello!'}
            ... ]
            >>> messages = MessageConverter.from_dicts(data)
        """
        return [MessageConverter.from_dict(d) for d in data_list]
    
    @staticmethod
    def to_openai_format(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Convert messages to OpenAI API format.
        
        Args:
            messages: List of messages to convert
        
        Returns:
            List of dictionaries in OpenAI format
        
        Example:
            >>> openai_msgs = MessageConverter.to_openai_format(messages)
        """
        return MessageConverter.to_dicts(messages)
    
    @staticmethod
    def from_openai_format(
        data_list: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """
        Convert from OpenAI API format to messages.
        
        Args:
            data_list: List of dictionaries in OpenAI format
        
        Returns:
            List of LangChain messages
        
        Example:
            >>> messages = MessageConverter.from_openai_format(openai_msgs)
        """
        return MessageConverter.from_dicts(data_list)
    
    @staticmethod
    def to_string(
        messages: List[BaseMessage],
        separator: str = "\n\n"
    ) -> str:
        """
        Convert messages to a single string.
        
        Args:
            messages: List of messages
            separator: String to separate messages
        
        Returns:
            Concatenated string
        
        Example:
            >>> text = MessageConverter.to_string(messages)
        """
        parts = []
        for msg in messages:
            msg_type = type(msg).__name__.replace('Message', '')
            parts.append(f"[{msg_type}] {msg.content}")
        return separator.join(parts)
    
    @staticmethod
    def extract_content(messages: List[BaseMessage]) -> List[str]:
        """
        Extract just the content from messages.
        
        Args:
            messages: List of messages
        
        Returns:
            List of content strings
        
        Example:
            >>> contents = MessageConverter.extract_content(messages)
            >>> for content in contents:
            ...     print(content)
        """
        return [msg.content for msg in messages]
    
    @staticmethod
    def merge_messages(
        *message_lists: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        Merge multiple message lists.
        
        Args:
            *message_lists: Variable number of message lists
        
        Returns:
            Combined message list
        
        Example:
            >>> history = [SystemMessage(...), HumanMessage(...)]
            >>> new_msgs = [HumanMessage(...)]
            >>> all_msgs = MessageConverter.merge_messages(history, new_msgs)
        """
        result = []
        for msg_list in message_lists:
            result.extend(msg_list)
        return result
    
    @staticmethod
    def clone_message(message: BaseMessage) -> BaseMessage:
        """
        Create a copy of a message.
        
        Args:
            message: Message to clone
        
        Returns:
            Cloned message
        
        Example:
            >>> original = HumanMessage(content="Hello!")
            >>> copy = MessageConverter.clone_message(original)
        """
        if isinstance(message, SystemMessage):
            return SystemMessage(
                content=message.content,
                name=getattr(message, 'name', None)
            )
        elif isinstance(message, HumanMessage):
            return HumanMessage(
                content=message.content,
                name=getattr(message, 'name', None)
            )
        elif isinstance(message, AIMessage):
            return AIMessage(
                content=message.content,
                name=getattr(message, 'name', None)
            )
        elif isinstance(message, ToolMessage):
            return ToolMessage(
                content=message.content,
                tool_call_id=message.tool_call_id,
                name=getattr(message, 'name', None)
            )
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
    
    @staticmethod
    def clone_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Create copies of multiple messages.
        
        Args:
            messages: List of messages to clone
        
        Returns:
            List of cloned messages
        
        Example:
            >>> copies = MessageConverter.clone_messages(messages)
        """
        return [MessageConverter.clone_message(msg) for msg in messages]
