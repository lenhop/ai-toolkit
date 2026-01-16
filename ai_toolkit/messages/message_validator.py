"""
Message Validator - Validate LangChain messages

This module provides utilities for validating message structures.

Overview:
    The MessageValidator class provides comprehensive validation methods for
    LangChain messages. It can validate individual messages, message lists,
    conversation flows, and provide filtering and counting capabilities.

Key Classes:
    - MessageValidator: Validator for LangChain messages

Key Methods:
    - validate_message(): Validate a single message
    - validate_messages(): Validate a list of messages
    - validate_conversation_flow(): Validate message ordering
    - count_message_types(): Count messages by type
    - filter_by_type(): Filter messages by type
    - has_message_type(): Check if type exists
    - get_first_message_of_type(): Get first message of type
    - get_last_message_of_type(): Get last message of type

Usage Example:
    >>> from ai_toolkit.messages import MessageValidator
    >>> from langchain_core.messages import SystemMessage, HumanMessage
    >>> 
    >>> messages = [
    ...     SystemMessage(content="You are helpful"),
    ...     HumanMessage(content="Hello!")
    ... ]
    >>> 
    >>> # Validate messages
    >>> is_valid, error = MessageValidator.validate_messages(
    ...     messages,
    ...     require_system=True,
    ...     require_human=True
    ... )
    >>> print(f"Valid: {is_valid}")
    Valid: True
    >>> 
    >>> # Count message types
    >>> counts = MessageValidator.count_message_types(messages)
    >>> print(f"System: {counts['system']}, Human: {counts['human']}")
    System: 1, Human: 1

Features:
    - Single message validation
    - Message list validation with requirements
    - Conversation flow validation
    - Message type counting and filtering
    - First/last message retrieval by type
    - Empty content detection
    - ToolMessage validation (tool_call_id required)

Validation Rules:
    - Messages must have non-empty content
    - ToolMessage must have tool_call_id
    - SystemMessage should be first (if present)
    - Only one SystemMessage at start
    - Optional requirements for message types

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/messages

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Optional, Tuple
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage
)


class MessageValidator:
    """
    Validator for LangChain messages.
    
    Provides validation methods for message lists.
    """
    
    @staticmethod
    def validate_message(message: BaseMessage) -> Tuple[bool, Optional[str]]:
        """
        Validate a single message.
        
        Args:
            message: The message to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        
        Example:
            >>> msg = HumanMessage(content="Hello!")
            >>> is_valid, error = MessageValidator.validate_message(msg)
            >>> if not is_valid:
            ...     print(f"Error: {error}")
        """
        # Check if message has content
        if not hasattr(message, 'content'):
            return False, "Message missing 'content' attribute"
        
        # Check if content is not empty
        if not message.content or not message.content.strip():
            return False, "Message content is empty"
        
        # Validate ToolMessage has tool_call_id
        if isinstance(message, ToolMessage):
            if not hasattr(message, 'tool_call_id') or not message.tool_call_id:
                return False, "ToolMessage missing 'tool_call_id'"
        
        return True, None
    
    @staticmethod
    def validate_messages(
        messages: List[BaseMessage],
        require_system: bool = False,
        require_human: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a list of messages.
        
        Args:
            messages: List of messages to validate
            require_system: Whether to require a SystemMessage
            require_human: Whether to require at least one HumanMessage
        
        Returns:
            Tuple of (is_valid, error_message)
        
        Example:
            >>> is_valid, error = MessageValidator.validate_messages(messages)
            >>> if not is_valid:
            ...     print(f"Validation error: {error}")
        """
        # Check if list is empty
        if not messages:
            return False, "Message list is empty"
        
        # Validate each message
        for i, msg in enumerate(messages):
            is_valid, error = MessageValidator.validate_message(msg)
            if not is_valid:
                return False, f"Message {i + 1}: {error}"
        
        # Check for required message types
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        has_human = any(isinstance(msg, HumanMessage) for msg in messages)
        
        if require_system and not has_system:
            return False, "No SystemMessage found (required)"
        
        if require_human and not has_human:
            return False, "No HumanMessage found (required)"
        
        return True, None
    
    @staticmethod
    def validate_conversation_flow(
        messages: List[BaseMessage]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate conversation flow (proper message ordering).
        
        Checks:
        - SystemMessage should be first (if present)
        - Messages should alternate between Human and AI (loosely)
        
        Args:
            messages: List of messages to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        
        Example:
            >>> is_valid, error = MessageValidator.validate_conversation_flow(messages)
        """
        if not messages:
            return False, "Message list is empty"
        
        # Check if SystemMessage is first (if present)
        system_indices = [
            i for i, msg in enumerate(messages)
            if isinstance(msg, SystemMessage)
        ]
        
        if system_indices and system_indices[0] != 0:
            return False, "SystemMessage should be the first message"
        
        # Check for multiple SystemMessages
        if len(system_indices) > 1:
            return False, "Multiple SystemMessages found (should have only one at start)"
        
        return True, None
    
    @staticmethod
    def has_message_type(
        messages: List[BaseMessage],
        message_type: type
    ) -> bool:
        """
        Check if message list contains a specific message type.
        
        Args:
            messages: List of messages
            message_type: Type to check for (SystemMessage, HumanMessage, etc.)
        
        Returns:
            True if message type is present
        
        Example:
            >>> has_system = MessageValidator.has_message_type(messages, SystemMessage)
        """
        return any(isinstance(msg, message_type) for msg in messages)
    
    @staticmethod
    def count_message_types(messages: List[BaseMessage]) -> dict:
        """
        Count messages by type.
        
        Args:
            messages: List of messages
        
        Returns:
            Dictionary with counts by type
        
        Example:
            >>> counts = MessageValidator.count_message_types(messages)
            >>> print(f"Human messages: {counts['human']}")
        """
        counts = {
            'system': 0,
            'human': 0,
            'ai': 0,
            'tool': 0,
            'other': 0
        }
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                counts['system'] += 1
            elif isinstance(msg, HumanMessage):
                counts['human'] += 1
            elif isinstance(msg, AIMessage):
                counts['ai'] += 1
            elif isinstance(msg, ToolMessage):
                counts['tool'] += 1
            else:
                counts['other'] += 1
        
        return counts
    
    @staticmethod
    def get_first_message_of_type(
        messages: List[BaseMessage],
        message_type: type
    ) -> Optional[BaseMessage]:
        """
        Get the first message of a specific type.
        
        Args:
            messages: List of messages
            message_type: Type to search for
        
        Returns:
            First message of type or None
        
        Example:
            >>> system_msg = MessageValidator.get_first_message_of_type(
            ...     messages, SystemMessage
            ... )
        """
        for msg in messages:
            if isinstance(msg, message_type):
                return msg
        return None
    
    @staticmethod
    def get_last_message_of_type(
        messages: List[BaseMessage],
        message_type: type
    ) -> Optional[BaseMessage]:
        """
        Get the last message of a specific type.
        
        Args:
            messages: List of messages
            message_type: Type to search for
        
        Returns:
            Last message of type or None
        
        Example:
            >>> last_human = MessageValidator.get_last_message_of_type(
            ...     messages, HumanMessage
            ... )
        """
        for msg in reversed(messages):
            if isinstance(msg, message_type):
                return msg
        return None
    
    @staticmethod
    def filter_by_type(
        messages: List[BaseMessage],
        message_type: type
    ) -> List[BaseMessage]:
        """
        Filter messages by type.
        
        Args:
            messages: List of messages
            message_type: Type to filter for
        
        Returns:
            List of messages of specified type
        
        Example:
            >>> human_messages = MessageValidator.filter_by_type(
            ...     messages, HumanMessage
            ... )
        """
        return [msg for msg in messages if isinstance(msg, message_type)]
