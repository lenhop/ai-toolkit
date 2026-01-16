"""Tests for MessageBuilder."""

import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from ai_toolkit.messages.message_builder import (
    MessageBuilder,
    create_system_message,
    create_human_message,
    create_ai_message,
    create_tool_message
)


class TestMessageBuilder:
    """Test MessageBuilder class."""
    
    def test_add_system(self):
        """Test adding system message."""
        builder = MessageBuilder()
        builder.add_system("You are helpful")
        
        messages = builder.build()
        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are helpful"
    
    def test_add_human(self):
        """Test adding human message."""
        builder = MessageBuilder()
        builder.add_human("Hello!", name="Alice")
        
        messages = builder.build()
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello!"
        assert messages[0].name == "Alice"
    
    def test_add_ai(self):
        """Test adding AI message."""
        builder = MessageBuilder()
        builder.add_ai("Hi there!")
        
        messages = builder.build()
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "Hi there!"
    
    def test_add_tool(self):
        """Test adding tool message."""
        builder = MessageBuilder()
        builder.add_tool("Result: 42", tool_call_id="call_123", name="calculator")
        
        messages = builder.build()
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].content == "Result: 42"
        assert messages[0].tool_call_id == "call_123"
        assert messages[0].name == "calculator"
    
    def test_fluent_interface(self):
        """Test fluent interface chaining."""
        builder = MessageBuilder()
        messages = (builder
            .add_system("You are helpful")
            .add_human("Hello!")
            .add_ai("Hi!")
            .build())
        
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)
    
    def test_add_conversation(self):
        """Test adding conversation exchanges."""
        builder = MessageBuilder()
        builder.add_conversation([
            ("Hello!", "Hi!"),
            ("How are you?", "I'm good!")
        ])
        
        messages = builder.build()
        assert len(messages) == 4
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Hello!"
        assert messages[1].content == "Hi!"
    
    def test_clear(self):
        """Test clearing messages."""
        builder = MessageBuilder()
        builder.add_human("Hello!")
        builder.clear()
        
        messages = builder.build()
        assert len(messages) == 0
    
    def test_count(self):
        """Test message count."""
        builder = MessageBuilder()
        assert builder.count() == 0
        
        builder.add_human("Hello!")
        assert builder.count() == 1
        
        builder.add_ai("Hi!")
        assert builder.count() == 2
    
    def test_get_last(self):
        """Test getting last message."""
        builder = MessageBuilder()
        assert builder.get_last() is None
        
        builder.add_human("Hello!")
        last = builder.get_last()
        assert isinstance(last, HumanMessage)
        assert last.content == "Hello!"
    
    def test_len(self):
        """Test __len__ method."""
        builder = MessageBuilder()
        assert len(builder) == 0
        
        builder.add_human("Hello!")
        assert len(builder) == 1


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_system_message(self):
        """Test creating system message."""
        msg = create_system_message("You are helpful")
        assert isinstance(msg, SystemMessage)
        assert msg.content == "You are helpful"
    
    def test_create_human_message(self):
        """Test creating human message."""
        msg = create_human_message("Hello!")
        assert isinstance(msg, HumanMessage)
        assert msg.content == "Hello!"
    
    def test_create_ai_message(self):
        """Test creating AI message."""
        msg = create_ai_message("Hi!")
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hi!"
    
    def test_create_tool_message(self):
        """Test creating tool message."""
        msg = create_tool_message("Result: 42", tool_call_id="call_123")
        assert isinstance(msg, ToolMessage)
        assert msg.content == "Result: 42"
        assert msg.tool_call_id == "call_123"
