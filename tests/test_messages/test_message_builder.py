"""
Test Message Builder

Tests for ai_toolkit.messages.message_builder module:
- MessageBuilder class and methods

Based on examples: 15_message_base.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

# Test imports
try:
    from ai_toolkit.messages import MessageBuilder
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_message_builder_basic():
    """Test basic MessageBuilder functionality."""
    print("\n" + "=" * 80)
    print("Test 1: MessageBuilder Basic Operations")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        builder = MessageBuilder()
        
        # Test: Add system message
        builder.add_system("You are a helpful assistant")
        print("✓ System message added")
        
        # Test: Add human message
        builder.add_human("Hello!")
        print("✓ Human message added")
        
        # Test: Add AI message
        builder.add_ai("Hi! How can I help?")
        print("✓ AI message added")
        
        # Test: Build
        messages = builder.build()
        print(f"✓ Built {len(messages)} messages")
        
        # Test: Count
        print(f"✓ Message count: {builder.count()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_message_builder_conversation():
    """Test MessageBuilder conversation building."""
    print("\n" + "=" * 80)
    print("Test 2: MessageBuilder Conversation Building")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        builder = MessageBuilder()
        
        # Test: Add conversation
        builder.add_conversation([
            ("Hello!", "Hi! How can I help?"),
            ("What is 2+2?", "2+2 equals 4")
        ])
        
        print("✓ Conversation added")
        print(f"✓ Total messages: {len(builder)}")
        
        # Test: Build and verify
        messages = builder.build()
        print(f"✓ Built {len(messages)} messages")
        
        # Verify message types
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)
        assert isinstance(messages[3], AIMessage)
        print("✓ Message types verified")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_message_builder_tool_message():
    """Test MessageBuilder with tool messages."""
    print("\n" + "=" * 80)
    print("Test 3: MessageBuilder with Tool Messages")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        builder = MessageBuilder()
        
        builder.add_system("You are a helpful assistant")
        builder.add_human("What is 2+2?")
        builder.add_tool("Result: 4", tool_call_id="call_123", name="calculator")
        builder.add_ai("The answer is 4")
        
        messages = builder.build()
        print(f"✓ Built {len(messages)} messages")
        
        # Verify tool message
        tool_msg = messages[2]
        assert isinstance(tool_msg, ToolMessage)
        print("✓ Tool message verified")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_message_builder_clear():
    """Test MessageBuilder clear functionality."""
    print("\n" + "=" * 80)
    print("Test 4: MessageBuilder Clear")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        builder = MessageBuilder()
        builder.add_system("Test")
        builder.add_human("Hello")
        
        print(f"✓ Messages before clear: {len(builder)}")
        
        builder.clear()
        
        print(f"✓ Messages after clear: {len(builder)}")
        assert len(builder) == 0
        print("✓ Clear verified")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Message Builder")
    print("=" * 80)
    
    test_message_builder_basic()
    test_message_builder_conversation()
    test_message_builder_tool_message()
    test_message_builder_clear()
    
    print("\n" + "=" * 80)
    print("Message Builder Tests Completed")
    print("=" * 80)
