#!/usr/bin/env python3
"""
Message Toolkit Guide - Working with LangChain Messages

This example demonstrates the AI Toolkit's MessageBuilder for working with
LangChain message types following official documentation:
https://docs.langchain.com/oss/python/langchain/messages

Components Demonstrated:
1. Message Types - SystemMessage, HumanMessage, AIMessage, ToolMessage
2. MessageBuilder - Fluent interface for building message lists

Official Documentation:
https://docs.langchain.com/oss/python/langchain/messages
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Import message toolkit
from ai_toolkit.messages import MessageBuilder


print("\n" + "=" * 80)
print("MESSAGE TOOLKIT GUIDE - LANGCHAIN MESSAGES")
print("=" * 80)
print("\nDemonstrating AI Toolkit's MessageBuilder")
print("Official Docs: https://docs.langchain.com/oss/python/langchain/messages")
print()


# =============================================================================
# 1. MESSAGE TYPES - Understanding the Four Message Types
# =============================================================================
print("=" * 80)
print("1. MESSAGE TYPES - Four Core Types")
print("=" * 80)
print()

# SystemMessage - Sets agent behavior
print("SystemMessage - Sets agent behavior and personality")
system_msg = SystemMessage(content="You are a helpful AI assistant with access to tools.")
print(f"   Content: '{system_msg.content}'")
print(f"   Role: system")
print(f"   Use: At the start of conversation to define capabilities")
print()

# HumanMessage - User input
print("HumanMessage - Represents user input")
human_msg = HumanMessage(content="What is 2 + 2?", name="Alice")
print(f"   Content: '{human_msg.content}'")
print(f"   Name: {human_msg.name}")
print(f"   Role: user")
print(f"   Use: Every user question or request")
print()

# AIMessage - Model output
print("AIMessage - Represents model output")
ai_msg = AIMessage(content="2 + 2 equals 4.")
print(f"   Content: '{ai_msg.content}'")
print(f"   Role: assistant")
print(f"   Use: Model responses (usually auto-generated)")
print()

# ToolMessage - Tool execution results
print("ToolMessage - Tool execution results")
tool_msg = ToolMessage(
    content="Result: 4",
    tool_call_id="call_123",
    name="calculator"
)
print(f"   Content: '{tool_msg.content}'")
print(f"   Tool Call ID: {tool_msg.tool_call_id}")
print(f"   Name: {tool_msg.name}")
print(f"   Role: tool")
print(f"   Use: Provide tool output back to agent")
print()


# =============================================================================
# 2. MESSAGE BUILDER - Fluent Interface for Building Messages
# =============================================================================
print("=" * 80)
print("2. MESSAGE BUILDER - Fluent Interface")
print("=" * 80)
print()

# Example 1: Basic message building
print("Example 1: Basic Message Building")
print("-" * 80)

builder = MessageBuilder()
messages = (builder
    .add_system("You are a helpful math tutor")
    .add_human("What is 5 + 3?")
    .add_ai("5 + 3 equals 8")
    .add_human("What about 10 - 4?")
    .build())

print(f"Built {len(messages)} messages using fluent interface")
for i, msg in enumerate(messages, 1):
    print(f"   {i}. {type(msg).__name__}: {msg.content}")
print()

# Example 2: Adding conversation history
print("Example 2: Adding Conversation History")
print("-" * 80)

builder = MessageBuilder()
builder.add_system("You are a helpful assistant")
builder.add_conversation([
    ("Hello!", "Hi! How can I help you today?"),
    ("What's the weather?", "I don't have access to weather data."),
    ("Thanks anyway", "You're welcome! Let me know if you need anything else.")
])

messages = builder.build()
print(f"Built conversation with {len(messages)} messages")
print(f"   System messages: 1")
print(f"   Human-AI exchanges: 3")
print()

# Example 3: Builder methods
print("Example 3: Builder Utility Methods")
print("-" * 80)

builder = MessageBuilder()
builder.add_system("You are helpful")
builder.add_human("Hello!")

print(f"Message count: {builder.count()}")
print(f"Builder length: {len(builder)}")
print()


# =============================================================================
# 3. WORKING WITH MESSAGES - Using LangChain Directly
# =============================================================================
print("=" * 80)
print("3. WORKING WITH MESSAGES - LangChain Native Operations")
print("=" * 80)
print()

# Example 1: Convert to dictionaries (LangChain native)
print("Example 1: Convert to Dictionaries")
print("-" * 80)

msg = HumanMessage(content="Hello!", name="Alice")
# LangChain messages have dict() method
dict_msg = {"role": "user", "content": msg.content}
if msg.name:
    dict_msg["name"] = msg.name
print(f"Message as dict: {dict_msg}")
print()

# Example 2: Filter messages by type
print("Example 2: Filter Messages by Type")
print("-" * 80)

sample_messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello!", name="Alice"),
    AIMessage(content="Hi Alice!"),
    HumanMessage(content="How are you?", name="Alice"),
]

human_messages = [msg for msg in sample_messages if isinstance(msg, HumanMessage)]
print(f"Found {len(human_messages)} HumanMessages:")
for msg in human_messages:
    print(f"   - {msg.content}")
print()

# Example 3: Count message types
print("Example 3: Count Message Types")
print("-" * 80)

from collections import Counter
message_types = Counter(type(msg).__name__ for msg in sample_messages)
print("Message type counts:")
for msg_type, count in message_types.items():
    print(f"   {msg_type}: {count}")
print()

# Example 4: Extract content
print("Example 4: Extract Content")
print("-" * 80)

contents = [msg.content for msg in sample_messages]
print(f"Extracted {len(contents)} content strings:")
for i, content in enumerate(contents, 1):
    print(f"   {i}. {content}")
print()


# =============================================================================
# 4. PRACTICAL EXAMPLES - Real-world Usage
# =============================================================================
print("=" * 80)
print("4. PRACTICAL EXAMPLES - Real-world Usage")
print("=" * 80)
print()

# Example 1: Building a conversation for an agent
print("Example 1: Building Agent Conversation")
print("-" * 80)

builder = MessageBuilder()
conversation = (builder
    .add_system("You are a helpful math tutor. Use the calculator tool for complex calculations.")
    .add_human("What is 123 * 456?")
    .add_ai("Let me calculate that for you.")
    .add_tool("Result: 56088", tool_call_id="call_001", name="calculator")
    .add_ai("123 multiplied by 456 equals 56,088.")
    .build())

print("Built agent conversation:")
for msg in conversation:
    role = type(msg).__name__.replace("Message", "")
    print(f"   [{role}] {msg.content}")
print()

# Example 2: Building multi-turn dialogue
print("Example 2: Multi-turn Dialogue")
print("-" * 80)

builder = MessageBuilder()
builder.add_system("You are a helpful assistant")
builder.add_conversation([
    ("Hello!", "Hi! How can I help?"),
    ("Tell me about Python", "Python is a programming language..."),
    ("What about Java?", "Java is another popular language..."),
])

messages = builder.build()
print(f"Built dialogue with {len(messages)} messages")
print(f"   Total characters: {sum(len(msg.content) for msg in messages)}")
print()

# Example 3: Adding tool messages
print("Example 3: Tool Integration")
print("-" * 80)

builder = MessageBuilder()
builder.add_system("You have access to a calculator tool")
builder.add_human("What is 15 * 23?")
builder.add_tool("Result: 345", tool_call_id="calc_001", name="calculator")
builder.add_ai("15 multiplied by 23 equals 345")

messages = builder.build()
print(f"Built tool conversation with {len(messages)} messages")
for msg in messages:
    if isinstance(msg, ToolMessage):
        print(f"   Tool result: {msg.content} (call_id: {msg.tool_call_id})")
print()


# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("EXAMPLE COMPLETE!")
print("=" * 80)

print("\nWhat You Learned:")
print("   1. Message Types - SystemMessage, HumanMessage, AIMessage, ToolMessage")
print("   2. MessageBuilder - Fluent interface for building message lists")
print("   3. LangChain Native - Using LangChain's built-in capabilities")

print("\nKey Takeaways:")
print("   - SystemMessage: Sets agent behavior (use at start)")
print("   - HumanMessage: User input (every user message)")
print("   - AIMessage: Model output (usually auto-generated)")
print("   - ToolMessage: Tool results (with tool_call_id)")
print("   - Use MessageBuilder for clean, fluent message construction")
print("   - Use LangChain's native methods for validation and conversion")

print("\nNext Steps:")
print("   - Use MessageBuilder in your agents for cleaner code")
print("   - Leverage LangChain's built-in message operations")
print("   - Build multi-turn conversations with add_conversation()")

print("\nOfficial Documentation:")
print("   https://docs.langchain.com/oss/python/langchain/messages")

print()
