#!/usr/bin/env python3
"""
Message Toolkit Guide - Working with LangChain Messages

This example demonstrates the AI Toolkit's message utilities for working with
LangChain message types following official documentation:
https://docs.langchain.com/oss/python/langchain/messages

Components Demonstrated:
1. Message Types - SystemMessage, HumanMessage, AIMessage, ToolMessage
2. MessageBuilder - Fluent interface for building message lists
3. MessageFormatter - Display and format messages
4. MessageValidator - Validate message structures
5. MessageConverter - Convert between formats

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
from ai_toolkit.messages import (
    MessageBuilder,
    MessageFormatter,
    MessageValidator,
    MessageConverter
)


print("\n" + "=" * 80)
print("MESSAGE TOOLKIT GUIDE - LANGCHAIN MESSAGES")
print("=" * 80)
print("\nDemonstrating AI Toolkit's message utilities")
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
print("📌 SystemMessage - Sets agent behavior and personality")
system_msg = SystemMessage(content="You are a helpful AI assistant with access to tools.")
print(f"   Content: '{system_msg.content}'")
print(f"   Role: system")
print(f"   Use: At the start of conversation to define capabilities")
print()

# HumanMessage - User input
print("📌 HumanMessage - Represents user input")
human_msg = HumanMessage(content="What is 2 + 2?", name="Alice")
print(f"   Content: '{human_msg.content}'")
print(f"   Name: {human_msg.name}")
print(f"   Role: user")
print(f"   Use: Every user question or request")
print()

# AIMessage - Model output
print("📌 AIMessage - Represents model output")
ai_msg = AIMessage(content="2 + 2 equals 4.")
print(f"   Content: '{ai_msg.content}'")
print(f"   Role: assistant")
print(f"   Use: Model responses (usually auto-generated)")
print()

# ToolMessage - Tool execution results
print("📌 ToolMessage - Tool execution results")
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

print(f"✅ Built {len(messages)} messages using fluent interface")
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
print(f"✅ Built conversation with {len(messages)} messages")
print(f"   System messages: 1")
print(f"   Human-AI exchanges: 3")
print()

# Example 3: Builder methods
print("Example 3: Builder Utility Methods")
print("-" * 80)

builder = MessageBuilder()
builder.add_system("You are helpful")
builder.add_human("Hello!")

print(f"✅ Message count: {builder.count()}")
print(f"✅ Last message: {type(builder.get_last()).__name__}")
print(f"✅ Builder length: {len(builder)}")
print()


# =============================================================================
# 3. MESSAGE FORMATTER - Display and Format Messages
# =============================================================================
print("=" * 80)
print("3. MESSAGE FORMATTER - Display Messages")
print("=" * 80)
print()

# Create sample messages
sample_messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!", name="Alice"),
    AIMessage(content="Hi Alice! How can I help you today?"),
    HumanMessage(content="What is 2 + 2?", name="Alice"),
    AIMessage(content="2 + 2 equals 4."),
]

# Example 1: Format as conversation
print("Example 1: Conversation Format")
print("-" * 80)
print(MessageFormatter.format_conversation(sample_messages))
print()

# Example 2: Numbered format
print("Example 2: Numbered Format")
print("-" * 80)
print(MessageFormatter.format_messages(sample_messages, numbered=True))
print()

# Example 3: Message statistics
print("Example 3: Message Statistics")
print("-" * 80)
stats = MessageFormatter.get_message_stats(sample_messages)
print(f"✅ Total messages: {stats['total']}")
print(f"   System: {stats['system']}")
print(f"   Human: {stats['human']}")
print(f"   AI: {stats['ai']}")
print(f"   Tool: {stats['tool']}")
print(f"   Total characters: {stats['total_chars']}")
print()

# Example 4: Format as dictionaries
print("Example 4: Dictionary Format")
print("-" * 80)
dicts = MessageFormatter.format_as_dicts(sample_messages[:2])
for i, d in enumerate(dicts, 1):
    print(f"   {i}. {d}")
print()


# =============================================================================
# 4. MESSAGE VALIDATOR - Validate Message Structures
# =============================================================================
print("=" * 80)
print("4. MESSAGE VALIDATOR - Validate Messages")
print("=" * 80)
print()

# Example 1: Validate single message
print("Example 1: Validate Single Message")
print("-" * 80)

valid_msg = HumanMessage(content="Hello!")
is_valid, error = MessageValidator.validate_message(valid_msg)
print(f"✅ Valid message: {is_valid}")

invalid_msg = HumanMessage(content="")
is_valid, error = MessageValidator.validate_message(invalid_msg)
print(f"❌ Invalid message: {is_valid}, Error: {error}")
print()

# Example 2: Validate message list
print("Example 2: Validate Message List")
print("-" * 80)

is_valid, error = MessageValidator.validate_messages(
    sample_messages,
    require_system=True,
    require_human=True
)
print(f"✅ Valid message list: {is_valid}")
print()

# Example 3: Validate conversation flow
print("Example 3: Validate Conversation Flow")
print("-" * 80)

is_valid, error = MessageValidator.validate_conversation_flow(sample_messages)
print(f"✅ Valid conversation flow: {is_valid}")
print()

# Example 4: Count message types
print("Example 4: Count Message Types")
print("-" * 80)

counts = MessageValidator.count_message_types(sample_messages)
print(f"✅ Message type counts:")
for msg_type, count in counts.items():
    if count > 0:
        print(f"   {msg_type}: {count}")
print()

# Example 5: Filter by type
print("Example 5: Filter Messages by Type")
print("-" * 80)

human_messages = MessageValidator.filter_by_type(sample_messages, HumanMessage)
print(f"✅ Found {len(human_messages)} HumanMessages:")
for msg in human_messages:
    print(f"   - {msg.content}")
print()


# =============================================================================
# 5. MESSAGE CONVERTER - Convert Between Formats
# =============================================================================
print("=" * 80)
print("5. MESSAGE CONVERTER - Convert Formats")
print("=" * 80)
print()

# Example 1: Convert to dictionaries
print("Example 1: Convert to Dictionaries")
print("-" * 80)

msg = HumanMessage(content="Hello!", name="Alice")
dict_msg = MessageConverter.to_dict(msg)
print(f"✅ Message as dict: {dict_msg}")
print()

# Example 2: Convert from dictionaries
print("Example 2: Convert from Dictionaries")
print("-" * 80)

data = [
    {'role': 'system', 'content': 'You are helpful'},
    {'role': 'user', 'content': 'Hello!', 'name': 'Alice'},
    {'role': 'assistant', 'content': 'Hi Alice!'}
]

messages = MessageConverter.from_dicts(data)
print(f"✅ Converted {len(messages)} messages from dicts:")
for msg in messages:
    print(f"   - {type(msg).__name__}: {msg.content}")
print()

# Example 3: OpenAI format
print("Example 3: OpenAI Format Conversion")
print("-" * 80)

openai_format = MessageConverter.to_openai_format(sample_messages[:3])
print(f"✅ Converted to OpenAI format:")
for msg in openai_format:
    print(f"   {msg}")
print()

# Example 4: Extract content
print("Example 4: Extract Content Only")
print("-" * 80)

contents = MessageConverter.extract_content(sample_messages)
print(f"✅ Extracted {len(contents)} content strings:")
for i, content in enumerate(contents, 1):
    print(f"   {i}. {content}")
print()

# Example 5: Merge message lists
print("Example 5: Merge Message Lists")
print("-" * 80)

history = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello!")
]

new_messages = [
    HumanMessage(content="What is 2+2?")
]

merged = MessageConverter.merge_messages(history, new_messages)
print(f"✅ Merged messages: {len(history)} + {len(new_messages)} = {len(merged)}")
print()

# Example 6: Clone messages
print("Example 6: Clone Messages")
print("-" * 80)

original = HumanMessage(content="Original message", name="Alice")
cloned = MessageConverter.clone_message(original)

print(f"✅ Original: {original.content} (name: {original.name})")
print(f"✅ Cloned: {cloned.content} (name: {cloned.name})")
print(f"✅ Are they the same object? {original is cloned}")
print()


# =============================================================================
# 6. PRACTICAL EXAMPLES - Real-world Usage
# =============================================================================
print("=" * 80)
print("6. PRACTICAL EXAMPLES - Real-world Usage")
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

print("✅ Built agent conversation:")
MessageFormatter.print_messages(conversation, style='conversation')
print()

# Example 2: Validating and formatting user input
print("Example 2: Validate and Format User Input")
print("-" * 80)

user_messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello!"),
    HumanMessage(content="How are you?")
]

is_valid, error = MessageValidator.validate_messages(user_messages)
if is_valid:
    print("✅ Messages are valid")
    print("\nFormatted output:")
    MessageFormatter.print_messages(user_messages, style='conversation')
else:
    print(f"❌ Validation error: {error}")
print()

# Example 3: Converting for API calls
print("Example 3: Convert for API Calls")
print("-" * 80)

api_messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is Python?")
]

# Convert to OpenAI format for API call
openai_msgs = MessageConverter.to_openai_format(api_messages)
print("✅ Converted to OpenAI format for API:")
for msg in openai_msgs:
    print(f"   {msg}")
print()

# Example 4: Message statistics and analysis
print("Example 4: Analyze Conversation")
print("-" * 80)

long_conversation = MessageBuilder()
long_conversation.add_system("You are helpful")
long_conversation.add_conversation([
    ("Hello!", "Hi! How can I help?"),
    ("Tell me about Python", "Python is a programming language..."),
    ("What about Java?", "Java is another popular language..."),
    ("Which is better?", "Both have their strengths...")
])

messages = long_conversation.build()
stats = MessageFormatter.get_message_stats(messages)
counts = MessageValidator.count_message_types(messages)

print(f"✅ Conversation Analysis:")
print(f"   Total messages: {stats['total']}")
print(f"   Human messages: {counts['human']}")
print(f"   AI messages: {counts['ai']}")
print(f"   Total characters: {stats['total_chars']}")
print(f"   Average message length: {stats['total_chars'] // stats['total']} chars")
print()


# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("EXAMPLE COMPLETE!")
print("=" * 80)

print("\n🎓 What You Learned:")
print("   1. Message Types - SystemMessage, HumanMessage, AIMessage, ToolMessage")
print("   2. MessageBuilder - Fluent interface for building message lists")
print("   3. MessageFormatter - Display and format messages in various styles")
print("   4. MessageValidator - Validate message structures and flows")
print("   5. MessageConverter - Convert between formats (dict, OpenAI, etc.)")

print("\n📚 Key Takeaways:")
print("   - SystemMessage: Sets agent behavior (use at start)")
print("   - HumanMessage: User input (every user message)")
print("   - AIMessage: Model output (usually auto-generated)")
print("   - ToolMessage: Tool results (with tool_call_id)")
print("   - Use MessageBuilder for clean, fluent message construction")
print("   - Use MessageFormatter for displaying conversations")
print("   - Use MessageValidator for ensuring message quality")
print("   - Use MessageConverter for format transformations")

print("\n💡 Next Steps:")
print("   - Use MessageBuilder in your agents for cleaner code")
print("   - Validate messages before sending to models")
print("   - Format messages for better debugging")
print("   - Convert messages for different API formats")

print("\n📖 Official Documentation:")
print("   https://docs.langchain.com/oss/python/langchain/messages")

print()
