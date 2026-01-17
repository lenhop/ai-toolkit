#!/usr/bin/env python3
"""
Memory Management Guide - Managing Agent Memory

This example demonstrates memory management for LangChain agents including
checkpointers, message trimming, summarization, and custom state.

Topics Covered:
1. Checkpointer Types - InMemory and PostgreSQL
2. Message Trimming - Keep conversations manageable
3. Message Deletion - Remove old messages
4. Summarization - Compress long conversations
5. Custom State - Add custom fields to agent state
6. Dynamic Prompts - Generate prompts from context

Official Documentation:
https://docs.langchain.com/oss/python/langchain/short-term-memory
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
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

# Import ai_toolkit
from ai_toolkit.models.model_manager import ModelManager
from ai_toolkit.memory import (
    MemoryManager,
    CheckpointerFactory,
    MessageTrimmer,
    create_trimming_middleware,
    create_deletion_middleware,
    create_dynamic_prompt_middleware
)


print("\n" + "=" * 80)
print("MEMORY MANAGEMENT GUIDE")
print("=" * 80)
print("\nDemonstrating memory management for LangChain agents")
print()


# Create model for examples
model_manager = ModelManager()
model = model_manager.create_model(
    provider_name="deepseek",
    model_name="deepseek-chat",
    temperature=0.7,
    max_tokens=2000
)

# Create simple tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# 1. CHECKPOINTER TYPES - InMemory and PostgreSQL
# =============================================================================

print("=" * 80)
print("1. CHECKPOINTER TYPES")
print("=" * 80)
print()

# Example 1: InMemory Checkpointer (Development)
print("Example 1: InMemory Checkpointer")
print("-" * 80)

factory = CheckpointerFactory()
inmemory_checkpointer = factory.create_inmemory()

print("✅ Created InMemory checkpointer")
print("   Best for: Development, testing, single-session")
print("   Persistence: Lost on restart")
print()

# Example 2: PostgreSQL Checkpointer (Production)
print("Example 2: PostgreSQL Checkpointer")
print("-" * 80)

print("✅ PostgreSQL checkpointer (example)")
print("   Best for: Production, multi-user, persistent storage")
print("   Usage:")
print("   >>> db_uri = 'postgresql://user:pass@host:port/db'")
print("   >>> checkpointer = factory.create_postgres(db_uri)")
print("   >>> # Tables auto-created with auto_setup=True")
print()

# Example 3: Using MemoryManager
print("Example 3: Using MemoryManager")
print("-" * 80)

manager = MemoryManager()
checkpointer = manager.create_checkpointer("inmemory")
config = manager.create_config("thread-1")

print("✅ Created checkpointer with MemoryManager")
print(f"   Type: InMemory")
print(f"   Config: {config}")
print()


# =============================================================================
# 2. MESSAGE TRIMMING - Keep Conversations Manageable
# =============================================================================

print("=" * 80)
print("2. MESSAGE TRIMMING")
print("=" * 80)
print()

# Example 1: Manual Trimming
print("Example 1: Manual Message Trimming")
print("-" * 80)

from langchain_core.messages import SystemMessage, AIMessage

# Create sample messages
messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi!"),
    HumanMessage(content="What is 2+2?"),
    AIMessage(content="4"),
    HumanMessage(content="What is 3+3?"),
    AIMessage(content="6"),
    HumanMessage(content="What is 4+4?"),
]

print(f"Original messages: {len(messages)}")

# Trim messages
trimmer = MessageTrimmer(strategy="keep_first_and_recent", max_messages=5)
trimmed = trimmer.trim(messages)

print(f"Trimmed messages: {len(trimmed)}")
print(f"Strategy: Keep first (system) + 4 recent messages")
print()

# Example 2: Automatic Trimming with Middleware
print("Example 2: Automatic Trimming with Middleware")
print("-" * 80)

trim_middleware = create_trimming_middleware(max_messages=5)

agent_with_trimming = create_agent(
    model=model,
    tools=[],  # No tools to avoid tool message issues
    middleware=[trim_middleware],
    checkpointer=checkpointer,
    system_prompt="You are a helpful assistant."
)

print("✅ Created agent with automatic message trimming")
print("   Max messages: 5")
print("   Strategy: Keep first + recent")
print()

# Test the agent
config_trim = manager.create_config("trim-demo")

print("Testing agent with trimming:")
for i in range(3):
    query = f"Tell me a fact about number {i+1}"
    print(f"   Query {i+1}: {query}")
    result = agent_with_trimming.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config_trim
    )
    response = result["messages"][-1].content
    print(f"   Response: {response[:50]}...")

print(f"\n   Total messages in memory: {len(result['messages'])}")
print("   (Older messages automatically trimmed)")
print()


# =============================================================================
# 3. MESSAGE DELETION - Remove Old Messages
# =============================================================================

print("=" * 80)
print("3. MESSAGE DELETION")
print("=" * 80)
print()

print("Example: Automatic Message Deletion")
print("-" * 80)

delete_middleware = create_deletion_middleware(
    delete_count=2,
    trigger_count=6
)

agent_with_deletion = create_agent(
    model=model,
    tools=[],
    middleware=[delete_middleware],
    checkpointer=checkpointer,
    system_prompt="You are concise and helpful."
)

print("✅ Created agent with automatic message deletion")
print("   Delete count: 2 oldest messages")
print("   Trigger: When > 6 messages")
print()

config_delete = manager.create_config("delete-demo")

print("Testing agent with deletion:")
result = agent_with_deletion.invoke(
    {"messages": [HumanMessage(content="Hi, I'm Bob")]},
    config=config_delete
)
print(f"   After message 1: {len(result['messages'])} messages")

result = agent_with_deletion.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config_delete
)
print(f"   After message 2: {len(result['messages'])} messages")
print("   (Old messages deleted when threshold reached)")
print()


# =============================================================================
# 4. SUMMARIZATION - Compress Long Conversations
# =============================================================================

print("=" * 80)
print("4. SUMMARIZATION")
print("=" * 80)
print()

print("Example: Message Summarization")
print("-" * 80)

from ai_toolkit.memory import MessageSummarizer

summarizer = manager.create_summarizer(
    trigger_tokens=100,  # Low threshold for demo
    keep_messages=3
)

# Create long conversation
long_messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Tell me about Python"),
    AIMessage(content="Python is a programming language..."),
    HumanMessage(content="What about Java?"),
    AIMessage(content="Java is another language..."),
    HumanMessage(content="Which is better?"),
    AIMessage(content="Both have strengths..."),
]

print(f"Original messages: {len(long_messages)}")
print(f"Should summarize: {summarizer.should_summarize(long_messages)}")

if summarizer.should_summarize(long_messages):
    # Summarize old messages
    old_messages = long_messages[:-summarizer.keep_messages]
    summary = summarizer.summarize(old_messages)
    print(f"\nSummary of {len(old_messages)} messages:")
    print(f"   {summary[:100]}...")
    
    # Keep recent messages
    recent = long_messages[-summarizer.keep_messages:]
    print(f"\nKept {len(recent)} recent messages")

print()


# =============================================================================
# 5. CUSTOM STATE - Add Custom Fields
# =============================================================================

print("=" * 80)
print("5. CUSTOM STATE")
print("=" * 80)
print()

print("Example: Custom Agent State")
print("-" * 80)

print("✅ Custom state allows adding fields to agent state")
print("\nUsage:")
print("   >>> from langchain.agents import AgentState")
print("   >>> ")
print("   >>> class CustomState(AgentState):")
print("   >>>     user_id: str")
print("   >>>     preferences: dict")
print("   >>> ")
print("   >>> agent = create_agent(")
print("   >>>     model,")
print("   >>>     tools,")
print("   >>>     state_schema=CustomState")
print("   >>> )")
print("   >>> ")
print("   >>> result = agent.invoke({")
print("   >>>     'messages': [...],")
print("   >>>     'user_id': 'user_123',")
print("   >>>     'preferences': {'theme': 'dark'}")
print("   >>> })")
print()


# =============================================================================
# 6. DYNAMIC PROMPTS - Generate from Context
# =============================================================================

print("=" * 80)
print("6. DYNAMIC PROMPTS")
print("=" * 80)
print()

print("Example: Dynamic Prompt Generation")
print("-" * 80)

def generate_prompt(context):
    """Generate system prompt from context."""
    user_name = context.get("user_name", "User")
    return f"You are a helpful assistant. Address the user as {user_name}."

# Create middleware
prompt_middleware = create_dynamic_prompt_middleware(generate_prompt)

print("✅ Created dynamic prompt middleware")
print("\nUsage:")
print("   >>> agent = create_agent(")
print("   >>>     model,")
print("   >>>     tools,")
print("   >>>     middleware=[prompt_middleware],")
print("   >>>     context_schema=CustomContext")
print("   >>> )")
print("   >>> ")
print("   >>> result = agent.invoke(")
print("   >>>     {'messages': [...]},")
print("   >>>     context={'user_name': 'John'}")
print("   >>> )")
print("\nPrompt will be: 'You are helpful. Address user as John.'")
print()


# =============================================================================
# 7. PRACTICAL EXAMPLE - Combining Features
# =============================================================================

print("=" * 80)
print("7. PRACTICAL EXAMPLE - Combining Features")
print("=" * 80)
print()

print("Example: Agent with Multiple Memory Features")
print("-" * 80)

# Create agent without middleware to avoid tool message issues
practical_agent = create_agent(
    model=model,
    tools=[],
    checkpointer=checkpointer,
    system_prompt="You are a helpful assistant."
)

config_practical = manager.create_config("practical-demo")

print("✅ Created agent with:")
print("   - InMemory checkpointer")
print("   - Conversation memory enabled")
print()

# Test the agent
print("Testing practical agent:")
queries = [
    "Hello, my name is Alice",
    "What's the capital of France?",
    "What's my name?"
]

for i, query in enumerate(queries, 1):
    print(f"\n   Query {i}: {query}")
    result = practical_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config_practical
    )
    response = result["messages"][-1].content
    print(f"   Response: {response[:80]}...")

print(f"\n   Final message count: {len(result['messages'])}")
print("   (Memory managed automatically)")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("EXAMPLE COMPLETE!")
print("=" * 80)

print("\n🎓 Memory Management Features Covered:")
print("   1. Checkpointer Types - InMemory and PostgreSQL")
print("   2. Message Trimming - Keep conversations manageable")
print("   3. Message Deletion - Remove old messages")
print("   4. Summarization - Compress long conversations")
print("   5. Custom State - Add custom fields to agent state")
print("   6. Dynamic Prompts - Generate prompts from context")

print("\n📚 Key Classes:")
print("   - MemoryManager: High-level memory management")
print("   - CheckpointerFactory: Create checkpointers")
print("   - MessageTrimmer: Trim messages")
print("   - MessageSummarizer: Summarize conversations")

print("\n💡 Best Practices:")
print("   - Use InMemory for development")
print("   - Use PostgreSQL for production")
print("   - Trim messages to prevent context overflow")
print("   - Summarize long conversations")
print("   - Use custom state for user-specific data")
print("   - Generate dynamic prompts for personalization")

print("\n📖 Official Documentation:")
print("   https://docs.langchain.com/oss/python/langchain/short-term-memory")

print()
