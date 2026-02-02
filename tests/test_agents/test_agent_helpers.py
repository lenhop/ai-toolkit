"""
Test Agent Helper Functions

Tests for ai_toolkit.agents.agent_helpers module:
- create_agent_with_tools
- create_agent_with_memory
- create_streaming_agent
- create_structured_output_agent

Based on examples: 13_agent_base.py, 14_agent_advanced.py
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
    from ai_toolkit.models import ModelManager
    from ai_toolkit.agents import (
        create_agent_with_tools,
        create_agent_with_memory,
        create_streaming_agent,
        create_structured_output_agent,
    )
    from langchain.tools import tool
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_create_agent_with_tools():
    """Test create_agent_with_tools function."""
    print("\n" + "=" * 80)
    print("Test 1: create_agent_with_tools")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"
        
        @tool
        def get_weather(location: str) -> str:
            """Get weather information for a location."""
            return f"Weather in {location}: Sunny, 72°F"
        
        tools = [search, get_weather]
        
        # Test: Create agent with tools
        agent = create_agent_with_tools(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant."
        )
        
        print("✓ Agent created successfully")
        
        # Test: Invoke agent
        result = agent.invoke({
            "messages": [HumanMessage(content="What's the weather in Beijing?")]
        })
        
        print(f"✓ Agent invoked successfully")
        print(f"✓ Response type: {type(result)}")
        print(f"✓ Has messages: {'messages' in result}")
        
        if 'messages' in result and result['messages']:
            last_msg = result['messages'][-1]
            print(f"✓ Last message type: {type(last_msg).__name__}")
            if hasattr(last_msg, 'content'):
                print(f"✓ Response preview: {str(last_msg.content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_create_agent_with_memory():
    """Test create_agent_with_memory function."""
    print("\n" + "=" * 80)
    print("Test 2: create_agent_with_memory")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"
        
        tools = [search]
        
        # Test: Create agent with in-memory checkpointer
        agent = create_agent_with_memory(
            model=model,
            tools=tools,
            checkpointer_type="inmemory",
            system_prompt="You are a helpful assistant."
        )
        
        print("✓ Agent with memory created successfully")
        
        # Test: Invoke with thread_id
        config = {"configurable": {"thread_id": "test-thread-1"}}
        
        result1 = agent.invoke({
            "messages": [HumanMessage(content="Hi! My name is Alice.")]
        }, config=config)
        
        print("✓ First message sent")
        
        # Test: Continue conversation (memory should persist)
        result2 = agent.invoke({
            "messages": [HumanMessage(content="What's my name?")]
        }, config=config)
        
        print("✓ Second message sent")
        print(f"✓ Memory test - response: {str(result2['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_create_streaming_agent():
    """Test create_streaming_agent function."""
    print("\n" + "=" * 80)
    print("Test 3: create_streaming_agent")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a given city."""
            return f"It's always sunny in {city}!"
        
        tools = [get_weather]
        
        # Test: Create streaming agent
        agent = create_streaming_agent(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant."
        )
        
        print("✓ Streaming agent created successfully")
        
        # Test: Stream agent progress
        print("\nTesting stream_mode='updates':")
        update_count = 0
        for chunk in agent.stream(
            {"messages": [HumanMessage(content="What's the weather in SF?")]},
            stream_mode="updates"
        ):
            update_count += 1
            if update_count <= 3:  # Show first 3 updates
                for step, data in chunk.items():
                    print(f"  Update {update_count}: step={step}")
        
        print(f"✓ Received {update_count} updates")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_create_structured_output_agent():
    """Test create_structured_output_agent function."""
    print("\n" + "=" * 80)
    print("Test 4: create_structured_output_agent")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"
        
        tools = [search]
        
        # Define schema
        class ContactInfo(BaseModel):
            """Contact information for a person."""
            name: str = Field(description="The name of the person")
            email: str = Field(description="The email address of the person")
            phone: str = Field(description="The phone number of the person")
        
        # Test: Create agent with ToolStrategy
        agent = create_structured_output_agent(
            model=model,
            tools=tools,
            schema=ContactInfo,
            strategy="tool"
        )
        
        print("✓ Structured output agent created (ToolStrategy)")
        
        # Test: Invoke with structured output
        result = agent.invoke({
            "messages": [HumanMessage(
                content="Extract contact info from: John Doe, john@example.com, (555) 123-4567"
            )]
        })
        
        print("✓ Agent invoked successfully")
        
        if 'structured_response' in result:
            print(f"✓ Structured response: {result['structured_response']}")
        else:
            print("⚠ No 'structured_response' in result")
            print(f"  Available keys: {list(result.keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Agent Helper Functions")
    print("=" * 80)
    
    test_create_agent_with_tools()
    test_create_agent_with_memory()
    test_create_streaming_agent()
    test_create_structured_output_agent()
    
    print("\n" + "=" * 80)
    print("Agent Helper Tests Completed")
    print("=" * 80)
