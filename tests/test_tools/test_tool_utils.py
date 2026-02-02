"""
Test Tool Utilities

Tests for ai_toolkit.tools.tool_utils module:
- create_search_tool
- create_weather_tool
- create_calculator_tool
- create_memory_access_tool
- create_memory_update_tool
- wrap_tool_with_error_handler

Based on examples: 13_agent_base.py, 16_tool_base.py
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
    from ai_toolkit.tools import (
        create_search_tool,
        create_weather_tool,
        create_calculator_tool,
        create_memory_access_tool,
        create_memory_update_tool,
        wrap_tool_with_error_handler,
    )
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langgraph.store.memory import InMemoryStore
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_create_search_tool():
    """Test create_search_tool function."""
    print("\n" + "=" * 80)
    print("Test 1: create_search_tool")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Test: Default mock search tool
        search = create_search_tool()
        print("✓ Default search tool created")
        
        # Test: Custom search tool
        def my_search(query: str) -> str:
            return f"Custom results for: {query}"
        
        custom_search = create_search_tool(
            search_function=my_search,
            description="Search the knowledge base."
        )
        print("✓ Custom search tool created")
        
        # Test: Use in agent
        from ai_toolkit.models import ModelManager
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        agent = create_agent(model, tools=[custom_search])
        result = agent.invoke({
            "messages": [HumanMessage(content="Search for AI news")]
        })
        
        print("✓ Tool used in agent successfully")
        print(f"✓ Response preview: {str(result['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_create_weather_tool():
    """Test create_weather_tool function."""
    print("\n" + "=" * 80)
    print("Test 2: create_weather_tool")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Test: Default mock weather tool
        weather = create_weather_tool()
        print("✓ Default weather tool created")
        
        # Test: Custom weather tool
        def my_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 72°F"
        
        custom_weather = create_weather_tool(
            weather_function=my_weather,
            description="Get current weather for any city."
        )
        print("✓ Custom weather tool created")
        
        # Test: Direct tool call
        result = custom_weather.invoke({"location": "Beijing"})
        print(f"✓ Tool invoked directly: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_create_calculator_tool():
    """Test create_calculator_tool function."""
    print("\n" + "=" * 80)
    print("Test 3: create_calculator_tool")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Test: Safe calculator
        calc = create_calculator_tool(safe_mode=True)
        print("✓ Safe calculator tool created")
        
        # Test: Direct tool call
        result = calc.invoke({"expression": "25 * 4"})
        print(f"✓ Calculator result: {result}")
        
        # Test: Use in agent
        from ai_toolkit.models import ModelManager
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        agent = create_agent(model, tools=[calc])
        result = agent.invoke({
            "messages": [HumanMessage(content="What is 123 * 456?")]
        })
        
        print("✓ Calculator used in agent")
        print(f"✓ Response preview: {str(result['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_memory_tools():
    """Test memory access and update tools."""
    print("\n" + "=" * 80)
    print("Test 4: Memory Tools (get_memory, save_memory)")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Create memory tools
        get_memory = create_memory_access_tool(
            memory_key="users",
            description="Look up user info by user_id."
        )
        
        save_memory = create_memory_update_tool(
            memory_key="users",
            description="Save user info with user_id as key."
        )
        
        print("✓ Memory tools created")
        
        # Create store
        store = InMemoryStore()
        
        # Test: Save to memory
        # Note: Tools need runtime, so we test via agent
        from ai_toolkit.models import ModelManager
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        agent = create_agent(
            model,
            tools=[get_memory, save_memory],
            store=store,
            system_prompt="""You are a user information management assistant. 
            When user asks to save info, use save_memory tool.
            When user asks to get info, use get_memory tool."""
        )
        
        print("✓ Agent created with memory tools")
        
        # Test: Save user info
        result1 = agent.invoke({
            "messages": [HumanMessage(
                content="Save user: user_id=abc123, name=Foo, age=25, email=foo@example.com"
            )]
        })
        
        print("✓ Save operation attempted")
        print(f"✓ Response: {str(result1['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_wrap_tool_with_error_handler():
    """Test wrap_tool_with_error_handler function."""
    print("\n" + "=" * 80)
    print("Test 5: wrap_tool_with_error_handler")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        from langchain.tools import tool
        
        # Create a risky tool
        @tool
        def risky_tool(input: str) -> str:
            """A tool that might fail."""
            if not input:
                raise ValueError("Input required")
            return f"Processed: {input}"
        
        # Wrap with error handler
        safe_tool = wrap_tool_with_error_handler(
            risky_tool,
            error_message="Tool failed. Please provide valid input."
        )
        
        print("✓ Tool wrapped with error handler")
        
        # Test: Valid input
        result1 = safe_tool.invoke({"input": "test"})
        print(f"✓ Valid input result: {result1}")
        
        # Test: Invalid input (should return error message, not raise)
        result2 = safe_tool.invoke({"input": ""})
        print(f"✓ Invalid input handled: {result2}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Tool Utilities")
    print("=" * 80)
    
    test_create_search_tool()
    test_create_weather_tool()
    test_create_calculator_tool()
    test_memory_tools()
    test_wrap_tool_with_error_handler()
    
    print("\n" + "=" * 80)
    print("Tool Utilities Tests Completed")
    print("=" * 80)
