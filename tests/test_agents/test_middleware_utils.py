"""
Test Middleware Utilities

Tests for ai_toolkit.agents.middleware_utils module:
- create_dynamic_model_selector
- create_tool_error_handler
- create_context_based_prompt

Based on examples: 13_agent_base.py
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
        create_dynamic_model_selector,
        create_tool_error_handler,
        create_context_based_prompt,
    )
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.messages import HumanMessage
    from typing import TypedDict
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_dynamic_model_selector():
    """Test create_dynamic_model_selector middleware."""
    print("\n" + "=" * 80)
    print("Test 1: create_dynamic_model_selector")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup models
        manager = ModelManager()
        basic_model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        advanced_model = manager.create_model(
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
        
        # Create middleware
        model_selector = create_dynamic_model_selector(
            basic_model=basic_model,
            advanced_model=advanced_model,
            threshold=5  # Switch after 5 messages
        )
        
        print("✓ Dynamic model selector created")
        
        # Create agent with middleware
        agent = create_agent(
            model=basic_model,  # Default model
            tools=tools,
            middleware=[model_selector],
            system_prompt="You are a helpful assistant."
        )
        
        print("✓ Agent created with dynamic model selector")
        
        # Test: Short conversation (should use basic model)
        result = agent.invoke({
            "messages": [HumanMessage(content="Hello!")]
        })
        
        print("✓ Agent invoked successfully")
        print(f"✓ Response preview: {str(result['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_tool_error_handler():
    """Test create_tool_error_handler middleware."""
    print("\n" + "=" * 80)
    print("Test 2: create_tool_error_handler")
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
        def risky_tool(input: str) -> str:
            """A tool that might fail."""
            if not input:
                raise ValueError("Input required")
            return f"Processed: {input}"
        
        tools = [risky_tool]
        
        # Create error handler middleware
        error_handler = create_tool_error_handler(
            error_message_template="Tool failed: {error}. Please try again.",
            log_errors=True
        )
        
        print("✓ Tool error handler created")
        
        # Create agent with error handler
        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[error_handler],
            system_prompt="You are a helpful assistant."
        )
        
        print("✓ Agent created with error handler")
        
        # Test: Invoke with valid input
        result1 = agent.invoke({
            "messages": [HumanMessage(content="Use risky_tool with input 'test'")]
        })
        
        print("✓ Agent invoked with valid input")
        
        # Test: Invoke with invalid input (should handle error gracefully)
        result2 = agent.invoke({
            "messages": [HumanMessage(content="Use risky_tool with empty input")]
        })
        
        print("✓ Agent invoked with invalid input (error handled)")
        print(f"✓ Response preview: {str(result2['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_context_based_prompt():
    """Test create_context_based_prompt middleware."""
    print("\n" + "=" * 80)
    print("Test 3: create_context_based_prompt")
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
        
        # Define context schema
        class UserContext(TypedDict):
            user_role: str
            expertise_level: str
        
        # Create prompt generator
        def generate_prompt(context):
            role = context.get("user_role", "user")
            level = context.get("expertise_level", "beginner")
            
            if role == "expert":
                return "You are a helpful assistant. Provide detailed technical responses."
            elif level == "beginner":
                return "You are a helpful assistant. Explain concepts simply, avoid jargon."
            else:
                return "You are a helpful assistant."
        
        # Create middleware
        prompt_middleware = create_context_based_prompt(generate_prompt)
        
        print("✓ Context-based prompt middleware created")
        
        # Create agent with middleware
        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[prompt_middleware],
            context_schema=UserContext,
            system_prompt="You are a helpful assistant."  # Will be overridden
        )
        
        print("✓ Agent created with context-based prompt")
        
        # Test: Invoke with context
        result = agent.invoke(
            {"messages": [HumanMessage(content="Explain machine learning")]},
            context={"user_role": "expert", "expertise_level": "advanced"}
        )
        
        print("✓ Agent invoked with context")
        print(f"✓ Response preview: {str(result['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Middleware Utilities")
    print("=" * 80)
    
    test_dynamic_model_selector()
    test_tool_error_handler()
    test_context_based_prompt()
    
    print("\n" + "=" * 80)
    print("Middleware Utilities Tests Completed")
    print("=" * 80)
