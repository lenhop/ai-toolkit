#!/usr/bin/env python3
"""
Streaming Guide - Simple examples for AI model streaming

Based on official LangChain streaming patterns.
Official docs: https://docs.langchain.com/oss/python/langchain-streaming

Examples:
1. Basic streaming - Direct model.stream()
2. Callback streaming - Custom token processing
3. Agent streaming - Streaming with tools

Requirements:
- Set DEEPSEEK_API_KEY, QWEN_API_KEY, or GLM_API_KEY in .env file
"""

import os
import sys
from pathlib import Path

# Setup path and load environment
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from ai_toolkit.models import ModelManager
from ai_toolkit.streaming import StreamCallback


def basic_streaming_example():
    """Example 1: Basic streaming - simplest way to stream."""
    print("=" * 60)
    print("Example 1: Basic Streaming")
    print("=" * 60)
    
    try:
        # Create model
        manager = ModelManager()
        model = manager.create_model("deepseek")
        
        print("\n👤 User: Tell me a short joke about programming")
        print("🤖 AI: ", end="", flush=True)
        
        # Stream directly - simplest approach
        for chunk in model.stream("Tell me a short joke about programming"):
            print(chunk.content, end="", flush=True)
        
        print("\n\n✅ Done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have API keys configured in .env file")


def callback_streaming_example():
    """Example 2: Streaming with callback for custom processing."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming with Callback")
    print("=" * 60)
    
    try:
        # Create model
        manager = ModelManager()
        model = manager.create_model("deepseek")
        
        # Create callback with custom processing
        def on_token(token: str):
            # Custom processing - could save to file, send to UI, etc.
            pass
        
        def on_complete(text: str):
            print(f"\n\n📊 Total characters: {len(text)}")
            print(f"📊 Total words: {len(text.split())}")
        
        callback = StreamCallback(
            on_token=on_token,
            on_complete=on_complete,
            verbose=True  # Print tokens as they arrive
        )
        
        print("\n👤 User: What is Python?")
        print("🤖 AI: ", end="", flush=True)
        
        # Stream with callback
        from langchain_core.runnables import RunnableConfig
        config = RunnableConfig(callbacks=[callback])
        
        for chunk in model.stream("What is Python in one sentence?", config=config):
            pass  # Callback handles display
        
        # Get accumulated text
        full_text = callback.get_accumulated_text()
        print(f"✅ Captured {len(full_text)} characters")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def agent_streaming_example():
    """Example 3: Streaming with agent and tools."""
    print("\n" + "=" * 60)
    print("Example 3: Agent Streaming with Tools")
    print("=" * 60)
    
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import PromptTemplate
        from langchain.tools import tool
        
        # Create model
        manager = ModelManager()
        model = manager.create_model("deepseek")
        
        # Create a simple tool
        @tool
        def calculate(expression: str) -> str:
            """Calculate a mathematical expression. Input should be a valid Python expression."""
            try:
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        
        tools = [calculate]
        
        # Create agent
        prompt = PromptTemplate.from_template(
            "Answer the following questions as best you can. You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "Thought:{agent_scratchpad}"
        )
        
        agent = create_react_agent(model, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # Show agent reasoning
            handle_parsing_errors=True
        )
        
        print("\n👤 User: What is 15 * 23?")
        print("🤖 Agent thinking...\n")
        
        # Run agent (it will stream internally if verbose=True)
        result = agent_executor.invoke({"input": "What is 15 * 23?"})
        
        print(f"\n✅ Final answer: {result['output']}")
        
    except ImportError as e:
        print(f"⚠️  Missing dependencies: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def run_all_examples():
    """Run all streaming examples."""
    print("\n🎯 AI Toolkit Streaming Guide")
    print("Based on official LangChain patterns\n")
    
    # Check API keys
    has_keys = any([
        os.getenv('DEEPSEEK_API_KEY'),
        os.getenv('QWEN_API_KEY'),
        os.getenv('GLM_API_KEY')
    ])
    
    if not has_keys:
        print("⚠️  No API keys found!")
        print("Please set DEEPSEEK_API_KEY, QWEN_API_KEY, or GLM_API_KEY in .env file")
        return
    
    print("🔑 API keys found\n")
    
    # Run examples
    examples = [
        basic_streaming_example,
        callback_streaming_example,
        agent_streaming_example,
    ]
    
    for example in examples:
        try:
            example()
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Streaming Guide Complete!")
    print("=" * 60)
    print("\n💡 Key Takeaways:")
    print("   1. Use model.stream() for simple streaming")
    print("   2. Use StreamCallback for custom processing")
    print("   3. Agents stream automatically with verbose=True")
    print("\n📚 Learn more: https://docs.langchain.com/oss/python/langchain-streaming")


if __name__ == "__main__":
    run_all_examples()
