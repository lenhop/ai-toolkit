#!/usr/bin/env python3
"""
Streaming Guide - Simple examples for AI model streaming

Based on official LangChain streaming patterns.
Official docs: https://docs.langchain.com/oss/python/langchain/streaming/overview

Examples:
1. Basic streaming - Direct model.stream()
2. Accumulating text - Save while streaming
3. Custom callbacks - Using LangChain's BaseCallbackHandler
4. StreamHandler - Optional convenience wrapper

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


def accumulate_streaming_example():
    """Example 2: Accumulating text while streaming."""
    print("\n" + "=" * 60)
    print("Example 2: Accumulating Text")
    print("=" * 60)
    
    try:
        # Create model
        manager = ModelManager()
        model = manager.create_model("deepseek")
        
        print("\n👤 User: What is Python?")
        print("🤖 AI: ", end="", flush=True)
        
        # Accumulate while streaming - simple Python!
        chunks = []
        for chunk in model.stream("What is Python in one sentence?"):
            token = chunk.content
            print(token, end="", flush=True)
            chunks.append(token)
        
        # Get complete text
        full_text = "".join(chunks)
        
        print(f"\n\n📊 Total characters: {len(full_text)}")
        print(f"📊 Total words: {len(full_text.split())}")
        print(f"✅ Captured: {full_text[:50]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def callback_streaming_example():
    """Example 3: Custom callbacks using LangChain's BaseCallbackHandler."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Callbacks")
    print("=" * 60)
    
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        
        # Create model
        manager = ModelManager()
        model = manager.create_model("deepseek")
        
        # Define custom callback
        class TokenCounter(BaseCallbackHandler):
            def __init__(self):
                super().__init__()
                self.token_count = 0
                self.tokens = []
            
            def on_llm_new_token(self, token: str, **kwargs):
                self.token_count += 1
                self.tokens.append(token)
                print(token, end="", flush=True)
            
            def on_llm_end(self, response, **kwargs):
                print(f"\n\n📊 Tokens received: {self.token_count}")
        
        print("\n👤 User: Tell me about AI")
        print("🤖 AI: ", end="", flush=True)
        
        # Use callback
        callback = TokenCounter()
        for chunk in model.stream(
            "Tell me about AI in one sentence",
            config={"callbacks": [callback]}
        ):
            pass  # Callback handles everything
        
        print(f"✅ Complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def streamhandler_example():
    """Example 4: Using StreamHandler (optional convenience wrapper)."""
    print("\n" + "=" * 60)
    print("Example 4: StreamHandler (Optional)")
    print("=" * 60)
    
    try:
        from ai_toolkit.streaming import StreamHandler
        
        # Create model
        manager = ModelManager()
        model = manager.create_model("deepseek")
        handler = StreamHandler(model)
        
        print("\n👤 User: What is machine learning?")
        print("🤖 AI: ", end="", flush=True)
        
        # StreamHandler is just a convenience wrapper
        for token in handler.stream_tokens("What is machine learning in one sentence?"):
            print(token, end="", flush=True)
        
        print("\n\n✅ Done!")
        print("\n💡 Note: StreamHandler is optional - you can use model.stream() directly!")
        
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
        accumulate_streaming_example,
        callback_streaming_example,
        streamhandler_example,
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
    print("   1. Use model.stream() for simple streaming (simplest!)")
    print("   2. Use lists to accumulate: chunks = []; final = ''.join(chunks)")
    print("   3. Use LangChain's BaseCallbackHandler for custom logic")
    print("   4. StreamHandler is optional - just a convenience wrapper")
    print("\n📚 Learn more: https://docs.langchain.com/oss/python/langchain/streaming/overview")


if __name__ == "__main__":
    run_all_examples()
