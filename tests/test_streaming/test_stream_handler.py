"""
Test Stream Handler

Tests for ai_toolkit.streaming.stream_handler module:
- StreamHandler class and methods

Based on examples: 18_streaming_base.py
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
    from ai_toolkit.streaming import StreamHandler
    from ai_toolkit.models import ModelManager
    from langchain_core.messages import HumanMessage
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_stream_handler_tokens():
    """Test StreamHandler stream_tokens method."""
    print("\n" + "=" * 80)
    print("Test 1: StreamHandler stream_tokens")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        # Create handler
        handler = StreamHandler(model)
        print("✓ StreamHandler created")
        
        # Test: Stream tokens
        print("\nStreaming tokens:")
        tokens = []
        for token in handler.stream_tokens("Count from 1 to 5"):
            print(token, end="", flush=True)
            tokens.append(token)
        
        print(f"\n✓ Received {len(tokens)} tokens")
        print(f"✓ Full response: {''.join(tokens)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_stream_handler_with_messages():
    """Test StreamHandler with message objects."""
    print("\n" + "=" * 80)
    print("Test 2: StreamHandler with Message Objects")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        handler = StreamHandler(model)
        
        # Test: Stream with HumanMessage
        messages = [HumanMessage(content="What is Python?")]
        
        print("\nStreaming with HumanMessage:")
        tokens = []
        for token in handler.stream_tokens(messages):
            print(token, end="", flush=True)
            tokens.append(token)
        
        print(f"\n✓ Received {len(tokens)} tokens")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Stream Handler")
    print("=" * 80)
    
    test_stream_handler_tokens()
    test_stream_handler_with_messages()
    
    print("\n" + "=" * 80)
    print("Stream Handler Tests Completed")
    print("=" * 80)
