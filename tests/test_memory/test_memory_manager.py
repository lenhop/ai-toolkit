"""
Test Memory Manager

Tests for ai_toolkit.memory.memory_manager module:
- MemoryManager
- CheckpointerFactory
- MessageTrimmer
- MessageSummarizer
- Middleware creation functions

Based on examples: 17_memory_base.py
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
    from ai_toolkit.memory import (
        MemoryManager,
        CheckpointerFactory,
        MessageTrimmer,
        MessageSummarizer,
        create_trimming_middleware,
        create_deletion_middleware,
        create_summarization_middleware,
        create_dynamic_prompt_middleware,
    )
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_checkpointer_factory():
    """Test CheckpointerFactory."""
    print("\n" + "=" * 80)
    print("Test 1: CheckpointerFactory")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Test: Create in-memory checkpointer
        checkpointer = CheckpointerFactory.create_inmemory()
        print("✓ In-memory checkpointer created")
        
        # Test: Create config
        config = MemoryManager.create_config("test-thread-1")
        print(f"✓ Config created: {config}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_memory_manager():
    """Test MemoryManager."""
    print("\n" + "=" * 80)
    print("Test 2: MemoryManager")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        manager = MemoryManager()
        
        # Test: Create checkpointer
        checkpointer = manager.create_checkpointer("inmemory")
        print("✓ Checkpointer created via MemoryManager")
        
        # Test: Create config
        config = MemoryManager.create_config("user-123")
        print(f"✓ Config created: {config}")
        
        # Test: Create trimmer
        trimmer = MemoryManager.create_trimmer(max_messages=5)
        print("✓ Trimmer created")
        
        # Test: Create summarizer
        summarizer = MemoryManager.create_summarizer()
        print("✓ Summarizer created")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_message_trimmer():
    """Test MessageTrimmer."""
    print("\n" + "=" * 80)
    print("Test 3: MessageTrimmer")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Create test messages
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Message 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Message 2"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Message 3"),
            AIMessage(content="Response 3"),
        ]
        
        # Test: Trim with keep_first_and_recent
        trimmer = MessageTrimmer(
            strategy="keep_first_and_recent",
            max_messages=5
        )
        
        trimmed = trimmer.trim(messages)
        print(f"✓ Trimmed from {len(messages)} to {len(trimmed)} messages")
        print(f"✓ Strategy: keep_first_and_recent")
        
        # Test: Trim with keep_recent
        trimmer2 = MessageTrimmer(
            strategy="keep_recent",
            max_messages=3
        )
        
        trimmed2 = trimmer2.trim(messages)
        print(f"✓ Trimmed from {len(messages)} to {len(trimmed2)} messages")
        print(f"✓ Strategy: keep_recent")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_middleware_creation():
    """Test middleware creation functions."""
    print("\n" + "=" * 80)
    print("Test 4: Middleware Creation Functions")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Test: Create trimming middleware
        trim_middleware = create_trimming_middleware(max_messages=5)
        print("✓ Trimming middleware created")
        
        # Test: Create deletion middleware
        delete_middleware = create_deletion_middleware(delete_count=2, trigger_count=10)
        print("✓ Deletion middleware created")
        
        # Test: Create summarization middleware (requires model)
        from ai_toolkit.models import ModelManager
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        summarization_middleware = create_summarization_middleware(
            model=model,
            trigger_tokens=4000,
            keep_messages=20
        )
        print("✓ Summarization middleware created")
        
        # Test: Create dynamic prompt middleware
        def prompt_generator(context):
            user_name = context.get("user_name", "User")
            return f"You are helpful. Address user as {user_name}."
        
        prompt_middleware = create_dynamic_prompt_middleware(prompt_generator)
        print("✓ Dynamic prompt middleware created")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Memory Manager")
    print("=" * 80)
    
    test_checkpointer_factory()
    test_memory_manager()
    test_message_trimmer()
    test_middleware_creation()
    
    print("\n" + "=" * 80)
    print("Memory Manager Tests Completed")
    print("=" * 80)
