#!/usr/bin/env python3
"""
Integration tests for the Streaming Processing Toolkit.

This script tests the streaming functionality with real model interactions
and various streaming scenarios.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_toolkit.streaming import StreamHandler, StreamCallback


def test_basic_streaming():
    """Test basic streaming functionality."""
    print("🧪 Testing Basic Streaming Functionality")
    print("=" * 50)
    
    # Create stream handler
    handler = StreamHandler()
    
    # Test session management
    session_id = handler.start_session("test_session")
    print(f"✅ Started session: {session_id}")
    
    # Simulate streaming chunks
    chunks = ["Hello", " ", "world", "!", " How", " are", " you", "?"]
    
    for i, chunk in enumerate(chunks):
        processed_chunk = handler.handle_stream(
            chunk=chunk,
            metadata={"chunk_index": i, "source": "test"}
        )
        print(f"   📝 Chunk {i}: '{chunk}' -> '{processed_chunk.content}'")
    
    # Test aggregation
    full_content = handler.aggregate_stream(session_id)
    print(f"✅ Aggregated content: '{full_content}'")
    
    # Test statistics
    stats = handler.get_statistics(session_id)
    print(f"✅ Statistics: {stats['chunk_count']} chunks, {stats['total_characters']} chars")
    
    # End session
    completed_session = handler.end_session(session_id)
    print(f"✅ Session completed in {completed_session.duration:.3f}s")
    
    return True


def test_different_chunk_formats():
    """Test handling different chunk formats."""
    print("\n🧪 Testing Different Chunk Formats")
    print("=" * 50)
    
    handler = StreamHandler()
    session_id = handler.start_session()
    
    # Test different formats
    test_chunks = [
        # String format
        "Simple string",
        
        # Dictionary with content
        {"content": "Dict content", "type": "text"},
        
        # Dictionary with text
        {"text": "Dict text", "id": "chunk_123"},
        
        # Delta format (OpenAI style)
        {"delta": {"content": "Delta content"}},
        
        # Complex object
        {"content": "Complex", "metadata": {"model": "test"}, "finish_reason": None},
    ]
    
    for i, chunk in enumerate(test_chunks):
        processed = handler.handle_stream(chunk)
        print(f"   📝 Format {i+1}: {type(chunk).__name__} -> '{processed.content}'")
        if processed.metadata:
            print(f"      Metadata: {processed.metadata}")
    
    full_content = handler.aggregate_stream(session_id)
    print(f"✅ All formats processed: '{full_content}'")
    
    return True


def test_stream_callback():
    """Test StreamCallback functionality."""
    print("\n🧪 Testing Stream Callback")
    print("=" * 50)
    
    # Create handlers
    stream_handler = StreamHandler()
    
    # Track callback events
    events = []
    
    def on_chunk(chunk):
        events.append(f"chunk: {chunk.content}")
    
    def on_complete(content):
        events.append(f"complete: {content}")
    
    def on_error(error):
        events.append(f"error: {error}")
    
    # Create callback
    callback = StreamCallback(
        stream_handler=stream_handler,
        on_chunk_callback=on_chunk,
        on_complete_callback=on_complete,
        on_error_callback=on_error,
        verbose=True
    )
    
    # Simulate LLM interaction
    print("   🚀 Simulating LLM start...")
    callback.on_llm_start({"name": "test_model"}, ["Test prompt"])
    
    print("   📝 Simulating token generation...")
    tokens = ["Hello", " there", "!", " How", " can", " I", " help", "?"]
    for token in tokens:
        callback.on_llm_new_token(token)
    
    print("   ✅ Simulating LLM completion...")
    from langchain_core.outputs import LLMResult, Generation
    result = LLMResult(generations=[[Generation(text="Hello there! How can I help?")]])
    callback.on_llm_end(result)
    
    # Check events
    print(f"✅ Callback events captured: {len(events)}")
    for event in events:
        print(f"   - {event}")
    
    # Check statistics
    stats = callback.get_session_statistics()
    print(f"✅ Session stats: {stats['chunk_count']} chunks, {stats['duration_seconds']:.3f}s")
    
    return True


def test_buffered_streaming():
    """Test buffered streaming callback."""
    print("\n🧪 Testing Buffered Streaming")
    print("=" * 50)
    
    from ai_toolkit.streaming.stream_callback import BufferedStreamCallback
    
    stream_handler = StreamHandler()
    buffered_chunks = []
    
    def on_buffered_chunk(chunk):
        buffered_chunks.append(chunk.content)
        print(f"   📦 Buffered chunk: '{chunk.content}' ({chunk.metadata.get('token_count', 0)} tokens)")
    
    # Create buffered callback with small buffer
    callback = BufferedStreamCallback(
        stream_handler=stream_handler,
        buffer_size=3,
        flush_interval=0.5,
        on_chunk_callback=on_buffered_chunk,
        verbose=True
    )
    
    # Simulate streaming
    callback.on_llm_start({}, ["test"])
    
    # Send tokens (should buffer and flush in groups of 3)
    tokens = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog"]
    for token in tokens:
        callback.on_llm_new_token(token)
        time.sleep(0.05)  # Small delay
    
    # End to flush remaining
    from langchain_core.outputs import LLMResult, Generation
    result = LLMResult(generations=[[Generation(text="The quick brown fox jumps over the lazy dog")]])
    callback.on_llm_end(result)
    
    print(f"✅ Buffered chunks received: {len(buffered_chunks)}")
    for i, chunk in enumerate(buffered_chunks):
        print(f"   {i+1}. '{chunk}'")
    
    return True


def test_multiple_sessions():
    """Test handling multiple concurrent sessions."""
    print("\n🧪 Testing Multiple Sessions")
    print("=" * 50)
    
    handler = StreamHandler()
    
    # Create multiple sessions
    sessions = []
    for i in range(3):
        session_id = handler.start_session(f"session_{i}")
        sessions.append(session_id)
        print(f"   📝 Created session: {session_id}")
    
    # Add content to each session
    for i, session_id in enumerate(sessions):
        content = f"Content for session {i}: "
        words = ["Hello", "from", f"session_{i}"]
        
        for word in words:
            handler.handle_stream(f"{word} ", session_id=session_id)
        
        result = handler.aggregate_stream(session_id)
        print(f"   ✅ Session {i}: '{result.strip()}'")
    
    # Test session listing
    all_sessions = handler.list_sessions()
    print(f"✅ Total sessions: {len(all_sessions)}")
    
    # Test statistics for each session
    for session_id in sessions:
        stats = handler.get_statistics(session_id)
        print(f"   📊 {session_id}: {stats['chunk_count']} chunks, {stats['total_characters']} chars")
    
    return True


def test_stream_iterator():
    """Test stream iterator functionality."""
    print("\n🧪 Testing Stream Iterator")
    print("=" * 50)
    
    handler = StreamHandler()
    session_id = handler.start_session()
    
    # Add some chunks
    chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Final chunk"]
    for chunk in chunks:
        handler.handle_stream(chunk)
    
    handler.end_session(session_id)
    
    # Test iterator
    print("   🔄 Iterating through chunks:")
    for i, chunk in enumerate(handler.stream_iterator(session_id)):
        print(f"   {i+1}. '{chunk.content}' (timestamp: {chunk.timestamp})")
    
    print("✅ Stream iteration completed")
    
    return True


def test_error_handling():
    """Test error handling in streaming."""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    stream_handler = StreamHandler()
    errors_caught = []
    
    def on_error(error):
        errors_caught.append(str(error))
        print(f"   ❌ Error caught: {error}")
    
    callback = StreamCallback(
        stream_handler=stream_handler,
        on_error_callback=on_error
    )
    
    # Test LLM error
    callback.on_llm_start({}, ["test"])
    callback.on_llm_new_token("Hello")
    
    test_error = Exception("Simulated LLM error")
    callback.on_llm_error(test_error)
    
    print(f"✅ Errors handled: {len(errors_caught)}")
    
    return True


def test_performance():
    """Test streaming performance with many chunks."""
    print("\n🧪 Testing Performance")
    print("=" * 50)
    
    handler = StreamHandler()
    session_id = handler.start_session()
    
    # Generate many small chunks
    num_chunks = 1000
    start_time = time.time()
    
    for i in range(num_chunks):
        handler.handle_stream(f"chunk_{i} ")
    
    processing_time = time.time() - start_time
    
    # Test aggregation performance
    start_time = time.time()
    full_content = handler.aggregate_stream(session_id)
    aggregation_time = time.time() - start_time
    
    # Get statistics
    stats = handler.get_statistics(session_id)
    
    print(f"✅ Processed {num_chunks} chunks in {processing_time:.3f}s")
    print(f"   Rate: {num_chunks/processing_time:.0f} chunks/second")
    print(f"✅ Aggregated {len(full_content)} characters in {aggregation_time:.3f}s")
    print(f"   Total characters: {stats['total_characters']}")
    
    return True


def run_all_tests():
    """Run all streaming integration tests."""
    print("🎯 AI Toolkit Streaming Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Streaming", test_basic_streaming),
        ("Chunk Formats", test_different_chunk_formats),
        ("Stream Callback", test_stream_callback),
        ("Buffered Streaming", test_buffered_streaming),
        ("Multiple Sessions", test_multiple_sessions),
        ("Stream Iterator", test_stream_iterator),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results.append((test_name, success, None))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if error:
            print(f"     Error: {error}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All streaming tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)