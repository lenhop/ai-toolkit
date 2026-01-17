#!/usr/bin/env python3
"""
Streaming Processing Toolkit Examples

This script demonstrates how to use the streaming processing toolkit
for handling real-time output from AI models.

Examples included:
1. Basic streaming - Simple chunk processing and aggregation
2. Different formats - Handling various chunk formats (OpenAI, Anthropic, etc.)
3. Callbacks - LangChain callback integration
4. Buffered streaming - Efficient batch processing
5. Multi-callback - Multiple concurrent handlers
6. Session management - Managing multiple conversations
7. Stream iteration - Replaying and iterating streams
8. Performance - Large-scale streaming
9. Real model streaming - Actual AI model streaming (requires API key)
10. Real agent streaming - LangChain agent with tools (requires API key)
11. Real-world simulation - Chat interface simulation

Note: Examples 9-10 require valid API keys (DEEPSEEK_API_KEY, QWEN_API_KEY, or GLM_API_KEY)
      to demonstrate real streaming. They will be skipped if no keys are found.
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
else:
    print(f"⚠️  No .env file found at {env_path}")

from ai_toolkit.streaming import StreamHandler, StreamCallback
from ai_toolkit.streaming.stream_callback import BufferedStreamCallback, MultiStreamCallback
from ai_toolkit.models import ModelManager


def check_api_keys():
    """Check which API keys are available."""
    keys = {
        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
        'QWEN_API_KEY': os.getenv('QWEN_API_KEY'),
        'GLM_API_KEY': os.getenv('GLM_API_KEY'),
    }
    
    available = {k: v for k, v in keys.items() if v and not v.startswith('your_')}
    
    if available:
        print(f"🔑 Available API keys: {', '.join(available.keys())}")
    else:
        print("⚠️  No API keys found in environment")
    
    return available


def basic_streaming_example():
    """Demonstrate basic streaming functionality."""
    print("🚀 Basic Streaming Example")
    print("=" * 50)
    
    # Create a stream handler
    handler = StreamHandler(
        buffer_size=100,
        auto_aggregate=True,
        chunk_separator=""
    )
    
    # Start a streaming session
    session_id = handler.start_session("basic_example")
    print(f"Started session: {session_id}")
    
    # Simulate streaming chunks from an AI model
    message_chunks = [
        "Hello", " there", "!", " I'm", " an", " AI", " assistant", ".",
        " How", " can", " I", " help", " you", " today", "?"
    ]
    
    print("\nProcessing streaming chunks:")
    for i, chunk in enumerate(message_chunks):
        processed_chunk = handler.handle_stream(
            chunk=chunk,
            metadata={
                "chunk_id": f"chunk_{i}",
                "timestamp": time.time(),
                "model": "example_model"
            }
        )
        print(f"  Chunk {i+1:2d}: '{chunk}' -> Processed: '{processed_chunk.content}'")
        time.sleep(0.1)  # Simulate streaming delay
    
    # Get the complete message
    complete_message = handler.aggregate_stream(session_id)
    print(f"\n✅ Complete message: '{complete_message}'")
    
    # Get streaming statistics
    stats = handler.get_statistics(session_id)
    print(f"\n📊 Streaming Statistics:")
    print(f"   Chunks: {stats['chunk_count']}")
    print(f"   Characters: {stats['total_characters']}")
    print(f"   Duration: {stats['duration_seconds']:.2f}s")
    print(f"   Rate: {stats['characters_per_second']:.1f} chars/s")
    
    # End the session
    handler.end_session(session_id)
    print(f"✅ Session ended")


def different_formats_example():
    """Demonstrate handling different chunk formats."""
    print("\n🔄 Different Chunk Formats Example")
    print("=" * 50)
    
    handler = StreamHandler()
    session_id = handler.start_session("formats_example")
    
    # Test various chunk formats
    test_chunks = [
        # Simple string
        "Simple text chunk",
        
        # OpenAI-style response
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": " from OpenAI format"}}]
        },
        
        # Anthropic-style response
        {
            "type": "content_block_delta",
            "delta": {"text": " from Anthropic format"}
        },
        
        # Generic dictionary
        {
            "content": " from generic dict",
            "metadata": {"source": "custom_model"}
        },
        
        # LangChain message-like object
        type('MockMessage', (), {
            'content': ' from LangChain format',
            'additional_kwargs': {'model': 'langchain_model'}
        })()
    ]
    
    print("Processing different formats:")
    for i, chunk in enumerate(test_chunks):
        # Handle different formats
        if isinstance(chunk, dict) and "choices" in chunk:
            # OpenAI format
            content = chunk["choices"][0]["delta"].get("content", "")
            processed = handler.handle_stream(content, metadata={"format": "openai"})
        elif isinstance(chunk, dict) and "delta" in chunk and "text" in chunk["delta"]:
            # Anthropic format
            content = chunk["delta"]["text"]
            processed = handler.handle_stream(content, metadata={"format": "anthropic"})
        else:
            # Let handler auto-detect format
            processed = handler.handle_stream(chunk)
        
        print(f"  Format {i+1}: {type(chunk).__name__} -> '{processed.content}'")
    
    complete_text = handler.aggregate_stream(session_id)
    print(f"\n✅ Combined text: '{complete_text}'")


def callback_example():
    """Demonstrate StreamCallback usage."""
    print("\n📞 Stream Callback Example")
    print("=" * 50)
    
    # Create stream handler
    stream_handler = StreamHandler()
    
    # Track events
    events = []
    
    def on_chunk_received(chunk):
        events.append(f"CHUNK: '{chunk.content}'")
        print(f"  📝 Received chunk: '{chunk.content}' (metadata: {chunk.metadata})")
    
    def on_stream_complete(final_content):
        events.append(f"COMPLETE: '{final_content}'")
        print(f"  ✅ Stream complete: '{final_content}'")
    
    def on_stream_error(error):
        events.append(f"ERROR: {error}")
        print(f"  ❌ Stream error: {error}")
    
    # Create callback with event handlers
    callback = StreamCallback(
        stream_handler=stream_handler,
        on_chunk_callback=on_chunk_received,
        on_complete_callback=on_stream_complete,
        on_error_callback=on_stream_error,
        verbose=True
    )
    
    # Simulate LangChain model interaction
    print("Simulating LangChain model streaming:")
    
    # Start LLM
    callback.on_llm_start(
        serialized={"name": "example_model"},
        prompts=["Tell me a short joke"]
    )
    
    # Stream tokens
    joke_tokens = [
        "Why", " don't", " scientists", " trust", " atoms", "?",
        " Because", " they", " make", " up", " everything", "!"
    ]
    
    for token in joke_tokens:
        callback.on_llm_new_token(token)
        time.sleep(0.1)
    
    # End LLM
    from langchain_core.outputs import LLMResult, Generation
    result = LLMResult(
        generations=[[Generation(text="Why don't scientists trust atoms? Because they make up everything!")]]
    )
    callback.on_llm_end(result)
    
    print(f"\n📋 Events captured: {len(events)}")
    for event in events:
        print(f"  - {event}")


def buffered_streaming_example():
    """Demonstrate buffered streaming."""
    print("\n📦 Buffered Streaming Example")
    print("=" * 50)
    
    stream_handler = StreamHandler()
    
    def on_buffered_chunk(chunk):
        token_count = chunk.metadata.get('token_count', 0)
        print(f"  📦 Buffered chunk ({token_count} tokens): '{chunk.content}'")
    
    # Create buffered callback
    buffered_callback = BufferedStreamCallback(
        stream_handler=stream_handler,
        buffer_size=5,  # Buffer 5 tokens at a time
        flush_interval=2.0,  # Or flush every 2 seconds
        on_chunk_callback=on_buffered_chunk,
        verbose=True
    )
    
    print("Streaming with buffering (5 tokens per batch):")
    
    # Start streaming
    buffered_callback.on_llm_start({}, ["Generate a sentence"])
    
    # Stream many small tokens
    sentence_tokens = [
        "The", " quick", " brown", " fox", " jumps",  # First batch (5 tokens)
        " over", " the", " lazy", " dog", " in",      # Second batch (5 tokens)
        " the", " sunny", " meadow", "."              # Final batch (4 tokens, flushed on end)
    ]
    
    for token in sentence_tokens:
        buffered_callback.on_llm_new_token(token)
        time.sleep(0.1)
    
    # End streaming (flushes remaining buffer)
    from langchain_core.outputs import LLMResult, Generation
    result = LLMResult(
        generations=[[Generation(text="The quick brown fox jumps over the lazy dog in the sunny meadow.")]]
    )
    buffered_callback.on_llm_end(result)
    
    print(f"✅ Final content: '{buffered_callback.get_accumulated_content()}'")


def multi_callback_example():
    """Demonstrate multiple callbacks."""
    print("\n🔀 Multi-Callback Example")
    print("=" * 50)
    
    # Create multiple stream handlers for different purposes
    console_handler = StreamHandler()
    file_handler = StreamHandler()
    
    # Console callback (immediate display)
    console_callback = StreamCallback(
        stream_handler=console_handler,
        on_chunk_callback=lambda chunk: print(f"  🖥️  Console: '{chunk.content}'"),
        session_id="console_session"
    )
    
    # File callback (buffered for efficiency)
    file_chunks = []
    file_callback = BufferedStreamCallback(
        stream_handler=file_handler,
        buffer_size=3,
        on_chunk_callback=lambda chunk: file_chunks.append(chunk.content),
        session_id="file_session"
    )
    
    # Combine callbacks
    multi_callback = MultiStreamCallback([console_callback, file_callback])
    
    print("Streaming to multiple handlers:")
    
    # Start streaming
    multi_callback.on_llm_start({}, ["Write a haiku"])
    
    # Stream haiku
    haiku_tokens = [
        "Cherry", " blossoms", " fall", "\n",
        "Gentle", " breeze", " carries", " petals", "\n",
        "Spring's", " fleeting", " beauty"
    ]
    
    for token in haiku_tokens:
        multi_callback.on_llm_new_token(token)
        time.sleep(0.2)
    
    # End streaming
    from langchain_core.outputs import LLMResult, Generation
    result = LLMResult(
        generations=[[Generation(text="Cherry blossoms fall\nGentle breeze carries petals\nSpring's fleeting beauty")]]
    )
    multi_callback.on_llm_end(result)
    
    # Show results from different handlers
    console_content = console_handler.aggregate_stream("console_session")
    file_content = "".join(file_chunks)
    
    print(f"\n📊 Results:")
    print(f"  Console content: '{console_content}'")
    print(f"  File chunks: {len(file_chunks)} buffered chunks")
    print(f"  File content: '{file_content}'")


def session_management_example():
    """Demonstrate session management."""
    print("\n🗂️  Session Management Example")
    print("=" * 50)
    
    handler = StreamHandler(buffer_size=10)
    
    # Create multiple sessions for different conversations
    sessions = {}
    
    conversations = [
        ("user_1", ["Hello", " AI", "!", " How", " are", " you", "?"]),
        ("user_2", ["What's", " the", " weather", " like", "?"]),
        ("user_3", ["Tell", " me", " a", " joke", " please", "."]),
    ]
    
    print("Creating multiple conversation sessions:")
    
    # Process each conversation
    for user, tokens in conversations:
        session_id = handler.start_session(f"conversation_{user}")
        sessions[user] = session_id
        print(f"\n  👤 {user} (session: {session_id}):")
        
        for token in tokens:
            chunk = handler.handle_stream(
                chunk=token,
                session_id=session_id,
                metadata={"user": user, "timestamp": time.time()}
            )
            print(f"    '{token}'", end="")
        
        # Complete the conversation
        handler.end_session(session_id)
        content = handler.aggregate_stream(session_id)
        print(f"\n    Complete: '{content}'")
    
    # Show session statistics
    print(f"\n📊 Session Statistics:")
    for user, session_id in sessions.items():
        stats = handler.get_statistics(session_id)
        print(f"  {user}: {stats['chunk_count']} chunks, {stats['duration_seconds']:.2f}s")
    
    # List all sessions
    all_sessions = handler.list_sessions()
    print(f"\n📋 Total sessions: {len(all_sessions)}")


def stream_iterator_example():
    """Demonstrate stream iteration."""
    print("\n🔄 Stream Iterator Example")
    print("=" * 50)
    
    handler = StreamHandler()
    session_id = handler.start_session("iterator_example")
    
    # Add chunks with delays to simulate real streaming
    story_chunks = [
        "Once upon a time,",
        " in a land far away,",
        " there lived a brave knight",
        " who sought to find",
        " the legendary treasure.",
        " After many adventures,",
        " the knight succeeded!"
    ]
    
    print("Adding chunks to stream:")
    for chunk in story_chunks:
        handler.handle_stream(chunk)
        print(f"  Added: '{chunk}'")
    
    handler.end_session(session_id)
    
    # Iterate through the stream
    print(f"\nIterating through stream chunks:")
    for i, chunk in enumerate(handler.stream_iterator(session_id)):
        print(f"  {i+1}. '{chunk.content}' (time: {chunk.timestamp})")
    
    # Show complete story
    complete_story = handler.aggregate_stream(session_id)
    print(f"\n📖 Complete story: '{complete_story}'")


def performance_example():
    """Demonstrate performance with large streams."""
    print("\n⚡ Performance Example")
    print("=" * 50)
    
    handler = StreamHandler()
    session_id = handler.start_session("performance_test")
    
    # Generate a large number of chunks
    num_chunks = 1000
    chunk_size = 10
    
    print(f"Processing {num_chunks} chunks of {chunk_size} characters each...")
    
    start_time = time.time()
    
    for i in range(num_chunks):
        chunk_content = f"chunk_{i:04d}_" + "x" * (chunk_size - len(f"chunk_{i:04d}_"))
        handler.handle_stream(chunk_content)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_chunks} chunks...")
    
    processing_time = time.time() - start_time
    
    # Test aggregation performance
    print("Aggregating stream...")
    start_time = time.time()
    full_content = handler.aggregate_stream(session_id)
    aggregation_time = time.time() - start_time
    
    # Get statistics
    stats = handler.get_statistics(session_id)
    
    print(f"\n📊 Performance Results:")
    print(f"  Processing: {num_chunks} chunks in {processing_time:.3f}s")
    print(f"  Rate: {num_chunks/processing_time:.0f} chunks/second")
    print(f"  Aggregation: {len(full_content)} chars in {aggregation_time:.3f}s")
    print(f"  Total size: {stats['total_characters']} characters")
    print(f"  Average chunk size: {stats['average_chunk_size']:.1f} characters")


def real_model_streaming_example():
    """Demonstrate streaming with real AI model."""
    print("\n🤖 Real Model Streaming Example")
    print("=" * 50)
    
    try:
        # Initialize model manager
        manager = ModelManager()
        
        # Try to create a model using environment variables
        # ModelManager will automatically use env vars for API keys
        model = None
        model_name = None
        
        for provider in ['deepseek', 'qwen', 'glm']:
            try:
                # Check if API key exists for this provider
                api_key_var = f"{provider.upper()}_API_KEY"
                api_key = os.getenv(api_key_var)
                
                if not api_key or api_key.startswith('your_'):
                    continue
                
                # Try to create model
                model = manager.create_model(provider)
                model_name = provider
                print(f"✅ Using {provider.upper()} model for streaming")
                break
            except Exception as e:
                print(f"⚠️  Failed to create {provider} model: {e}")
                continue
        
        if model is None:
            print("⚠️  No API keys found in environment. Skipping real model example.")
            print("   Set DEEPSEEK_API_KEY, QWEN_API_KEY, or GLM_API_KEY to test real streaming.")
            return
        
        # Create stream handler and callback
        handler = StreamHandler()
        
        # Custom callback for real-time display
        class RealTimeCallback(StreamCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.display_buffer = ""
            
            def on_llm_new_token(self, token: str, **kwargs):
                super().on_llm_new_token(token, **kwargs)
                self.display_buffer += token
                print(token, end="", flush=True)
            
            def on_llm_end(self, response, **kwargs):
                super().on_llm_end(response, **kwargs)
                print()  # New line
        
        callback = RealTimeCallback(stream_handler=handler, verbose=False)
        
        # Test streaming with a simple prompt
        print(f"\n� User: Tell me a short joke about programming")
        print(f"🤖 AI ({model_name}): ", end="", flush=True)
        
        # Stream the response using config
        from langchain_core.runnables import RunnableConfig
        config = RunnableConfig(callbacks=[callback])
        
        # Use stream with config for callback support
        for chunk in model.stream("Tell me a short joke about programming", config=config):
            pass  # Callback handles display
        
        # Get statistics
        stats = handler.get_statistics()
        if stats:
            print(f"\n📊 Streaming stats: {stats['chunk_count']} chunks, {stats['characters_per_second']:.1f} chars/s")
        
    except Exception as e:
        print(f"❌ Error in real model streaming: {e}")
        print("   Make sure you have valid API keys configured.")
        import traceback
        traceback.print_exc()


def real_agent_streaming_example():
    """Demonstrate streaming with LangChain agent."""
    print("\n🤖 Real Agent Streaming Example")
    print("=" * 50)
    
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import PromptTemplate
        from langchain.tools import tool
        
        # Initialize model manager
        manager = ModelManager()
        
        # Try to create a model
        model = None
        model_name = None
        
        for provider in ['deepseek', 'qwen', 'glm']:
            try:
                # Check if API key exists
                api_key_var = f"{provider.upper()}_API_KEY"
                api_key = os.getenv(api_key_var)
                
                if not api_key or api_key.startswith('your_'):
                    continue
                
                model = manager.create_model(provider)
                model_name = provider
                print(f"✅ Using {provider.upper()} model for agent streaming")
                break
            except Exception as e:
                print(f"⚠️  Failed to create {provider} model: {e}")
                continue
        
        if model is None:
            print("⚠️  No API keys found. Skipping agent streaming example.")
            return
        
        # Create a simple tool
        @tool
        def get_word_length(word: str) -> int:
            """Get the length of a word."""
            return len(word)
        
        tools = [get_word_length]
        
        # Create agent prompt (ReAct style)
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
        
        # Create agent
        agent = create_react_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
        
        # Create streaming callback
        handler = StreamHandler()
        
        class AgentStreamCallback(StreamCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.in_tool = False
            
            def on_tool_start(self, serialized, input_str, **kwargs):
                self.in_tool = True
                print(f"\n🔧 Using tool: {serialized.get('name', 'unknown')}")
            
            def on_tool_end(self, output, **kwargs):
                self.in_tool = False
                print(f"   Result: {output}")
            
            def on_llm_new_token(self, token: str, **kwargs):
                if not self.in_tool:
                    super().on_llm_new_token(token, **kwargs)
                    print(token, end="", flush=True)
            
            def on_llm_end(self, response, **kwargs):
                if not self.in_tool:
                    super().on_llm_end(response, **kwargs)
                    print()
        
        callback = AgentStreamCallback(stream_handler=handler, verbose=False)
        
        # Run agent with streaming
        print(f"\n👤 User: What is the length of the word 'streaming'?")
        print(f"🤖 Agent ({model_name}): ", end="", flush=True)
        
        result = agent_executor.invoke(
            {"input": "What is the length of the word 'streaming'?"},
            {"callbacks": [callback]}
        )
        
        print(f"\n✅ Final answer: {result['output']}")
        
    except ImportError as e:
        print(f"⚠️  Missing dependencies for agent example: {e}")
    except Exception as e:
        print(f"❌ Error in agent streaming: {e}")
        import traceback
        traceback.print_exc()


def real_world_example():
    """Demonstrate real-world usage patterns."""
    print("\n🌍 Real-World Usage Example")
    print("=" * 50)
    
    # Simulate a chatbot conversation with streaming
    handler = StreamHandler()
    
    # Custom callback for chat display
    class ChatDisplayCallback(StreamCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.display_buffer = ""
        
        def on_llm_new_token(self, token: str, **kwargs):
            super().on_llm_new_token(token, **kwargs)
            self.display_buffer += token
            # Simulate real-time display update
            print(f"\r🤖 AI: {self.display_buffer}", end="", flush=True)
        
        def on_llm_end(self, response, **kwargs):
            super().on_llm_end(response, **kwargs)
            print()  # New line after completion
    
    # Create chat callback
    chat_callback = ChatDisplayCallback(
        stream_handler=handler,
        verbose=False
    )
    
    # Simulate conversation
    conversations = [
        "Hello! How can I help you today?",
        "I can assist with various tasks like answering questions, writing, and problem-solving.",
        "What would you like to know?"
    ]
    
    print("Simulating real-time chat responses:")
    
    for i, response in enumerate(conversations):
        print(f"\n👤 User: Question {i+1}")
        
        # Start response
        chat_callback.on_llm_start({}, [f"Question {i+1}"])
        
        # Stream response word by word
        words = response.split()
        for j, word in enumerate(words):
            token = word if j == 0 else f" {word}"
            chat_callback.on_llm_new_token(token)
            time.sleep(0.1)  # Simulate typing speed
        
        # End response
        from langchain_core.outputs import LLMResult, Generation
        result = LLMResult(generations=[[Generation(text=response)]])
        chat_callback.on_llm_end(result)
        
        time.sleep(0.5)  # Pause between responses
    
    print("\n✅ Chat simulation complete!")


def run_all_examples():
    """Run all streaming examples."""
    print("🎯 AI Toolkit Streaming Examples")
    print("=" * 60)
    
    # Check API keys
    check_api_keys()
    print()
    
    examples = [
        basic_streaming_example,
        different_formats_example,
        callback_example,
        buffered_streaming_example,
        multi_callback_example,
        session_management_example,
        stream_iterator_example,
        performance_example,
        real_model_streaming_example,  # Real model streaming
        real_agent_streaming_example,  # Real agent streaming
        real_world_example,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            if i > 1:
                print(f"\n{'='*20} Example {i} {'='*20}")
            example_func()
        except KeyboardInterrupt:
            print("\n\n⏹️  Examples interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in example: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Streaming Examples Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Basic streaming and chunk processing")
    print("   ✅ Multiple chunk format handling")
    print("   ✅ LangChain callback integration")
    print("   ✅ Buffered streaming for efficiency")
    print("   ✅ Multiple concurrent callbacks")
    print("   ✅ Session management and statistics")
    print("   ✅ Stream iteration and replay")
    print("   ✅ Performance with large streams")
    print("   ✅ Real model streaming (with API)")
    print("   ✅ Real agent streaming (with tools)")
    print("   ✅ Real-world chat simulation")


if __name__ == "__main__":
    run_all_examples()