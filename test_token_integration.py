#!/usr/bin/env python3
"""
Integration tests for the Token Toolkit.

This script tests the token counting and optimization functionality
with real scenarios and various text types.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_toolkit.tokens import TokenCounter, TokenOptimizer
from ai_toolkit.tokens.token_counter import ModelType, TokenUsage, CostEstimate
from ai_toolkit.tokens.token_optimizer import OptimizationStrategy

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def test_basic_token_counting():
    """Test basic token counting functionality."""
    print("🧪 Testing Basic Token Counting")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Test different types of text
    test_texts = [
        "Hello, world!",
        "This is a longer sentence with more words to count tokens for.",
        "Short text",
        "",
        "Text with numbers: 123, 456, 789 and symbols: @#$%^&*()",
        "Multi-line text\nwith line breaks\nand different content.",
        "Very long text that goes on and on with many words and should result in a higher token count than shorter texts because it contains more information and complexity."
    ]
    
    print("Token counts for different texts:")
    for i, text in enumerate(test_texts, 1):
        token_count = counter.count_tokens(text)
        char_count = len(text)
        word_count = len(text.split()) if text else 0
        
        print(f"  {i}. Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"     Tokens: {token_count}, Characters: {char_count}, Words: {word_count}")
        
        if token_count > 0 and word_count > 0:
            ratio = token_count / word_count
            print(f"     Tokens per word: {ratio:.2f}")
    
    return True


def test_message_token_counting():
    """Test token counting for messages."""
    print("\n🧪 Testing Message Token Counting")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Test different message formats
    langchain_messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Hello, how are you today?"),
        AIMessage(content="I'm doing well, thank you for asking! How can I help you?"),
        HumanMessage(content="Can you explain quantum computing?")
    ]
    
    dict_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you?"},
        {"role": "user", "content": "Can you explain quantum computing?"}
    ]
    
    print("1. LangChain Message Format:")
    for i, message in enumerate(langchain_messages, 1):
        token_count = counter.count_message_tokens(message)
        role = type(message).__name__.replace("Message", "").lower()
        content_preview = str(message.content)[:40] + "..." if len(str(message.content)) > 40 else str(message.content)
        print(f"   {i}. {role}: '{content_preview}' -> {token_count} tokens")
    
    langchain_usage = counter.count_messages_tokens(langchain_messages)
    print(f"   Total conversation: {langchain_usage.total_tokens} tokens")
    
    print("\n2. Dictionary Message Format:")
    for i, message in enumerate(dict_messages, 1):
        token_count = counter.count_message_tokens(message)
        content_preview = message['content'][:40] + "..." if len(message['content']) > 40 else message['content']
        print(f"   {i}. {message['role']}: '{content_preview}' -> {token_count} tokens")
    
    dict_usage = counter.count_messages_tokens(dict_messages)
    print(f"   Total conversation: {dict_usage.total_tokens} tokens")
    
    return True


def test_cost_estimation():
    """Test cost estimation functionality."""
    print("\n🧪 Testing Cost Estimation")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Test cost estimation for different scenarios
    scenarios = [
        ("Short prompt", 50, 25),
        ("Medium prompt", 500, 200),
        ("Long prompt", 2000, 800),
        ("Very long prompt", 8000, 2000)
    ]
    
    print("Cost estimates for different usage scenarios:")
    for scenario_name, prompt_tokens, completion_tokens in scenarios:
        cost = counter.estimate_cost(prompt_tokens, completion_tokens)
        
        print(f"  {scenario_name}:")
        print(f"    Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
        print(f"    Prompt cost: ${cost.prompt_cost:.4f}")
        print(f"    Completion cost: ${cost.completion_cost:.4f}")
        print(f"    Total cost: ${cost.total_cost:.4f}")
    
    # Compare costs across models
    print(f"\nCost comparison across models (1000 prompt + 500 completion tokens):")
    comparison = counter.compare_models(1000, 500)
    
    sorted_models = sorted(comparison.items(), key=lambda x: x[1].total_cost)
    for model_name, cost in sorted_models:
        print(f"  {model_name:20}: ${cost.total_cost:.4f}")
    
    return True


def test_text_analysis():
    """Test text analysis functionality."""
    print("\n🧪 Testing Text Analysis")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Analyze different types of text
    texts_to_analyze = [
        "Simple sentence.",
        "This is a more complex sentence with various words and punctuation marks!",
        "Technical text with jargon: API, JSON, HTTP, REST, microservices, containerization.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Code-like text: def function(param): return param * 2 + 1"
    ]
    
    print("Text analysis results:")
    for i, text in enumerate(texts_to_analyze, 1):
        analysis = counter.analyze_text(text)
        
        print(f"\n  {i}. Text: '{text}'")
        print(f"     Tokens: {analysis['token_count']}")
        print(f"     Characters: {analysis['character_count']}")
        print(f"     Words: {analysis['word_count']}")
        print(f"     Tokens per word: {analysis['tokens_per_word']:.2f}")
        print(f"     Characters per token: {analysis['characters_per_token']:.2f}")
        print(f"     Estimated cost: ${analysis['estimated_cost'].total_cost:.6f}")
    
    return True


def test_token_optimization():
    """Test token optimization functionality."""
    print("\n🧪 Testing Token Optimization")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Test text compression
    print("1. Text Compression:")
    redundant_text = "This is a very very really quite pretty text with um extra   spaces and... redundant punctuation!!! It has like many filler words you know."
    
    compression_result = optimizer.compress_text(redundant_text)
    
    print(f"   Original: '{redundant_text}'")
    print(f"   Compressed: '{compression_result.optimized_text}'")
    print(f"   Tokens saved: {compression_result.tokens_saved}")
    print(f"   Compression ratio: {compression_result.compression_ratio:.2f}")
    print(f"   Savings: {compression_result.savings_percentage:.1f}%")
    
    # Test text truncation
    print("\n2. Text Truncation:")
    long_text = "This is a very long text that needs to be truncated to fit within a specific token limit. It contains multiple sentences and should be shortened using different strategies. The truncation should preserve meaning while reducing token count."
    
    strategies = [
        OptimizationStrategy.TRUNCATE_END,
        OptimizationStrategy.TRUNCATE_START,
        OptimizationStrategy.TRUNCATE_MIDDLE
    ]
    
    for strategy in strategies:
        result = optimizer.truncate_text(long_text, max_tokens=20, strategy=strategy)
        print(f"   {strategy.value}:")
        print(f"     Result: '{result.optimized_text}'")
        print(f"     Tokens: {result.original_tokens} -> {result.optimized_tokens}")
        print(f"     Saved: {result.tokens_saved} ({result.savings_percentage:.1f}%)")
    
    return True


def test_message_optimization():
    """Test message optimization functionality."""
    print("\n🧪 Testing Message Optimization")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Create a long conversation
    long_conversation = [
        SystemMessage(content="You are a helpful AI assistant specialized in providing detailed explanations."),
        HumanMessage(content="Hello, I need help understanding machine learning concepts."),
        AIMessage(content="I'd be happy to help you understand machine learning! Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and make decisions from data."),
        HumanMessage(content="Can you explain supervised learning in detail?"),
        AIMessage(content="Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The goal is to learn a mapping from inputs to outputs based on example input-output pairs."),
        HumanMessage(content="What about unsupervised learning?"),
        AIMessage(content="Unsupervised learning works with data that doesn't have labeled examples. Instead, it tries to find hidden patterns or structures in the data without being told what to look for."),
        HumanMessage(content="Can you give me examples of both?")
    ]
    
    # Calculate original token usage
    original_usage = counter.count_messages_tokens(long_conversation)
    print(f"Original conversation: {len(long_conversation)} messages, {original_usage.total_tokens} tokens")
    
    # Optimize for different token limits
    token_limits = [100, 50, 30]
    
    for limit in token_limits:
        optimized_messages, result = optimizer.optimize_messages(
            long_conversation, 
            max_tokens=limit,
            preserve_system_message=True,
            preserve_last_messages=2
        )
        
        print(f"\nOptimized for {limit} tokens:")
        print(f"  Messages: {len(long_conversation)} -> {len(optimized_messages)}")
        print(f"  Tokens: {result.original_tokens} -> {result.optimized_tokens}")
        print(f"  Saved: {result.tokens_saved} tokens ({result.savings_percentage:.1f}%)")
        
        # Show preserved messages
        print(f"  Preserved messages:")
        for i, msg in enumerate(optimized_messages):
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "unknown"
            
            content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
            print(f"    {i+1}. {role}: '{content_preview}'")
    
    return True


def test_batch_optimization():
    """Test batch optimization functionality."""
    print("\n🧪 Testing Batch Optimization")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Create multiple texts for batch processing
    texts = [
        "This is the first text that needs optimization for token efficiency.",
        "Second text with very very redundant words and um unnecessary filler content.",
        "Third text containing technical jargon: API endpoints, JSON serialization, HTTP protocols.",
        "Fourth text with a longer narrative that describes various concepts and ideas in detail.",
        "Fifth and final text for batch processing with mixed content types and structures."
    ]
    
    print(f"Batch optimizing {len(texts)} texts:")
    
    # Test different optimization strategies
    strategies = [
        OptimizationStrategy.COMPRESS,
        OptimizationStrategy.TRUNCATE_END,
        OptimizationStrategy.SUMMARIZE
    ]
    
    for strategy in strategies:
        print(f"\n  Strategy: {strategy.value}")
        results = optimizer.batch_optimize(texts, max_tokens_per_text=15, strategy=strategy)
        
        total_original = sum(r.original_tokens for r in results)
        total_optimized = sum(r.optimized_tokens for r in results)
        total_saved = sum(r.tokens_saved for r in results)
        
        print(f"    Total tokens: {total_original} -> {total_optimized}")
        print(f"    Tokens saved: {total_saved} ({(total_saved/total_original)*100:.1f}%)")
        
        # Show individual results
        for i, result in enumerate(results, 1):
            print(f"    Text {i}: {result.original_tokens} -> {result.optimized_tokens} tokens")
    
    # Get optimization statistics
    compress_results = optimizer.batch_optimize(texts, max_tokens_per_text=15, strategy=OptimizationStrategy.COMPRESS)
    stats = optimizer.get_optimization_stats(compress_results)
    
    print(f"\nOptimization Statistics:")
    print(f"  Total texts processed: {stats['total_texts']}")
    print(f"  Average compression ratio: {stats['average_compression_ratio']:.2f}")
    print(f"  Average savings: {stats['average_savings_percentage']:.1f}%")
    print(f"  Strategies used: {', '.join(stats['strategies_used'])}")
    
    return True


def test_cost_optimization():
    """Test cost-based optimization."""
    print("\n🧪 Testing Cost-Based Optimization")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Test optimizing for different cost limits
    expensive_text = "This is a very long and detailed text that would be expensive to process with a high-cost model. It contains extensive information and would result in significant token usage and associated costs. The text goes on with more details and explanations that increase the overall token count substantially."
    
    original_tokens = counter.count_tokens(expensive_text)
    original_cost = counter.estimate_cost(original_tokens)
    
    print(f"Original text: {original_tokens} tokens, ${original_cost.total_cost:.4f}")
    
    # Test different cost limits
    cost_limits = [0.01, 0.005, 0.001]
    
    for cost_limit in cost_limits:
        result = optimizer.optimize_for_cost(expensive_text, cost_limit)
        optimized_cost = counter.estimate_cost(result.optimized_tokens)
        
        print(f"\nOptimized for ${cost_limit:.3f} limit:")
        print(f"  Tokens: {result.original_tokens} -> {result.optimized_tokens}")
        print(f"  Cost: ${original_cost.total_cost:.4f} -> ${optimized_cost.total_cost:.4f}")
        print(f"  Savings: {result.savings_percentage:.1f}%")
        print(f"  Text preview: '{result.optimized_text[:100]}...'")
    
    return True


def test_conversation_cost_calculation():
    """Test conversation cost calculation."""
    print("\n🧪 Testing Conversation Cost Calculation")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Create a realistic conversation
    conversation = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="I need help with Python programming."),
        AIMessage(content="I'd be happy to help with Python! What specific topic would you like to learn about?"),
        HumanMessage(content="How do I work with lists and dictionaries?"),
        AIMessage(content="Lists and dictionaries are fundamental data structures in Python. Lists are ordered collections that can hold multiple items, while dictionaries store key-value pairs."),
        HumanMessage(content="Can you show me some examples?")
    ]
    
    # Calculate conversation cost with estimated response
    estimated_response_tokens = 150
    usage, cost = counter.calculate_conversation_cost(conversation, estimated_response_tokens)
    
    print(f"Conversation Analysis:")
    print(f"  Messages: {len(conversation)}")
    print(f"  Prompt tokens: {usage.prompt_tokens}")
    print(f"  Estimated response tokens: {usage.completion_tokens}")
    print(f"  Total tokens: {usage.total_tokens}")
    print(f"  Estimated cost: ${cost.total_cost:.4f}")
    
    # Compare costs across different models
    print(f"\nCost comparison across models:")
    models_to_compare = [ModelType.GPT_4, ModelType.GPT_3_5_TURBO, ModelType.CLAUDE_3_SONNET]
    
    for model in models_to_compare:
        model_counter = TokenCounter(model=model)
        _, model_cost = model_counter.calculate_conversation_cost(conversation, estimated_response_tokens)
        print(f"  {model.value:20}: ${model_cost.total_cost:.4f}")
    
    return True


def test_model_information():
    """Test model information and pricing."""
    print("\n🧪 Testing Model Information")
    print("=" * 50)
    
    counter = TokenCounter()
    
    # Get supported models
    supported_models = counter.get_supported_models()
    print(f"Supported models ({len(supported_models)}):")
    
    for model_name in supported_models:
        try:
            model_type = ModelType(model_name)
            info = counter.get_model_info(model_type)
            
            print(f"  {model_name:25}:")
            print(f"    Has pricing: {info['has_pricing']}")
            if info['has_pricing']:
                print(f"    Prompt: ${info['prompt_price_per_1k']:.4f}/1K tokens")
                print(f"    Completion: ${info['completion_price_per_1k']:.4f}/1K tokens")
            print(f"    Tokenizer available: {info['tokenizer_available']}")
        except ValueError:
            print(f"  {model_name}: Model type not found")
    
    return True


def run_all_tests():
    """Run all token toolkit integration tests."""
    print("🎯 AI Toolkit Token Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Token Counting", test_basic_token_counting),
        ("Message Token Counting", test_message_token_counting),
        ("Cost Estimation", test_cost_estimation),
        ("Text Analysis", test_text_analysis),
        ("Token Optimization", test_token_optimization),
        ("Message Optimization", test_message_optimization),
        ("Batch Optimization", test_batch_optimization),
        ("Cost Optimization", test_cost_optimization),
        ("Conversation Cost Calculation", test_conversation_cost_calculation),
        ("Model Information", test_model_information),
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
        print("🎉 All token toolkit tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)