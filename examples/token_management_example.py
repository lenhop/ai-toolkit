#!/usr/bin/env python3
"""
Token Management Toolkit Examples

This script demonstrates how to use the token management toolkit
for counting tokens, estimating costs, and optimizing token usage.
"""

import os
import sys
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_toolkit.tokens import TokenCounter, TokenOptimizer
from ai_toolkit.tokens.token_counter import ModelType, TokenUsage, CostEstimate, ModelPricing
from ai_toolkit.tokens.token_optimizer import OptimizationStrategy

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def basic_token_counting_examples():
    """Demonstrate basic token counting functionality."""
    print("🔢 Basic Token Counting Examples")
    print("=" * 50)
    
    # Create token counter for different models
    gpt4_counter = TokenCounter(model=ModelType.GPT_4)
    gpt35_counter = TokenCounter(model=ModelType.GPT_3_5_TURBO)
    
    # Test various text types
    sample_texts = [
        "Hello, world!",
        "This is a longer sentence with more complex vocabulary and structure.",
        "Technical text: API, JSON, HTTP, REST, microservices, containerization, Kubernetes.",
        "Code snippet: def calculate_tokens(text): return len(tokenizer.encode(text))",
        "Multi-line text\nwith line breaks\nand different formatting.",
        "Text with emojis 🚀 and special characters: @#$%^&*()!",
        ""  # Empty text
    ]
    
    print("Token counts for different text types:")
    for i, text in enumerate(sample_texts, 1):
        if text:  # Skip empty text for display
            gpt4_tokens = gpt4_counter.count_tokens(text)
            gpt35_tokens = gpt35_counter.count_tokens(text)
            
            print(f"\n{i}. Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
            print(f"   GPT-4 tokens: {gpt4_tokens}")
            print(f"   GPT-3.5 tokens: {gpt35_tokens}")
            print(f"   Characters: {len(text)}")
            print(f"   Words: {len(text.split())}")
        else:
            print(f"\n{i}. Empty text: 0 tokens")
    
    # Demonstrate batch counting
    print(f"\nBatch token counting:")
    batch_counts = gpt4_counter.batch_count_tokens(sample_texts[:-1])  # Exclude empty text
    for text, count in zip(sample_texts[:-1], batch_counts):
        preview = text[:30] + "..." if len(text) > 30 else text
        print(f"   '{preview}': {count} tokens")


def message_token_counting_examples():
    """Demonstrate message token counting."""
    print("\n💬 Message Token Counting Examples")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Example 1: LangChain message format
    print("1. LangChain Message Format:")
    langchain_messages = [
        SystemMessage(content="You are a helpful AI assistant specialized in programming."),
        HumanMessage(content="Can you help me understand Python decorators?"),
        AIMessage(content="Certainly! Python decorators are a powerful feature that allows you to modify or extend the behavior of functions or classes without permanently modifying their code."),
        HumanMessage(content="Can you show me a simple example?"),
        AIMessage(content="Here's a basic decorator example:\n\n```python\ndef my_decorator(func):\n    def wrapper():\n        print('Before function call')\n        func()\n        print('After function call')\n    return wrapper\n```")
    ]
    
    for i, message in enumerate(langchain_messages, 1):
        tokens = counter.count_message_tokens(message)
        msg_type = type(message).__name__.replace("Message", "")
        content_preview = str(message.content)[:50] + "..." if len(str(message.content)) > 50 else str(message.content)
        print(f"   {i}. {msg_type}: {tokens} tokens")
        print(f"      Content: '{content_preview}'")
    
    total_usage = counter.count_messages_tokens(langchain_messages)
    print(f"   Total conversation: {total_usage.total_tokens} tokens")
    
    # Example 2: Dictionary message format
    print(f"\n2. Dictionary Message Format:")
    dict_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data, but I can help you find weather information through various weather services or apps."},
        {"role": "user", "content": "How can I check the weather programmatically?"}
    ]
    
    for i, message in enumerate(dict_messages, 1):
        tokens = counter.count_message_tokens(message)
        content_preview = message['content'][:50] + "..." if len(message['content']) > 50 else message['content']
        print(f"   {i}. {message['role']}: {tokens} tokens")
        print(f"      Content: '{content_preview}'")
    
    dict_usage = counter.count_messages_tokens(dict_messages)
    print(f"   Total conversation: {dict_usage.total_tokens} tokens")


def cost_estimation_examples():
    """Demonstrate cost estimation functionality."""
    print("\n💰 Cost Estimation Examples")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Example 1: Different usage scenarios
    print("1. Cost Estimation for Different Scenarios:")
    scenarios = [
        ("Quick question", 20, 10),
        ("Short conversation", 150, 75),
        ("Medium article", 800, 400),
        ("Long document", 3000, 1500),
        ("Research paper", 8000, 2000),
        ("Book chapter", 15000, 5000)
    ]
    
    for scenario_name, prompt_tokens, completion_tokens in scenarios:
        cost = counter.estimate_cost(prompt_tokens, completion_tokens)
        print(f"   {scenario_name:15}: {prompt_tokens:5} + {completion_tokens:4} tokens = ${cost.total_cost:.4f}")
    
    # Example 2: Model comparison
    print(f"\n2. Cost Comparison Across Models (1000 prompt + 500 completion tokens):")
    comparison = counter.compare_models(1000, 500)
    
    # Sort by cost
    sorted_models = sorted(comparison.items(), key=lambda x: x[1].total_cost)
    
    print(f"   {'Model':<25} {'Prompt Cost':<12} {'Completion Cost':<15} {'Total Cost':<12}")
    print(f"   {'-'*25} {'-'*12} {'-'*15} {'-'*12}")
    
    for model_name, cost in sorted_models:
        print(f"   {model_name:<25} ${cost.prompt_cost:<11.4f} ${cost.completion_cost:<14.4f} ${cost.total_cost:<11.4f}")
    
    # Example 3: Conversation cost calculation
    print(f"\n3. Real Conversation Cost Calculation:")
    conversation = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="I need help debugging a Python function."),
        AIMessage(content="I'd be happy to help you debug your Python function! Please share the code and describe the issue you're experiencing."),
        HumanMessage(content="Here's my function: def calculate_average(numbers): return sum(numbers) / len(numbers). It crashes sometimes."),
        AIMessage(content="The issue is likely a division by zero error when the list is empty. Here's a safer version: def calculate_average(numbers): return sum(numbers) / len(numbers) if numbers else 0")
    ]
    
    estimated_response_tokens = 100
    usage, cost = counter.calculate_conversation_cost(conversation, estimated_response_tokens)
    
    print(f"   Messages in conversation: {len(conversation)}")
    print(f"   Prompt tokens: {usage.prompt_tokens}")
    print(f"   Estimated response tokens: {usage.completion_tokens}")
    print(f"   Total tokens: {usage.total_tokens}")
    print(f"   Estimated total cost: ${cost.total_cost:.4f}")


def text_analysis_examples():
    """Demonstrate text analysis functionality."""
    print("\n📊 Text Analysis Examples")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    
    # Analyze different types of content
    content_samples = [
        ("Simple sentence", "The quick brown fox jumps over the lazy dog."),
        ("Technical documentation", "The REST API endpoint accepts JSON payloads with authentication headers and returns structured data responses with appropriate HTTP status codes."),
        ("Creative writing", "In the misty mountains of ancient lore, where dragons once soared through crystalline skies, a young adventurer discovered a hidden treasure that would change the course of history forever."),
        ("Code documentation", "This function implements a binary search algorithm with O(log n) time complexity. It takes a sorted array and target value as parameters, returning the index of the target or -1 if not found."),
        ("Conversational text", "Hey there! How's it going? I was wondering if you could help me out with something. It's not super urgent, but I'd really appreciate your input when you get a chance.")
    ]
    
    print("Detailed text analysis:")
    for content_type, text in content_samples:
        analysis = counter.analyze_text(text)
        
        print(f"\n{content_type}:")
        print(f"   Text: '{text}'")
        print(f"   Tokens: {analysis['token_count']}")
        print(f"   Characters: {analysis['character_count']}")
        print(f"   Words: {analysis['word_count']}")
        print(f"   Tokens per word: {analysis['tokens_per_word']:.2f}")
        print(f"   Characters per token: {analysis['characters_per_token']:.2f}")
        print(f"   Estimated cost: ${analysis['estimated_cost'].total_cost:.6f}")


def token_optimization_examples():
    """Demonstrate token optimization functionality."""
    print("\n⚡ Token Optimization Examples")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Example 1: Text compression
    print("1. Text Compression:")
    redundant_text = "This is a very very really quite pretty amazing text with um lots of like unnecessary filler words you know. It has... excessive punctuation!!! And   extra   spaces   everywhere."
    
    compression_result = optimizer.compress_text(redundant_text)
    
    print(f"   Original text: '{redundant_text}'")
    print(f"   Compressed text: '{compression_result.optimized_text}'")
    print(f"   Original tokens: {compression_result.original_tokens}")
    print(f"   Optimized tokens: {compression_result.optimized_tokens}")
    print(f"   Tokens saved: {compression_result.tokens_saved}")
    print(f"   Compression ratio: {compression_result.compression_ratio:.2f}")
    print(f"   Savings percentage: {compression_result.savings_percentage:.1f}%")
    
    # Example 2: Text truncation strategies
    print(f"\n2. Text Truncation Strategies:")
    long_text = "This is a comprehensive guide to understanding machine learning algorithms and their applications in modern data science. The field encompasses various techniques including supervised learning, unsupervised learning, and reinforcement learning. Each approach has its own strengths and use cases in solving real-world problems."
    
    strategies = [
        (OptimizationStrategy.TRUNCATE_END, "Truncate from end"),
        (OptimizationStrategy.TRUNCATE_START, "Truncate from start"),
        (OptimizationStrategy.TRUNCATE_MIDDLE, "Truncate from middle")
    ]
    
    max_tokens = 25
    print(f"   Original text ({counter.count_tokens(long_text)} tokens): '{long_text}'")
    print(f"   Truncating to {max_tokens} tokens:")
    
    for strategy, description in strategies:
        result = optimizer.truncate_text(long_text, max_tokens, strategy=strategy)
        print(f"   {description}:")
        print(f"     Result: '{result.optimized_text}'")
        print(f"     Tokens: {result.optimized_tokens}, Saved: {result.tokens_saved}")
    
    # Example 3: Summarization
    print(f"\n3. Text Summarization:")
    article_text = "Artificial intelligence has revolutionized numerous industries in recent years. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural networks with multiple layers to solve complex problems. Natural language processing enables computers to understand and generate human language. Computer vision allows machines to interpret and analyze visual information. These technologies are being applied in healthcare, finance, transportation, and many other sectors."
    
    summarization_result = optimizer.summarize_context(article_text, target_tokens=30)
    
    print(f"   Original text ({summarization_result.original_tokens} tokens):")
    print(f"   '{article_text}'")
    print(f"   Summarized text ({summarization_result.optimized_tokens} tokens):")
    print(f"   '{summarization_result.optimized_text}'")
    print(f"   Compression ratio: {summarization_result.compression_ratio:.2f}")


def message_optimization_examples():
    """Demonstrate message optimization functionality."""
    print("\n📝 Message Optimization Examples")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Create a long conversation that needs optimization
    long_conversation = [
        SystemMessage(content="You are an expert software engineer with extensive experience in Python, JavaScript, and system design. You provide detailed, accurate, and helpful responses to programming questions."),
        HumanMessage(content="I'm working on a web application and need help with database optimization. The application is getting slow with large datasets."),
        AIMessage(content="Database optimization is crucial for application performance. There are several strategies you can employ: indexing, query optimization, connection pooling, caching, and database normalization. Let me explain each approach in detail."),
        HumanMessage(content="Can you explain indexing in more detail? I'm not sure which columns to index."),
        AIMessage(content="Indexing is like creating a table of contents for your database. You should index columns that are frequently used in WHERE clauses, JOIN conditions, and ORDER BY statements. However, be careful not to over-index as it can slow down INSERT and UPDATE operations."),
        HumanMessage(content="What about query optimization techniques?"),
        AIMessage(content="Query optimization involves writing efficient SQL queries. Use EXPLAIN to analyze query execution plans, avoid SELECT *, use appropriate JOINs instead of subqueries when possible, and consider query caching for frequently executed queries."),
        HumanMessage(content="How do I implement caching effectively?")
    ]
    
    original_usage = counter.count_messages_tokens(long_conversation)
    print(f"Original conversation: {len(long_conversation)} messages, {original_usage.total_tokens} tokens")
    
    # Show original conversation structure
    print(f"\nOriginal conversation structure:")
    for i, msg in enumerate(long_conversation, 1):
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = "Unknown"
        
        content_preview = str(msg.content)[:60] + "..." if len(str(msg.content)) > 60 else str(msg.content)
        tokens = counter.count_message_tokens(msg)
        print(f"   {i}. {role} ({tokens} tokens): '{content_preview}'")
    
    # Optimize for different token limits
    token_limits = [200, 100, 50]
    
    for limit in token_limits:
        print(f"\nOptimizing for {limit} token limit:")
        optimized_messages, result = optimizer.optimize_messages(
            long_conversation,
            max_tokens=limit,
            preserve_system_message=True,
            preserve_last_messages=2
        )
        
        print(f"   Result: {len(long_conversation)} -> {len(optimized_messages)} messages")
        print(f"   Tokens: {result.original_tokens} -> {result.optimized_tokens}")
        print(f"   Saved: {result.tokens_saved} tokens ({result.savings_percentage:.1f}%)")
        
        print(f"   Optimized conversation:")
        for i, msg in enumerate(optimized_messages, 1):
            if isinstance(msg, SystemMessage):
                role = "System"
            elif isinstance(msg, HumanMessage):
                role = "Human"
            elif isinstance(msg, AIMessage):
                role = "AI"
            else:
                role = "Unknown"
            
            content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
            tokens = counter.count_message_tokens(msg)
            print(f"     {i}. {role} ({tokens} tokens): '{content_preview}'")


def cost_optimization_examples():
    """Demonstrate cost-based optimization."""
    print("\n💸 Cost-Based Optimization Examples")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Example: Optimize content for different cost budgets
    expensive_content = """
    This comprehensive research document explores the intricate relationships between artificial intelligence, machine learning, and deep learning technologies in modern computational systems. The analysis delves into various algorithmic approaches, including supervised learning methodologies such as linear regression, logistic regression, decision trees, random forests, and support vector machines. Additionally, it examines unsupervised learning techniques like clustering algorithms, dimensionality reduction methods, and association rule mining. The document also covers reinforcement learning paradigms, neural network architectures, convolutional neural networks for computer vision applications, recurrent neural networks for sequential data processing, and transformer models for natural language understanding tasks.
    """
    
    original_tokens = counter.count_tokens(expensive_content)
    original_cost = counter.estimate_cost(original_tokens)
    
    print(f"Original content:")
    print(f"   Tokens: {original_tokens}")
    print(f"   Estimated cost: ${original_cost.total_cost:.4f}")
    print(f"   Content preview: '{expensive_content[:100]}...'")
    
    # Optimize for different cost budgets
    cost_budgets = [0.02, 0.01, 0.005, 0.001]
    
    print(f"\nOptimizing for different cost budgets:")
    for budget in cost_budgets:
        result = optimizer.optimize_for_cost(expensive_content, budget)
        optimized_cost = counter.estimate_cost(result.optimized_tokens)
        
        print(f"\n   Budget: ${budget:.3f}")
        print(f"   Tokens: {result.original_tokens} -> {result.optimized_tokens}")
        print(f"   Cost: ${original_cost.total_cost:.4f} -> ${optimized_cost.total_cost:.4f}")
        print(f"   Savings: {result.savings_percentage:.1f}%")
        print(f"   Optimized content: '{result.optimized_text[:80]}...'")


def batch_processing_examples():
    """Demonstrate batch processing functionality."""
    print("\n🔄 Batch Processing Examples")
    print("=" * 50)
    
    counter = TokenCounter(model=ModelType.GPT_4)
    optimizer = TokenOptimizer(token_counter=counter)
    
    # Create multiple documents for batch processing
    documents = [
        "Product documentation for our new API service including authentication, endpoints, and response formats.",
        "User manual with very very detailed instructions and um lots of redundant explanations that could be simplified.",
        "Technical specification document covering system architecture, database design, and integration patterns.",
        "Marketing copy with engaging content about our innovative solutions and cutting-edge technology offerings.",
        "Support documentation including troubleshooting guides, FAQ sections, and step-by-step tutorials for users."
    ]
    
    print(f"Batch processing {len(documents)} documents:")
    
    # Show original token counts
    original_counts = counter.batch_count_tokens(documents)
    total_original = sum(original_counts)
    
    print(f"\nOriginal token counts:")
    for i, (doc, count) in enumerate(zip(documents, original_counts), 1):
        preview = doc[:50] + "..." if len(doc) > 50 else doc
        print(f"   {i}. {count:3d} tokens: '{preview}'")
    print(f"   Total: {total_original} tokens")
    
    # Batch optimize with different strategies
    strategies = [
        (OptimizationStrategy.COMPRESS, "Compression"),
        (OptimizationStrategy.TRUNCATE_END, "Truncation"),
        (OptimizationStrategy.SUMMARIZE, "Summarization")
    ]
    
    max_tokens_per_doc = 20
    
    for strategy, strategy_name in strategies:
        print(f"\n{strategy_name} (max {max_tokens_per_doc} tokens per document):")
        results = optimizer.batch_optimize(documents, max_tokens_per_doc, strategy=strategy)
        
        total_optimized = sum(r.optimized_tokens for r in results)
        total_saved = sum(r.tokens_saved for r in results)
        
        print(f"   Total tokens: {total_original} -> {total_optimized}")
        print(f"   Tokens saved: {total_saved} ({(total_saved/total_original)*100:.1f}%)")
        
        for i, result in enumerate(results, 1):
            print(f"   Doc {i}: {result.original_tokens} -> {result.optimized_tokens} tokens")
    
    # Get comprehensive statistics
    compress_results = optimizer.batch_optimize(documents, max_tokens_per_doc, OptimizationStrategy.COMPRESS)
    stats = optimizer.get_optimization_stats(compress_results)
    
    print(f"\nBatch Optimization Statistics:")
    print(f"   Documents processed: {stats['total_texts']}")
    print(f"   Total original tokens: {stats['total_original_tokens']}")
    print(f"   Total optimized tokens: {stats['total_optimized_tokens']}")
    print(f"   Total tokens saved: {stats['total_saved_tokens']}")
    print(f"   Average compression ratio: {stats['average_compression_ratio']:.2f}")
    print(f"   Average savings percentage: {stats['average_savings_percentage']:.1f}%")


def custom_pricing_examples():
    """Demonstrate custom pricing functionality."""
    print("\n🏷️ Custom Pricing Examples")
    print("=" * 50)
    
    # Create counter with custom pricing
    custom_pricing = {
        "gpt-4": ModelPricing("gpt-4", 0.02, 0.04),  # Custom pricing
        "custom-model": ModelPricing("custom-model", 0.001, 0.002)
    }
    
    counter = TokenCounter(model=ModelType.GPT_4, custom_pricing=custom_pricing)
    
    print("Custom pricing configuration:")
    print(f"   GPT-4: $0.02/1K prompt, $0.04/1K completion")
    print(f"   Custom model: $0.001/1K prompt, $0.002/1K completion")
    
    # Test cost calculation with custom pricing
    test_usage = (1000, 500)  # 1000 prompt, 500 completion tokens
    
    cost = counter.estimate_cost(test_usage[0], test_usage[1])
    print(f"\nCost calculation for {test_usage[0]} prompt + {test_usage[1]} completion tokens:")
    print(f"   Prompt cost: ${cost.prompt_cost:.4f}")
    print(f"   Completion cost: ${cost.completion_cost:.4f}")
    print(f"   Total cost: ${cost.total_cost:.4f}")
    
    # Update pricing dynamically
    new_pricing = ModelPricing("gpt-4", 0.015, 0.03)
    counter.update_pricing(ModelType.GPT_4, new_pricing)
    
    updated_cost = counter.estimate_cost(test_usage[0], test_usage[1])
    print(f"\nAfter pricing update:")
    print(f"   New total cost: ${updated_cost.total_cost:.4f}")
    print(f"   Cost difference: ${updated_cost.total_cost - cost.total_cost:.4f}")


def model_information_examples():
    """Demonstrate model information functionality."""
    print("\n📋 Model Information Examples")
    print("=" * 50)
    
    counter = TokenCounter()
    
    # Get supported models
    supported_models = counter.get_supported_models()
    print(f"Supported models ({len(supported_models)}):")
    
    # Show detailed information for key models
    key_models = [ModelType.GPT_4, ModelType.GPT_3_5_TURBO, ModelType.CLAUDE_3_SONNET, ModelType.DEEPSEEK_CHAT]
    
    print(f"\nDetailed model information:")
    print(f"{'Model':<20} {'Prompt $/1K':<12} {'Completion $/1K':<15} {'Tokenizer':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*15} {'-'*10}")
    
    for model in key_models:
        info = counter.get_model_info(model)
        prompt_price = f"${info['prompt_price_per_1k']:.4f}" if info['has_pricing'] else "N/A"
        completion_price = f"${info['completion_price_per_1k']:.4f}" if info['has_pricing'] else "N/A"
        tokenizer_status = "Yes" if info['tokenizer_available'] else "No"
        
        print(f"{model.value:<20} {prompt_price:<12} {completion_price:<15} {tokenizer_status:<10}")
    
    # Show all supported models
    print(f"\nAll supported models:")
    for i, model_name in enumerate(supported_models, 1):
        print(f"   {i:2d}. {model_name}")


def run_all_examples():
    """Run all token management examples."""
    print("🎯 AI Toolkit Token Management Examples")
    print("=" * 60)
    
    examples = [
        basic_token_counting_examples,
        message_token_counting_examples,
        cost_estimation_examples,
        text_analysis_examples,
        token_optimization_examples,
        message_optimization_examples,
        cost_optimization_examples,
        batch_processing_examples,
        custom_pricing_examples,
        model_information_examples,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            if i > 1:
                print(f"\n{'='*20} Example {i} {'='*20}")
            example_func()
        except KeyboardInterrupt:
            print("\n\n⏹️ Examples interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in example: {e}")
    
    print(f"\n🎉 Token Management Examples Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Token counting for text and messages")
    print("   ✅ Cost estimation and model comparison")
    print("   ✅ Text analysis and statistics")
    print("   ✅ Token optimization strategies")
    print("   ✅ Message optimization and truncation")
    print("   ✅ Cost-based optimization")
    print("   ✅ Batch processing capabilities")
    print("   ✅ Custom pricing configuration")
    print("   ✅ Model information and pricing")


if __name__ == "__main__":
    run_all_examples()