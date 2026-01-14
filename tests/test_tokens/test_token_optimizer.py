"""
Tests for TokenOptimizer class.
"""

import pytest
from unittest.mock import Mock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ai_toolkit.tokens.token_optimizer import (
    TokenOptimizer, OptimizationResult, OptimizationStrategy
)
from ai_toolkit.tokens.token_counter import TokenCounter, ModelType


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test creating OptimizationResult."""
        result = OptimizationResult(
            original_text="Original text here",
            optimized_text="Optimized text",
            original_tokens=100,
            optimized_tokens=80,
            tokens_saved=20,
            compression_ratio=0.8,
            strategy_used="compress"
        )
        
        assert result.original_text == "Original text here"
        assert result.optimized_text == "Optimized text"
        assert result.original_tokens == 100
        assert result.optimized_tokens == 80
        assert result.tokens_saved == 20
        assert result.compression_ratio == 0.8
        assert result.strategy_used == "compress"
    
    def test_savings_percentage(self):
        """Test savings percentage calculation."""
        result = OptimizationResult(
            original_text="test",
            optimized_text="test",
            original_tokens=100,
            optimized_tokens=75,
            tokens_saved=25,
            compression_ratio=0.75,
            strategy_used="test"
        )
        
        assert result.savings_percentage == 25.0
        
        # Test with zero original tokens
        result_zero = OptimizationResult(
            original_text="",
            optimized_text="",
            original_tokens=0,
            optimized_tokens=0,
            tokens_saved=0,
            compression_ratio=1.0,
            strategy_used="test"
        )
        
        assert result_zero.savings_percentage == 0.0


class TestTokenOptimizer:
    """Test TokenOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.token_counter = TokenCounter(model=ModelType.GPT_4)
        self.optimizer = TokenOptimizer(token_counter=self.token_counter)
    
    def test_initialization(self):
        """Test TokenOptimizer initialization."""
        optimizer = TokenOptimizer()
        assert optimizer.token_counter is not None
        
        # Test with custom token counter
        custom_counter = TokenCounter(model=ModelType.GPT_3_5_TURBO)
        optimizer_custom = TokenOptimizer(token_counter=custom_counter)
        assert optimizer_custom.token_counter == custom_counter
    
    def test_truncate_text_no_truncation_needed(self):
        """Test truncation when text is already within limit."""
        text = "Short text"
        max_tokens = 100
        
        result = self.optimizer.truncate_text(text, max_tokens)
        
        assert result.original_text == text
        assert result.optimized_text == text
        assert result.tokens_saved == 0
        assert result.compression_ratio == 1.0
    
    def test_truncate_text_end_strategy(self):
        """Test truncation with end strategy."""
        text = "This is a long text that needs to be truncated from the end to fit within the token limit."
        max_tokens = 10
        
        result = self.optimizer.truncate_text(
            text, 
            max_tokens, 
            strategy=OptimizationStrategy.TRUNCATE_END
        )
        
        assert result.optimized_tokens <= max_tokens
        assert result.tokens_saved > 0
        assert result.compression_ratio < 1.0
        assert result.strategy_used == OptimizationStrategy.TRUNCATE_END.value
    
    def test_truncate_text_start_strategy(self):
        """Test truncation with start strategy."""
        text = "This is a long text that needs to be truncated from the start to fit within the token limit."
        max_tokens = 10
        
        result = self.optimizer.truncate_text(
            text, 
            max_tokens, 
            strategy=OptimizationStrategy.TRUNCATE_START
        )
        
        assert result.optimized_tokens <= max_tokens
        assert result.tokens_saved > 0
        assert result.strategy_used == OptimizationStrategy.TRUNCATE_START.value
    
    def test_truncate_text_middle_strategy(self):
        """Test truncation with middle strategy."""
        text = "This is a long text that needs to be truncated from the middle to fit within the token limit."
        max_tokens = 15
        
        result = self.optimizer.truncate_text(
            text, 
            max_tokens, 
            strategy=OptimizationStrategy.TRUNCATE_MIDDLE
        )
        
        assert result.optimized_tokens <= max_tokens
        assert result.tokens_saved > 0
        assert result.strategy_used == OptimizationStrategy.TRUNCATE_MIDDLE.value
    
    def test_truncate_text_preserve_sentences(self):
        """Test truncation with sentence preservation."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        max_tokens = 8
        
        result = self.optimizer.truncate_text(
            text, 
            max_tokens, 
            preserve_sentences=True
        )
        
        # Should preserve complete sentences
        assert result.optimized_tokens <= max_tokens
        assert not result.optimized_text.endswith(".")  # May not end with period if truncated
    
    def test_compress_text(self):
        """Test text compression."""
        text = "This is a very very really quite pretty text with um extra   spaces and... redundant punctuation!!!"
        
        result = self.optimizer.compress_text(text)
        
        assert result.optimized_tokens <= result.original_tokens
        assert result.strategy_used == OptimizationStrategy.COMPRESS.value
        
        # Check that redundant elements are removed
        assert "very very" not in result.optimized_text
        assert "really" not in result.optimized_text
        assert "um" not in result.optimized_text
        assert "   " not in result.optimized_text  # Multiple spaces
        assert "..." not in result.optimized_text
        assert "!!!" not in result.optimized_text
    
    def test_compress_text_empty(self):
        """Test compressing empty text."""
        result = self.optimizer.compress_text("")
        
        assert result.original_tokens == 0
        assert result.optimized_tokens == 0
        assert result.tokens_saved == 0
        assert result.compression_ratio == 1.0
    
    def test_optimize_messages_no_optimization_needed(self):
        """Test message optimization when no optimization is needed."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        max_tokens = 1000
        
        optimized_messages, result = self.optimizer.optimize_messages(messages, max_tokens)
        
        assert len(optimized_messages) == len(messages)
        assert result.tokens_saved == 0
        assert result.compression_ratio == 1.0
    
    def test_optimize_messages_empty_list(self):
        """Test optimizing empty message list."""
        messages = []
        max_tokens = 100
        
        optimized_messages, result = self.optimizer.optimize_messages(messages, max_tokens)
        
        assert optimized_messages == []
        assert result.original_tokens == 0
        assert result.optimized_tokens == 0
    
    def test_optimize_messages_with_system_message(self):
        """Test message optimization preserving system message."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Tell me a very long story about adventures"),
            AIMessage(content="Once upon a time in a land far away..."),
            HumanMessage(content="Continue the story")
        ]
        max_tokens = 20
        
        optimized_messages, result = self.optimizer.optimize_messages(
            messages, 
            max_tokens, 
            preserve_system_message=True
        )
        
        # System message should be preserved
        assert any(isinstance(msg, SystemMessage) for msg in optimized_messages)
        assert len(optimized_messages) <= len(messages)
        assert result.tokens_saved > 0
    
    def test_optimize_messages_preserve_last_messages(self):
        """Test message optimization preserving last messages."""
        messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
            HumanMessage(content="Third message"),
            HumanMessage(content="Fourth message"),
            HumanMessage(content="Fifth message")
        ]
        max_tokens = 15
        
        optimized_messages, result = self.optimizer.optimize_messages(
            messages, 
            max_tokens, 
            preserve_last_messages=2
        )
        
        # Last 2 messages should be preserved
        assert len(optimized_messages) >= 2
        # Check that the last messages are preserved
        original_last_two = messages[-2:]
        optimized_last_two = optimized_messages[-2:]
        
        for orig, opt in zip(original_last_two, optimized_last_two):
            assert orig.content == opt.content
    
    def test_optimize_messages_dict_format(self):
        """Test optimizing messages in dictionary format."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"}
        ]
        max_tokens = 15
        
        optimized_messages, result = self.optimizer.optimize_messages(messages, max_tokens)
        
        assert len(optimized_messages) <= len(messages)
        assert all(isinstance(msg, dict) for msg in optimized_messages)
    
    def test_summarize_context_no_summarization_needed(self):
        """Test summarization when text is already within limit."""
        text = "Short text that doesn't need summarization."
        target_tokens = 100
        
        result = self.optimizer.summarize_context(text, target_tokens)
        
        assert result.original_text == text
        assert result.optimized_text == text
        assert result.tokens_saved == 0
        assert result.compression_ratio == 1.0
    
    def test_summarize_context_simple_summarization(self):
        """Test simple extractive summarization."""
        text = "First sentence is important. Second sentence is less important. Third sentence is also important. Fourth sentence is filler."
        target_tokens = 10
        
        result = self.optimizer.summarize_context(text, target_tokens)
        
        assert result.optimized_tokens <= target_tokens
        assert result.tokens_saved > 0
        assert result.strategy_used == OptimizationStrategy.SUMMARIZE.value
    
    def test_summarize_context_custom_summarizer(self):
        """Test summarization with custom summarizer function."""
        def custom_summarizer(text):
            return "Custom summary"
        
        text = "Long text that needs custom summarization."
        target_tokens = 5
        
        result = self.optimizer.summarize_context(
            text, 
            target_tokens, 
            summarizer=custom_summarizer
        )
        
        assert result.optimized_text == "Custom summary"
    
    def test_optimize_for_cost(self):
        """Test cost-based optimization."""
        text = "This is a text that needs to be optimized for cost considerations."
        max_cost = 0.001  # Very low cost limit
        
        result = self.optimizer.optimize_for_cost(text, max_cost)
        
        # Should optimize the text to reduce cost
        assert result.optimized_tokens <= result.original_tokens
        assert result.strategy_used in ["cost_optimization", "compress"]
    
    def test_optimize_for_cost_no_optimization_needed(self):
        """Test cost optimization when no optimization is needed."""
        text = "Short text"
        max_cost = 1.0  # High cost limit
        
        result = self.optimizer.optimize_for_cost(text, max_cost)
        
        assert result.original_text == text
        assert result.optimized_text == text
        assert result.tokens_saved == 0
    
    def test_batch_optimize(self):
        """Test batch optimization."""
        texts = [
            "First text that needs optimization",
            "Second text with very very redundant words",
            "Third text for batch processing"
        ]
        max_tokens_per_text = 5
        
        results = self.optimizer.batch_optimize(
            texts, 
            max_tokens_per_text, 
            strategy=OptimizationStrategy.COMPRESS
        )
        
        assert len(results) == len(texts)
        assert all(isinstance(result, OptimizationResult) for result in results)
        assert all(result.optimized_tokens <= max_tokens_per_text for result in results)
    
    def test_batch_optimize_different_strategies(self):
        """Test batch optimization with different strategies."""
        texts = ["Text to optimize"] * 3
        max_tokens = 5
        
        strategies = [
            OptimizationStrategy.TRUNCATE_END,
            OptimizationStrategy.TRUNCATE_START,
            OptimizationStrategy.SUMMARIZE
        ]
        
        for strategy in strategies:
            results = self.optimizer.batch_optimize(texts, max_tokens, strategy=strategy)
            assert len(results) == len(texts)
            assert all(result.strategy_used == strategy.value or 
                      result.strategy_used == "truncate_end" for result in results)
    
    def test_get_optimization_stats(self):
        """Test getting optimization statistics."""
        results = [
            OptimizationResult("text1", "opt1", 100, 80, 20, 0.8, "compress"),
            OptimizationResult("text2", "opt2", 200, 150, 50, 0.75, "truncate"),
            OptimizationResult("text3", "opt3", 150, 120, 30, 0.8, "compress")
        ]
        
        stats = self.optimizer.get_optimization_stats(results)
        
        assert stats['total_texts'] == 3
        assert stats['total_original_tokens'] == 450
        assert stats['total_optimized_tokens'] == 350
        assert stats['total_saved_tokens'] == 100
        assert stats['average_compression_ratio'] == (0.8 + 0.75 + 0.8) / 3
        assert stats['average_savings_percentage'] == (20 + 25 + 20) / 3
        assert 'compress' in stats['strategies_used']
        assert 'truncate' in stats['strategies_used']
    
    def test_get_optimization_stats_empty(self):
        """Test getting stats from empty results."""
        stats = self.optimizer.get_optimization_stats([])
        assert stats == {}
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = self.optimizer._split_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence" in sentences
        assert "Second sentence" in sentences
        assert "Third sentence" in sentences
        assert "Fourth sentence" in sentences
    
    def test_is_system_message(self):
        """Test system message detection."""
        system_msg = SystemMessage(content="System message")
        human_msg = HumanMessage(content="Human message")
        dict_system = {"role": "system", "content": "System message"}
        dict_user = {"role": "user", "content": "User message"}
        
        assert self.optimizer._is_system_message(system_msg) is True
        assert self.optimizer._is_system_message(human_msg) is False
        assert self.optimizer._is_system_message(dict_system) is True
        assert self.optimizer._is_system_message(dict_user) is False
    
    def test_compress_message_langchain_message(self):
        """Test compressing LangChain messages."""
        message = HumanMessage(content="This is a very very redundant message with um filler words.")
        max_tokens = 5
        
        compressed = self.optimizer._compress_message(message, max_tokens)
        
        assert isinstance(compressed, HumanMessage)
        assert len(compressed.content) < len(message.content)
    
    def test_compress_message_dict_format(self):
        """Test compressing dictionary format messages."""
        message = {"role": "user", "content": "This is a very very redundant message."}
        max_tokens = 5
        
        compressed = self.optimizer._compress_message(message, max_tokens)
        
        assert isinstance(compressed, dict)
        assert compressed["role"] == "user"
        assert len(compressed["content"]) <= len(message["content"])
    
    def test_simple_summarize_single_sentence(self):
        """Test simple summarization with single sentence."""
        text = "This is a single sentence."
        target_tokens = 3
        
        result = self.optimizer._simple_summarize(text, target_tokens)
        
        # Should fallback to truncation for single sentence
        assert len(result) <= len(text)
    
    def test_optimization_strategy_enum(self):
        """Test OptimizationStrategy enum values."""
        assert OptimizationStrategy.TRUNCATE_START.value == "truncate_start"
        assert OptimizationStrategy.TRUNCATE_END.value == "truncate_end"
        assert OptimizationStrategy.TRUNCATE_MIDDLE.value == "truncate_middle"
        assert OptimizationStrategy.SUMMARIZE.value == "summarize"
        assert OptimizationStrategy.COMPRESS.value == "compress"
        assert OptimizationStrategy.REMOVE_REDUNDANCY.value == "remove_redundancy"
    
    def test_truncate_text_very_short_limit(self):
        """Test truncation with very short token limit."""
        text = "This is a longer text that needs significant truncation."
        max_tokens = 1
        
        result = self.optimizer.truncate_text(text, max_tokens)
        
        assert result.optimized_tokens <= max_tokens
        assert result.tokens_saved > 0
    
    def test_compress_text_already_clean(self):
        """Test compressing text that's already clean."""
        text = "Clean text without redundancy."
        
        result = self.optimizer.compress_text(text)
        
        # Should have minimal or no compression
        assert result.optimized_tokens <= result.original_tokens
        assert result.compression_ratio >= 0.8  # Should be close to original
    
    def test_optimize_messages_mixed_types(self):
        """Test optimizing messages with mixed types."""
        messages = [
            SystemMessage(content="System message"),
            {"role": "user", "content": "User message"},
            HumanMessage(content="Human message"),
            {"role": "assistant", "content": "Assistant message"}
        ]
        max_tokens = 20
        
        optimized_messages, result = self.optimizer.optimize_messages(messages, max_tokens)
        
        assert len(optimized_messages) <= len(messages)
        # Should preserve message types
        for msg in optimized_messages:
            assert isinstance(msg, (SystemMessage, HumanMessage, AIMessage, dict))