"""
Tests for TokenCounter class.
"""

import pytest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ai_toolkit.tokens.token_counter import (
    TokenCounter, TokenUsage, CostEstimate, ModelPricing, ModelType
)


class TestTokenUsage:
    """Test TokenUsage dataclass."""
    
    def test_token_usage_creation(self):
        """Test creating TokenUsage."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.input_tokens == 100  # Alias
        assert usage.output_tokens == 50  # Alias
    
    def test_token_usage_addition(self):
        """Test adding TokenUsage objects."""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=75, total_tokens=275)
        
        combined = usage1 + usage2
        
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 125
        assert combined.total_tokens == 425


class TestCostEstimate:
    """Test CostEstimate dataclass."""
    
    def test_cost_estimate_creation(self):
        """Test creating CostEstimate."""
        cost = CostEstimate(
            prompt_cost=0.05,
            completion_cost=0.10,
            total_cost=0.15,
            currency="USD",
            model="gpt-4"
        )
        
        assert cost.prompt_cost == 0.05
        assert cost.completion_cost == 0.10
        assert cost.total_cost == 0.15
        assert cost.currency == "USD"
        assert cost.model == "gpt-4"
    
    def test_cost_estimate_addition(self):
        """Test adding CostEstimate objects."""
        cost1 = CostEstimate(prompt_cost=0.05, completion_cost=0.10, total_cost=0.15, model="gpt-4")
        cost2 = CostEstimate(prompt_cost=0.03, completion_cost=0.06, total_cost=0.09, model="gpt-4")
        
        combined = cost1 + cost2
        
        assert combined.prompt_cost == 0.08
        assert combined.completion_cost == 0.16
        assert combined.total_cost == 0.24
        assert combined.model == "gpt-4"


class TestModelPricing:
    """Test ModelPricing dataclass."""
    
    def test_model_pricing_creation(self):
        """Test creating ModelPricing."""
        pricing = ModelPricing(
            model="gpt-4",
            prompt_price_per_1k=0.03,
            completion_price_per_1k=0.06
        )
        
        assert pricing.model == "gpt-4"
        assert pricing.prompt_price_per_1k == 0.03
        assert pricing.completion_price_per_1k == 0.06
        assert pricing.currency == "USD"
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        pricing = ModelPricing("gpt-4", 0.03, 0.06)
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        
        cost = pricing.calculate_cost(usage)
        
        assert cost.prompt_cost == 0.03  # 1000/1000 * 0.03
        assert cost.completion_cost == 0.03  # 500/1000 * 0.06
        assert cost.total_cost == 0.06
        assert cost.model == "gpt-4"


class TestTokenCounter:
    """Test TokenCounter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter(model=ModelType.GPT_4)
    
    def test_initialization(self):
        """Test TokenCounter initialization."""
        counter = TokenCounter(model=ModelType.GPT_3_5_TURBO)
        assert counter.model == ModelType.GPT_3_5_TURBO
        assert ModelType.GPT_3_5_TURBO in counter.pricing
    
    def test_initialization_with_string_model(self):
        """Test initialization with string model name."""
        counter = TokenCounter(model="gpt-4")
        assert counter.model == ModelType.GPT_4
    
    def test_count_tokens_basic(self):
        """Test basic token counting."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_count_tokens_empty(self):
        """Test token counting with empty text."""
        assert self.counter.count_tokens("") == 0
        assert self.counter.count_tokens(None) == 0
    
    @patch('tiktoken.get_encoding')
    def test_count_tokens_with_tokenizer_error(self, mock_get_encoding):
        """Test token counting when tokenizer fails."""
        mock_get_encoding.side_effect = Exception("Tokenizer error")
        
        counter = TokenCounter()
        token_count = counter.count_tokens("Hello world")
        
        # Should fallback to character-based counting
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_count_message_tokens_langchain_message(self):
        """Test counting tokens in LangChain messages."""
        message = HumanMessage(content="Hello, how are you?")
        token_count = self.counter.count_message_tokens(message)
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_count_message_tokens_dict_message(self):
        """Test counting tokens in dictionary messages."""
        message = {"role": "user", "content": "Hello, how are you?"}
        token_count = self.counter.count_message_tokens(message)
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_count_messages_tokens(self):
        """Test counting tokens in multiple messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?")
        ]
        
        usage = self.counter.count_messages_tokens(messages)
        
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == usage.prompt_tokens
    
    def test_count_messages_tokens_dict_format(self):
        """Test counting tokens in dictionary format messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        usage = self.counter.count_messages_tokens(messages)
        
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens > 0
    
    def test_estimate_completion_tokens(self):
        """Test estimating completion tokens."""
        prompt = "Write a short story about a robot."
        
        estimated = self.counter.estimate_completion_tokens(prompt)
        assert isinstance(estimated, int)
        assert estimated > 0
        
        # Test with max_tokens limit
        estimated_limited = self.counter.estimate_completion_tokens(prompt, max_tokens=50)
        assert estimated_limited <= 50
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        cost = self.counter.estimate_cost(prompt_tokens=1000, completion_tokens=500)
        
        assert isinstance(cost, CostEstimate)
        assert cost.prompt_cost > 0
        assert cost.completion_cost > 0
        assert cost.total_cost > 0
        assert cost.model == ModelType.GPT_4.value
    
    def test_estimate_cost_different_model(self):
        """Test cost estimation with different model."""
        cost = self.counter.estimate_cost(
            prompt_tokens=1000, 
            completion_tokens=500,
            model=ModelType.GPT_3_5_TURBO
        )
        
        assert cost.model == ModelType.GPT_3_5_TURBO.value
    
    def test_estimate_cost_unknown_model(self):
        """Test cost estimation with unknown model."""
        # Create counter with custom model
        counter = TokenCounter()
        
        # Remove pricing for current model to simulate unknown model
        unknown_model = ModelType.GPT_4
        if unknown_model in counter.pricing:
            del counter.pricing[unknown_model]
        
        cost = counter.estimate_cost(1000, 500, unknown_model)
        
        assert cost.prompt_cost == 0.0
        assert cost.completion_cost == 0.0
        assert cost.total_cost == 0.0
    
    def test_analyze_text(self):
        """Test text analysis."""
        text = "This is a sample text for analysis. It has multiple sentences."
        
        analysis = self.counter.analyze_text(text)
        
        assert 'token_count' in analysis
        assert 'character_count' in analysis
        assert 'word_count' in analysis
        assert 'tokens_per_word' in analysis
        assert 'characters_per_token' in analysis
        assert 'estimated_cost' in analysis
        
        assert analysis['token_count'] > 0
        assert analysis['character_count'] == len(text)
        assert analysis['word_count'] > 0
        assert isinstance(analysis['estimated_cost'], CostEstimate)
    
    def test_analyze_text_empty(self):
        """Test analyzing empty text."""
        analysis = self.counter.analyze_text("")
        
        assert analysis['token_count'] == 0
        assert analysis['character_count'] == 0
        assert analysis['word_count'] == 0
        assert analysis['tokens_per_word'] == 0
        assert analysis['characters_per_token'] == 0
    
    def test_compare_models(self):
        """Test comparing costs across models."""
        comparison = self.counter.compare_models(prompt_tokens=1000, completion_tokens=500)
        
        assert isinstance(comparison, dict)
        assert len(comparison) > 0
        
        # Check that all model types are included
        for model_type in ModelType:
            if model_type in self.counter.pricing:
                assert model_type.value in comparison
                assert isinstance(comparison[model_type.value], CostEstimate)
    
    def test_batch_count_tokens(self):
        """Test batch token counting."""
        texts = [
            "First text to count",
            "Second text for counting",
            "Third and final text"
        ]
        
        counts = self.counter.batch_count_tokens(texts)
        
        assert len(counts) == len(texts)
        assert all(isinstance(count, int) and count > 0 for count in counts)
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.counter.get_model_info()
        
        assert 'model' in info
        assert 'has_pricing' in info
        assert 'prompt_price_per_1k' in info
        assert 'completion_price_per_1k' in info
        assert 'currency' in info
        assert 'tokenizer_available' in info
        
        assert info['model'] == ModelType.GPT_4.value
        assert info['has_pricing'] is True
    
    def test_get_model_info_specific_model(self):
        """Test getting info for specific model."""
        info = self.counter.get_model_info(ModelType.GPT_3_5_TURBO)
        
        assert info['model'] == ModelType.GPT_3_5_TURBO.value
    
    def test_update_pricing(self):
        """Test updating model pricing."""
        new_pricing = ModelPricing("custom-model", 0.01, 0.02)
        
        self.counter.update_pricing(ModelType.GPT_4, new_pricing)
        
        updated_pricing = self.counter.pricing[ModelType.GPT_4]
        assert updated_pricing.prompt_price_per_1k == 0.01
        assert updated_pricing.completion_price_per_1k == 0.02
    
    def test_get_supported_models(self):
        """Test getting supported models."""
        models = self.counter.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert ModelType.GPT_4.value in models
        assert ModelType.GPT_3_5_TURBO.value in models
    
    def test_calculate_conversation_cost(self):
        """Test calculating conversation cost."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?")
        ]
        
        usage, cost = self.counter.calculate_conversation_cost(
            messages, 
            estimated_response_tokens=50
        )
        
        assert isinstance(usage, TokenUsage)
        assert isinstance(cost, CostEstimate)
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens == 50
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert cost.total_cost > 0
    
    def test_custom_pricing_initialization(self):
        """Test initialization with custom pricing."""
        custom_pricing = {
            "gpt-4": ModelPricing("gpt-4", 0.01, 0.02)
        }
        
        counter = TokenCounter(custom_pricing=custom_pricing)
        
        pricing = counter.pricing[ModelType.GPT_4]
        assert pricing.prompt_price_per_1k == 0.01
        assert pricing.completion_price_per_1k == 0.02
    
    def test_fallback_tokenizer(self):
        """Test fallback when tiktoken is not available."""
        # Create counter and simulate tokenizer failure
        counter = TokenCounter()
        counter._tokenizer = None
        
        token_count = counter.count_tokens("Hello world test")
        
        # Should use fallback method
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_model_type_enum_values(self):
        """Test ModelType enum values."""
        assert ModelType.GPT_4.value == "gpt-4"
        assert ModelType.GPT_3_5_TURBO.value == "gpt-3.5-turbo"
        assert ModelType.CLAUDE_3_OPUS.value == "claude-3-opus"
        assert ModelType.DEEPSEEK_CHAT.value == "deepseek-chat"
    
    def test_different_message_types(self):
        """Test counting tokens for different message types."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        
        for message in messages:
            token_count = self.counter.count_message_tokens(message)
            assert isinstance(token_count, int)
            assert token_count > 0
    
    def test_mixed_message_formats(self):
        """Test counting tokens for mixed message formats."""
        messages = [
            HumanMessage(content="Hello"),
            {"role": "assistant", "content": "Hi there!"},
            "Just a string message"
        ]
        
        usage = self.counter.count_messages_tokens(messages)
        
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens > 0