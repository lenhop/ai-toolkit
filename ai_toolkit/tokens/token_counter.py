"""
Token counter for calculating token usage and costs.

This module provides functionality to count tokens in text and messages,
estimate API costs, and analyze token usage patterns.

Classes:
    ModelType: Enum for supported model types
        - GPT_4, GPT_4_TURBO, GPT_4O, GPT_3_5_TURBO
        - CLAUDE_3_OPUS, CLAUDE_3_SONNET, CLAUDE_3_HAIKU
        - DEEPSEEK_CHAT, QWEN_TURBO, GLM_4
    
    TokenUsage: Data class for token usage information
        - Tracks prompt, completion, and total tokens
        
        Properties:
            input_tokens: Alias for prompt_tokens
            output_tokens: Alias for completion_tokens
    
    CostEstimate: Data class for cost estimation
        - Tracks prompt, completion, and total costs
        
        Fields:
            prompt_cost: Cost for prompt tokens
            completion_cost: Cost for completion tokens
            total_cost: Total cost
            currency: Currency (default USD)
            model: Model name
    
    ModelPricing: Data class for model pricing information
        - Defines pricing per 1K tokens
        
        Fields:
            model: Model name
            prompt_price_per_1k: Price per 1K prompt tokens
            completion_price_per_1k: Price per 1K completion tokens
            currency: Currency
        
        Methods:
            calculate_cost(usage): Calculate cost from token usage
    
    TokenCounter: Counter for token usage and costs
        - Counts tokens using tiktoken
        - Estimates costs for multiple models
        - Analyzes token usage patterns
        
        Methods:
            __init__(model, custom_pricing, logger): Initialize counter
            count_tokens(text): Count tokens in text
            count_message_tokens(message): Count tokens in single message
            count_messages_tokens(messages): Count tokens in message list
            estimate_completion_tokens(prompt, max_tokens, estimated_ratio): Estimate completion tokens
            estimate_cost(prompt_tokens, completion_tokens, model): Estimate cost
            analyze_text(text): Analyze text for token patterns
            compare_models(prompt_tokens, completion_tokens): Compare costs across models
            batch_count_tokens(texts): Count tokens for multiple texts
            get_model_info(model): Get model information
            update_pricing(model, pricing): Update model pricing
            get_supported_models(): Get list of supported models
            calculate_conversation_cost(messages, estimated_response_tokens): Calculate conversation cost
"""

import tiktoken
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class ModelType(Enum):
    """Supported model types for token counting."""
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    TEXT_DAVINCI_003 = "text-davinci-003"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    DEEPSEEK_CHAT = "deepseek-chat"
    QWEN_TURBO = "qwen-turbo"
    GLM_4 = "glm-4"


@dataclass
class TokenUsage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def input_tokens(self) -> int:
        """Alias for prompt_tokens."""
        return self.prompt_tokens
    
    @property
    def output_tokens(self) -> int:
        """Alias for completion_tokens."""
        return self.completion_tokens
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage objects."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


@dataclass
class CostEstimate:
    """Cost estimation for token usage."""
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    model: str = ""
    
    def __add__(self, other: 'CostEstimate') -> 'CostEstimate':
        """Add two CostEstimate objects."""
        return CostEstimate(
            prompt_cost=self.prompt_cost + other.prompt_cost,
            completion_cost=self.completion_cost + other.completion_cost,
            total_cost=self.total_cost + other.total_cost,
            currency=self.currency,
            model=f"{self.model},{other.model}" if self.model != other.model else self.model
        )


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    model: str
    prompt_price_per_1k: float  # Price per 1K prompt tokens
    completion_price_per_1k: float  # Price per 1K completion tokens
    currency: str = "USD"
    
    def calculate_cost(self, usage: TokenUsage) -> CostEstimate:
        """Calculate cost based on token usage."""
        prompt_cost = (usage.prompt_tokens / 1000) * self.prompt_price_per_1k
        completion_cost = (usage.completion_tokens / 1000) * self.completion_price_per_1k
        total_cost = prompt_cost + completion_cost
        
        return CostEstimate(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost,
            currency=self.currency,
            model=self.model
        )


class TokenCounter:
    """
    Token counter for various AI models.
    
    Provides functionality to count tokens in text and messages,
    estimate costs, and analyze token usage patterns.
    """
    
    # Default pricing (as of 2024, subject to change)
    DEFAULT_PRICING = {
        ModelType.GPT_4: ModelPricing("gpt-4", 0.03, 0.06),
        ModelType.GPT_4_TURBO: ModelPricing("gpt-4-turbo", 0.01, 0.03),
        ModelType.GPT_4O: ModelPricing("gpt-4o", 0.005, 0.015),
        ModelType.GPT_3_5_TURBO: ModelPricing("gpt-3.5-turbo", 0.001, 0.002),
        ModelType.TEXT_DAVINCI_003: ModelPricing("text-davinci-003", 0.02, 0.02),
        ModelType.TEXT_EMBEDDING_ADA_002: ModelPricing("text-embedding-ada-002", 0.0001, 0.0),
        ModelType.CLAUDE_3_OPUS: ModelPricing("claude-3-opus", 0.015, 0.075),
        ModelType.CLAUDE_3_SONNET: ModelPricing("claude-3-sonnet", 0.003, 0.015),
        ModelType.CLAUDE_3_HAIKU: ModelPricing("claude-3-haiku", 0.00025, 0.00125),
        ModelType.DEEPSEEK_CHAT: ModelPricing("deepseek-chat", 0.0014, 0.0028),
        ModelType.QWEN_TURBO: ModelPricing("qwen-turbo", 0.002, 0.006),
        ModelType.GLM_4: ModelPricing("glm-4", 0.005, 0.005),
    }
    
    def __init__(self, 
                 model: Union[str, ModelType] = ModelType.GPT_4,
                 custom_pricing: Optional[Dict[str, ModelPricing]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the token counter.
        
        Args:
            model: Model type or name for token counting
            custom_pricing: Custom pricing information
            logger: Logger for debugging
        """
        try:
            self.model = model if isinstance(model, ModelType) else ModelType(model)
        except (ValueError, AttributeError):
            # Fallback to default model if model string doesn't match enum
            logger_instance = logger or logging.getLogger(__name__)
            logger_instance.warning(f"Unsupported model '{model}', falling back to GPT_4")
            self.model = ModelType.GPT_4
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup pricing
        self.pricing = self.DEFAULT_PRICING.copy()
        if custom_pricing:
            for model_name, pricing in custom_pricing.items():
                if isinstance(model_name, str):
                    # Find matching ModelType
                    for model_type in ModelType:
                        if model_type.value == model_name:
                            self.pricing[model_type] = pricing
                            break
                else:
                    self.pricing[model_name] = pricing
        
        # Initialize tokenizer
        self._tokenizer = None
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize the appropriate tokenizer."""
        try:
            # Map model types to tiktoken encodings
            encoding_map = {
                ModelType.GPT_4: "cl100k_base",
                ModelType.GPT_4_TURBO: "cl100k_base",
                ModelType.GPT_4O: "cl100k_base",
                ModelType.GPT_3_5_TURBO: "cl100k_base",
                ModelType.TEXT_DAVINCI_003: "p50k_base",
                ModelType.TEXT_EMBEDDING_ADA_002: "cl100k_base",
            }
            
            # For OpenAI models, use tiktoken
            if self.model in encoding_map:
                encoding_name = encoding_map[self.model]
                self._tokenizer = tiktoken.get_encoding(encoding_name)
                self.logger.debug(f"Initialized tiktoken with encoding: {encoding_name}")
            else:
                # For other models, use a fallback tokenizer
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                self.logger.warning(f"Using fallback tokenizer for model: {self.model.value}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            # Fallback to basic word counting
            self._tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            if self._tokenizer:
                return len(self._tokenizer.encode(text))
            else:
                # Fallback: approximate token count (1 token ≈ 0.75 words)
                word_count = len(text.split())
                return int(word_count / 0.75)
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            # Ultimate fallback: character count / 4
            return len(text) // 4
    
    def count_message_tokens(self, message: Union[BaseMessage, Dict[str, Any]]) -> int:
        """
        Count tokens in a single message.
        
        Args:
            message: Message to count tokens for
            
        Returns:
            Number of tokens
        """
        if isinstance(message, BaseMessage):
            content = message.content
            # Add tokens for message formatting
            base_tokens = 4  # Approximate overhead per message
            return base_tokens + self.count_tokens(str(content))
        
        elif isinstance(message, dict):
            # Handle dictionary format
            content = message.get('content', '')
            role = message.get('role', '')
            base_tokens = 4  # Message formatting overhead
            return base_tokens + self.count_tokens(str(content)) + self.count_tokens(role)
        
        else:
            return self.count_tokens(str(message))
    
    def count_messages_tokens(self, messages: List[Union[BaseMessage, Dict[str, Any]]]) -> TokenUsage:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of messages
            
        Returns:
            TokenUsage object with token counts
        """
        total_tokens = 0
        
        for message in messages:
            total_tokens += self.count_message_tokens(message)
        
        # Add conversation overhead (varies by model)
        conversation_overhead = 2  # Approximate
        total_tokens += conversation_overhead
        
        return TokenUsage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens
        )
    
    def estimate_completion_tokens(self, 
                                 prompt: str, 
                                 max_tokens: Optional[int] = None,
                                 estimated_ratio: float = 0.5) -> int:
        """
        Estimate completion tokens based on prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens allowed for completion
            estimated_ratio: Estimated completion/prompt ratio
            
        Returns:
            Estimated completion tokens
        """
        prompt_tokens = self.count_tokens(prompt)
        estimated_completion = int(prompt_tokens * estimated_ratio)
        
        if max_tokens:
            estimated_completion = min(estimated_completion, max_tokens)
        
        return estimated_completion
    
    def estimate_cost(self, 
                     prompt_tokens: int, 
                     completion_tokens: int = 0,
                     model: Optional[Union[str, ModelType]] = None) -> CostEstimate:
        """
        Estimate cost based on token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model to use for pricing (defaults to instance model)
            
        Returns:
            CostEstimate object
        """
        model_type = model if isinstance(model, ModelType) else (
            ModelType(model) if model else self.model
        )
        
        if model_type not in self.pricing:
            self.logger.warning(f"No pricing info for model: {model_type.value}")
            return CostEstimate(model=model_type.value)
        
        pricing = self.pricing[model_type]
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        return pricing.calculate_cost(usage)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for token usage patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        if not text:
            return {
                'token_count': 0,
                'character_count': 0,
                'word_count': 0,
                'tokens_per_word': 0,
                'characters_per_token': 0
            }
        
        token_count = self.count_tokens(text)
        character_count = len(text)
        word_count = len(text.split())
        
        tokens_per_word = token_count / word_count if word_count > 0 else 0
        characters_per_token = character_count / token_count if token_count > 0 else 0
        
        return {
            'token_count': token_count,
            'character_count': character_count,
            'word_count': word_count,
            'tokens_per_word': tokens_per_word,
            'characters_per_token': characters_per_token,
            'estimated_cost': self.estimate_cost(token_count)
        }
    
    def compare_models(self, 
                      prompt_tokens: int, 
                      completion_tokens: int = 0) -> Dict[str, CostEstimate]:
        """
        Compare costs across different models.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Dictionary of model names to cost estimates
        """
        results = {}
        
        for model_type, pricing in self.pricing.items():
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            cost = pricing.calculate_cost(usage)
            results[model_type.value] = cost
        
        return results
    
    def batch_count_tokens(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts efficiently.
        
        Args:
            texts: List of texts to count tokens for
            
        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]
    
    def get_model_info(self, model: Optional[Union[str, ModelType]] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model: Model to get info for (defaults to instance model)
            
        Returns:
            Model information
        """
        model_type = model if isinstance(model, ModelType) else (
            ModelType(model) if model else self.model
        )
        
        pricing = self.pricing.get(model_type)
        
        return {
            'model': model_type.value,
            'has_pricing': pricing is not None,
            'prompt_price_per_1k': pricing.prompt_price_per_1k if pricing else None,
            'completion_price_per_1k': pricing.completion_price_per_1k if pricing else None,
            'currency': pricing.currency if pricing else None,
            'tokenizer_available': self._tokenizer is not None
        }
    
    def update_pricing(self, model: Union[str, ModelType], pricing: ModelPricing):
        """
        Update pricing for a model.
        
        Args:
            model: Model to update pricing for
            pricing: New pricing information
        """
        model_type = model if isinstance(model, ModelType) else ModelType(model)
        self.pricing[model_type] = pricing
        self.logger.info(f"Updated pricing for {model_type.value}")
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported models.
        
        Returns:
            List of supported model names
        """
        return [model_type.value for model_type in ModelType]
    
    def calculate_conversation_cost(self, 
                                  messages: List[Union[BaseMessage, Dict[str, Any]]],
                                  estimated_response_tokens: int = 0) -> Tuple[TokenUsage, CostEstimate]:
        """
        Calculate cost for an entire conversation.
        
        Args:
            messages: List of conversation messages
            estimated_response_tokens: Estimated tokens for the response
            
        Returns:
            Tuple of (TokenUsage, CostEstimate)
        """
        usage = self.count_messages_tokens(messages)
        usage.completion_tokens = estimated_response_tokens
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        
        cost = self.estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        
        return usage, cost