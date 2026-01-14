"""
Token optimizer for reducing token usage and costs.

This module provides functionality to optimize token usage through
text compression, message summarization, and intelligent truncation.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .token_counter import TokenCounter, TokenUsage, ModelType


class OptimizationStrategy(Enum):
    """Token optimization strategies."""
    TRUNCATE_START = "truncate_start"
    TRUNCATE_END = "truncate_end"
    TRUNCATE_MIDDLE = "truncate_middle"
    SUMMARIZE = "summarize"
    COMPRESS = "compress"
    REMOVE_REDUNDANCY = "remove_redundancy"


@dataclass
class OptimizationResult:
    """Result of token optimization."""
    original_text: str
    optimized_text: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    compression_ratio: float
    strategy_used: str
    
    @property
    def savings_percentage(self) -> float:
        """Calculate percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


class TokenOptimizer:
    """
    Token optimizer for reducing token usage and costs.
    
    Provides various strategies to optimize text and messages
    to reduce token consumption while preserving meaning.
    """
    
    def __init__(self, 
                 token_counter: Optional[TokenCounter] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the token optimizer.
        
        Args:
            token_counter: TokenCounter instance for counting tokens
            logger: Logger for debugging
        """
        self.token_counter = token_counter or TokenCounter()
        self.logger = logger or logging.getLogger(__name__)
        
        # Optimization patterns
        self._whitespace_pattern = re.compile(r'\s+')
        self._redundant_punctuation = re.compile(r'[.]{2,}|[!]{2,}|[?]{2,}')
        self._redundant_words = re.compile(r'\b(very|really|quite|rather|pretty|fairly)\s+', re.IGNORECASE)
        self._filler_words = re.compile(r'\b(um|uh|er|ah|like|you know|I mean)\b', re.IGNORECASE)
    
    def truncate_text(self, 
                     text: str, 
                     max_tokens: int,
                     strategy: OptimizationStrategy = OptimizationStrategy.TRUNCATE_END,
                     preserve_sentences: bool = True) -> OptimizationResult:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            strategy: Truncation strategy
            preserve_sentences: Whether to preserve sentence boundaries
            
        Returns:
            OptimizationResult with truncated text
        """
        original_tokens = self.token_counter.count_tokens(text)
        
        if original_tokens <= max_tokens:
            return OptimizationResult(
                original_text=text,
                optimized_text=text,
                original_tokens=original_tokens,
                optimized_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=1.0,
                strategy_used=strategy.value
            )
        
        if strategy == OptimizationStrategy.TRUNCATE_END:
            optimized_text = self._truncate_end(text, max_tokens, preserve_sentences)
        elif strategy == OptimizationStrategy.TRUNCATE_START:
            optimized_text = self._truncate_start(text, max_tokens, preserve_sentences)
        elif strategy == OptimizationStrategy.TRUNCATE_MIDDLE:
            optimized_text = self._truncate_middle(text, max_tokens, preserve_sentences)
        else:
            optimized_text = self._truncate_end(text, max_tokens, preserve_sentences)
        
        optimized_tokens = self.token_counter.count_tokens(optimized_text)
        
        return OptimizationResult(
            original_text=text,
            optimized_text=optimized_text,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=original_tokens - optimized_tokens,
            compression_ratio=optimized_tokens / original_tokens,
            strategy_used=strategy.value
        )
    
    def _truncate_end(self, text: str, max_tokens: int, preserve_sentences: bool) -> str:
        """Truncate text from the end."""
        if preserve_sentences:
            sentences = self._split_sentences(text)
            result = ""
            
            for sentence in sentences:
                test_text = result + sentence
                if self.token_counter.count_tokens(test_text) <= max_tokens:
                    result = test_text
                else:
                    break
            
            return result.strip()
        else:
            # Binary search for optimal truncation point
            words = text.split()
            left, right = 0, len(words)
            
            while left < right:
                mid = (left + right + 1) // 2
                test_text = " ".join(words[:mid])
                
                if self.token_counter.count_tokens(test_text) <= max_tokens:
                    left = mid
                else:
                    right = mid - 1
            
            return " ".join(words[:left])
    
    def _truncate_start(self, text: str, max_tokens: int, preserve_sentences: bool) -> str:
        """Truncate text from the start."""
        if preserve_sentences:
            sentences = self._split_sentences(text)
            result = ""
            
            for sentence in reversed(sentences):
                test_text = sentence + result
                if self.token_counter.count_tokens(test_text) <= max_tokens:
                    result = test_text
                else:
                    break
            
            return result.strip()
        else:
            words = text.split()
            left, right = 0, len(words)
            
            while left < right:
                mid = (left + right) // 2
                test_text = " ".join(words[mid:])
                
                if self.token_counter.count_tokens(test_text) <= max_tokens:
                    right = mid
                else:
                    left = mid + 1
            
            return " ".join(words[left:])
    
    def _truncate_middle(self, text: str, max_tokens: int, preserve_sentences: bool) -> str:
        """Truncate text from the middle, keeping start and end."""
        target_tokens = max_tokens - 10  # Reserve tokens for ellipsis
        
        if preserve_sentences:
            sentences = self._split_sentences(text)
            if len(sentences) <= 2:
                return self._truncate_end(text, max_tokens, preserve_sentences)
            
            # Keep first and last sentences, truncate middle
            first_sentence = sentences[0]
            last_sentence = sentences[-1]
            
            first_tokens = self.token_counter.count_tokens(first_sentence)
            last_tokens = self.token_counter.count_tokens(last_sentence)
            
            if first_tokens + last_tokens >= target_tokens:
                return self._truncate_end(text, max_tokens, preserve_sentences)
            
            return f"{first_sentence} [...] {last_sentence}"
        else:
            words = text.split()
            if len(words) <= 20:
                return self._truncate_end(text, max_tokens, preserve_sentences)
            
            # Keep first and last 10 words, truncate middle
            start_words = words[:10]
            end_words = words[-10:]
            
            test_text = " ".join(start_words + ["[...]"] + end_words)
            
            if self.token_counter.count_tokens(test_text) <= max_tokens:
                return test_text
            else:
                return self._truncate_end(text, max_tokens, preserve_sentences)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def compress_text(self, text: str) -> OptimizationResult:
        """
        Compress text by removing redundancy and unnecessary elements.
        
        Args:
            text: Text to compress
            
        Returns:
            OptimizationResult with compressed text
        """
        original_tokens = self.token_counter.count_tokens(text)
        optimized_text = text
        
        # Remove extra whitespace
        optimized_text = self._whitespace_pattern.sub(' ', optimized_text)
        
        # Remove redundant punctuation
        optimized_text = self._redundant_punctuation.sub('.', optimized_text)
        
        # Remove redundant words
        optimized_text = self._redundant_words.sub('', optimized_text)
        
        # Remove filler words
        optimized_text = self._filler_words.sub('', optimized_text)
        
        # Clean up extra spaces
        optimized_text = optimized_text.strip()
        optimized_text = self._whitespace_pattern.sub(' ', optimized_text)
        
        optimized_tokens = self.token_counter.count_tokens(optimized_text)
        
        return OptimizationResult(
            original_text=text,
            optimized_text=optimized_text,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=original_tokens - optimized_tokens,
            compression_ratio=optimized_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy_used=OptimizationStrategy.COMPRESS.value
        )
    
    def optimize_messages(self, 
                         messages: List[Union[BaseMessage, Dict[str, Any]]],
                         max_tokens: int,
                         preserve_system_message: bool = True,
                         preserve_last_messages: int = 2) -> Tuple[List[Union[BaseMessage, Dict[str, Any]]], OptimizationResult]:
        """
        Optimize a list of messages to fit within token limit.
        
        Args:
            messages: List of messages to optimize
            max_tokens: Maximum total tokens
            preserve_system_message: Whether to preserve system message
            preserve_last_messages: Number of recent messages to preserve
            
        Returns:
            Tuple of (optimized_messages, OptimizationResult)
        """
        if not messages:
            return messages, OptimizationResult(
                original_text="",
                optimized_text="",
                original_tokens=0,
                optimized_tokens=0,
                tokens_saved=0,
                compression_ratio=1.0,
                strategy_used="no_optimization"
            )
        
        original_usage = self.token_counter.count_messages_tokens(messages)
        
        if original_usage.total_tokens <= max_tokens:
            return messages, OptimizationResult(
                original_text=str(messages),
                optimized_text=str(messages),
                original_tokens=original_usage.total_tokens,
                optimized_tokens=original_usage.total_tokens,
                tokens_saved=0,
                compression_ratio=1.0,
                strategy_used="no_optimization"
            )
        
        optimized_messages = []
        current_tokens = 0
        
        # Always preserve system message if requested
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if self._is_system_message(msg) and preserve_system_message:
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        # Add system messages first
        for msg in system_messages:
            msg_tokens = self.token_counter.count_message_tokens(msg)
            if current_tokens + msg_tokens <= max_tokens:
                optimized_messages.append(msg)
                current_tokens += msg_tokens
        
        # Preserve last N messages
        preserved_messages = other_messages[-preserve_last_messages:] if preserve_last_messages > 0 else []
        remaining_messages = other_messages[:-preserve_last_messages] if preserve_last_messages > 0 else other_messages
        
        # Calculate tokens for preserved messages
        preserved_tokens = sum(self.token_counter.count_message_tokens(msg) for msg in preserved_messages)
        
        # Add messages from the beginning until we run out of space
        for msg in remaining_messages:
            msg_tokens = self.token_counter.count_message_tokens(msg)
            if current_tokens + msg_tokens + preserved_tokens <= max_tokens:
                optimized_messages.append(msg)
                current_tokens += msg_tokens
            else:
                # Try to compress the message
                compressed_msg = self._compress_message(msg, max_tokens - current_tokens - preserved_tokens)
                if compressed_msg:
                    optimized_messages.append(compressed_msg)
                    current_tokens += self.token_counter.count_message_tokens(compressed_msg)
                break
        
        # Add preserved messages
        optimized_messages.extend(preserved_messages)
        
        optimized_usage = self.token_counter.count_messages_tokens(optimized_messages)
        
        return optimized_messages, OptimizationResult(
            original_text=f"{len(messages)} messages",
            optimized_text=f"{len(optimized_messages)} messages",
            original_tokens=original_usage.total_tokens,
            optimized_tokens=optimized_usage.total_tokens,
            tokens_saved=original_usage.total_tokens - optimized_usage.total_tokens,
            compression_ratio=optimized_usage.total_tokens / original_usage.total_tokens,
            strategy_used="message_optimization"
        )
    
    def _is_system_message(self, message: Union[BaseMessage, Dict[str, Any]]) -> bool:
        """Check if a message is a system message."""
        if isinstance(message, SystemMessage):
            return True
        elif isinstance(message, dict):
            return message.get('role') == 'system'
        return False
    
    def _compress_message(self, 
                         message: Union[BaseMessage, Dict[str, Any]], 
                         max_tokens: int) -> Optional[Union[BaseMessage, Dict[str, Any]]]:
        """Compress a single message to fit within token limit."""
        if isinstance(message, BaseMessage):
            content = str(message.content)
            compressed_result = self.compress_text(content)
            
            if compressed_result.optimized_tokens <= max_tokens:
                # Create new message with compressed content
                if isinstance(message, HumanMessage):
                    return HumanMessage(content=compressed_result.optimized_text)
                elif isinstance(message, AIMessage):
                    return AIMessage(content=compressed_result.optimized_text)
                elif isinstance(message, SystemMessage):
                    return SystemMessage(content=compressed_result.optimized_text)
                else:
                    return message.__class__(content=compressed_result.optimized_text)
            
            # If compression isn't enough, try truncation
            truncated_result = self.truncate_text(content, max_tokens)
            if isinstance(message, HumanMessage):
                return HumanMessage(content=truncated_result.optimized_text)
            elif isinstance(message, AIMessage):
                return AIMessage(content=truncated_result.optimized_text)
            elif isinstance(message, SystemMessage):
                return SystemMessage(content=truncated_result.optimized_text)
            else:
                return message.__class__(content=truncated_result.optimized_text)
        
        elif isinstance(message, dict):
            content = message.get('content', '')
            compressed_result = self.compress_text(str(content))
            
            if compressed_result.optimized_tokens <= max_tokens:
                new_message = message.copy()
                new_message['content'] = compressed_result.optimized_text
                return new_message
            
            # Try truncation
            truncated_result = self.truncate_text(str(content), max_tokens)
            new_message = message.copy()
            new_message['content'] = truncated_result.optimized_text
            return new_message
        
        return None
    
    def summarize_context(self, 
                         text: str, 
                         target_tokens: int,
                         summarizer: Optional[Callable[[str], str]] = None) -> OptimizationResult:
        """
        Summarize text to reduce token count.
        
        Args:
            text: Text to summarize
            target_tokens: Target number of tokens
            summarizer: Custom summarization function
            
        Returns:
            OptimizationResult with summarized text
        """
        original_tokens = self.token_counter.count_tokens(text)
        
        if original_tokens <= target_tokens:
            return OptimizationResult(
                original_text=text,
                optimized_text=text,
                original_tokens=original_tokens,
                optimized_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=1.0,
                strategy_used=OptimizationStrategy.SUMMARIZE.value
            )
        
        if summarizer:
            # Use custom summarizer
            summarized_text = summarizer(text)
        else:
            # Simple extractive summarization
            summarized_text = self._simple_summarize(text, target_tokens)
        
        optimized_tokens = self.token_counter.count_tokens(summarized_text)
        
        return OptimizationResult(
            original_text=text,
            optimized_text=summarized_text,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=original_tokens - optimized_tokens,
            compression_ratio=optimized_tokens / original_tokens,
            strategy_used=OptimizationStrategy.SUMMARIZE.value
        )
    
    def _simple_summarize(self, text: str, target_tokens: int) -> str:
        """Simple extractive summarization."""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return self.truncate_text(text, target_tokens).optimized_text
        
        # Score sentences by position and length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Prefer sentences at the beginning and end
            position_score = 1.0 if i < len(sentences) * 0.3 else 0.5
            if i >= len(sentences) * 0.7:
                position_score = 0.8
            
            # Prefer medium-length sentences
            length_score = min(len(sentence.split()) / 20, 1.0)
            
            total_score = position_score * length_score
            scored_sentences.append((sentence, total_score))
        
        # Sort by score and select sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences = []
        current_tokens = 0
        
        for sentence, score in scored_sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if current_tokens + sentence_tokens <= target_tokens:
                selected_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        # Maintain original order
        original_order = []
        for sentence in sentences:
            if sentence in selected_sentences:
                original_order.append(sentence)
        
        return " ".join(original_order)
    
    def optimize_for_cost(self, 
                         text: str, 
                         max_cost: float,
                         model: Optional[Union[str, ModelType]] = None) -> OptimizationResult:
        """
        Optimize text to stay within cost limit.
        
        Args:
            text: Text to optimize
            max_cost: Maximum cost in USD
            model: Model to use for cost calculation
            
        Returns:
            OptimizationResult with cost-optimized text
        """
        original_tokens = self.token_counter.count_tokens(text)
        original_cost = self.token_counter.estimate_cost(original_tokens, 0, model)
        
        if original_cost.total_cost <= max_cost:
            return OptimizationResult(
                original_text=text,
                optimized_text=text,
                original_tokens=original_tokens,
                optimized_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=1.0,
                strategy_used="cost_optimization"
            )
        
        # Calculate target tokens based on cost
        model_type = model if isinstance(model, ModelType) else (
            ModelType(model) if model else self.token_counter.model
        )
        
        pricing = self.token_counter.pricing.get(model_type)
        if not pricing:
            # Fallback to compression
            return self.compress_text(text)
        
        # Calculate max tokens for the cost
        max_tokens = int(max_cost / pricing.prompt_price_per_1k * 1000)
        
        # Try compression first
        compressed = self.compress_text(text)
        if compressed.optimized_tokens <= max_tokens:
            return compressed
        
        # If compression isn't enough, truncate
        return self.truncate_text(text, max_tokens)
    
    def batch_optimize(self, 
                      texts: List[str], 
                      max_tokens_per_text: int,
                      strategy: OptimizationStrategy = OptimizationStrategy.COMPRESS) -> List[OptimizationResult]:
        """
        Optimize multiple texts in batch.
        
        Args:
            texts: List of texts to optimize
            max_tokens_per_text: Maximum tokens per text
            strategy: Optimization strategy to use
            
        Returns:
            List of OptimizationResult objects
        """
        results = []
        
        for text in texts:
            if strategy == OptimizationStrategy.COMPRESS:
                result = self.compress_text(text)
                if result.optimized_tokens > max_tokens_per_text:
                    result = self.truncate_text(result.optimized_text, max_tokens_per_text)
            elif strategy in [OptimizationStrategy.TRUNCATE_START, 
                            OptimizationStrategy.TRUNCATE_END, 
                            OptimizationStrategy.TRUNCATE_MIDDLE]:
                result = self.truncate_text(text, max_tokens_per_text, strategy)
            elif strategy == OptimizationStrategy.SUMMARIZE:
                result = self.summarize_context(text, max_tokens_per_text)
            else:
                result = self.compress_text(text)
            
            results.append(result)
        
        return results
    
    def get_optimization_stats(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Get statistics from optimization results.
        
        Args:
            results: List of optimization results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        total_original_tokens = sum(r.original_tokens for r in results)
        total_optimized_tokens = sum(r.optimized_tokens for r in results)
        total_saved_tokens = sum(r.tokens_saved for r in results)
        
        avg_compression_ratio = sum(r.compression_ratio for r in results) / len(results)
        avg_savings_percentage = sum(r.savings_percentage for r in results) / len(results)
        
        return {
            'total_texts': len(results),
            'total_original_tokens': total_original_tokens,
            'total_optimized_tokens': total_optimized_tokens,
            'total_saved_tokens': total_saved_tokens,
            'average_compression_ratio': avg_compression_ratio,
            'average_savings_percentage': avg_savings_percentage,
            'strategies_used': list(set(r.strategy_used for r in results))
        }