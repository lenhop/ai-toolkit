"""
Retry manager for handling failed operations with intelligent retry strategies.

This module provides sophisticated retry mechanisms including exponential backoff,
jitter, circuit breakers, and conditional retry logic.

Classes:
    RetryStrategy: Enum for retry strategies
        - EXPONENTIAL: Exponential backoff
        - LINEAR: Linear backoff
        - FIBONACCI: Fibonacci backoff
        - FIXED: Fixed delay
    
    RetryManager: Manager for retry operations
        - Implements multiple retry strategies
        - Includes circuit breaker pattern
        - Tracks retry statistics
        - Supports conditional retry logic
        
        Methods:
            __init__(max_attempts, base_delay, max_delay, strategy): Initialize manager
            retry(func, *args, **kwargs): Retry function with strategy
            retry_with_backoff(func, *args, **kwargs): Retry with exponential backoff
            retry_with_condition(func, condition, *args, **kwargs): Conditional retry
            should_retry(error, attempt): Determine if should retry
            calculate_delay(attempt): Calculate retry delay
            reset_circuit(): Reset circuit breaker
            get_stats(): Get retry statistics
            clear_stats(): Clear retry statistics
"""

import time
import random
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Type
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps

from .exception_types import (
    AIException, RateLimitError, TimeoutError, APIError,
    AuthenticationError, ErrorSeverity
)


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    
    # Conditional retry settings
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        RateLimitError, TimeoutError, APIError
    ])
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        AuthenticationError
    ])
    
    # Custom retry condition
    retry_condition: Optional[Callable[[Exception, int], bool]] = None
    
    # Custom delay calculation
    delay_calculator: Optional[Callable[[int, float], float]] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # Successes needed to close circuit
    
    
@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    delay: float
    exception: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = False


class RetryManager:
    """
    Intelligent retry manager with multiple strategies and circuit breaking.
    
    Provides sophisticated retry mechanisms including exponential backoff,
    jitter, circuit breakers, and conditional retry logic.
    """
    
    def __init__(self,
                 config: Optional[RetryConfig] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the retry manager.
        
        Args:
            config: Retry configuration
            circuit_config: Circuit breaker configuration
            logger: Logger for retry events
        """
        self.config = config or RetryConfig()
        self.circuit_config = circuit_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._success_count = 0
        
        # Retry statistics
        self._total_attempts = 0
        self._total_successes = 0
        self._total_failures = 0
        self._retry_history: List[RetryAttempt] = []
        self._max_history = 1000
    
    def retry(self,
              func: Callable,
              *args,
              config: Optional[RetryConfig] = None,
              **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            config: Override retry configuration
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        retry_config = config or self.config
        
        # Check circuit breaker
        if self.circuit_config and not self._can_execute():
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        last_exception = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            self._total_attempts += 1
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Success
                self._record_success(attempt)
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not self._should_retry(e, attempt, retry_config):
                    self._record_failure(attempt, e)
                    raise e
                
                # Calculate delay for next attempt
                if attempt < retry_config.max_attempts:
                    delay = self._calculate_delay(attempt, retry_config)
                    
                    # Record the attempt
                    retry_attempt = RetryAttempt(
                        attempt_number=attempt,
                        delay=delay,
                        exception=e,
                        success=False
                    )
                    self._record_attempt(retry_attempt)
                    
                    # Log the retry
                    self.logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    
                    # Wait before retry
                    time.sleep(delay)
        
        # All attempts failed
        self._record_failure(retry_config.max_attempts, last_exception)
        raise last_exception
    
    def retry_with_backoff(self,
                          func: Callable,
                          max_attempts: int = 3,
                          base_delay: float = 1.0,
                          max_delay: float = 60.0,
                          exponential_base: float = 2.0,
                          jitter: bool = True) -> Any:
        """
        Convenience method for exponential backoff retry.
        
        Args:
            func: Function to execute
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential calculation
            jitter: Whether to add jitter
            
        Returns:
            Function result
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter
        )
        
        return self.retry(func, config=config)
    
    def retry_with_condition(self,
                           func: Callable,
                           condition: Callable[[Exception, int], bool],
                           max_attempts: int = 3,
                           base_delay: float = 1.0) -> Any:
        """
        Retry with custom condition.
        
        Args:
            func: Function to execute
            condition: Function that returns True if should retry
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            
        Returns:
            Function result
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            retry_condition=condition
        )
        
        return self.retry(func, config=config)
    
    def should_retry(self,
                    exception: Exception,
                    attempt: int = 1) -> bool:
        """
        Check if an exception should trigger a retry.
        
        Args:
            exception: Exception to check
            attempt: Current attempt number
            
        Returns:
            True if should retry
        """
        return self._should_retry(exception, attempt, self.config)
    
    def _should_retry(self,
                     exception: Exception,
                     attempt: int,
                     config: RetryConfig) -> bool:
        """Internal method to check if should retry."""
        # Check if we've exceeded max attempts
        if attempt >= config.max_attempts:
            return False
        
        # Use custom retry condition if provided
        if config.retry_condition:
            return config.retry_condition(exception, attempt)
        
        # Check non-retryable exceptions first
        for non_retryable in config.non_retryable_exceptions:
            if isinstance(exception, non_retryable):
                return False
        
        # Check retryable exceptions
        for retryable in config.retryable_exceptions:
            if isinstance(exception, retryable):
                return True
        
        # Special handling for AI exceptions
        if isinstance(exception, AIException):
            # Don't retry critical errors
            if exception.severity == ErrorSeverity.CRITICAL:
                return False
            
            # Retry based on category
            if exception.category.value in ['api', 'timeout', 'rate_limit']:
                return True
        
        # Default: don't retry unknown exceptions
        return False
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt."""
        # Use custom delay calculator if provided
        if config.delay_calculator:
            delay = config.delay_calculator(attempt, config.base_delay)
        else:
            # Calculate based on strategy
            if config.strategy == RetryStrategy.FIXED_DELAY:
                delay = config.base_delay
            
            elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                delay = config.base_delay * (config.exponential_base ** (attempt - 1))
            
            elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
                delay = config.base_delay * attempt
            
            elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
                delay = config.base_delay * self._fibonacci(attempt)
            
            else:
                delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Add jitter if enabled
        if config.jitter:
            jitter_amount = delay * config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _can_execute(self) -> bool:
        """Check if circuit breaker allows execution."""
        if not self.circuit_config:
            return True
        
        current_time = time.time()
        
        if self._circuit_state == CircuitState.CLOSED:
            return True
        
        elif self._circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self._last_failure_time >= self.circuit_config.recovery_timeout:
                self._circuit_state = CircuitState.HALF_OPEN
                self._success_count = 0
                self.logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            return False
        
        elif self._circuit_state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _record_success(self, attempt: int):
        """Record a successful execution."""
        self._total_successes += 1
        
        # Record attempt
        retry_attempt = RetryAttempt(
            attempt_number=attempt,
            delay=0.0,
            success=True
        )
        self._record_attempt(retry_attempt)
        
        # Update circuit breaker
        if self.circuit_config:
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.circuit_config.success_threshold:
                    self._circuit_state = CircuitState.CLOSED
                    self._failure_count = 0
                    self.logger.info("Circuit breaker moved to CLOSED state")
            else:
                self._failure_count = 0
    
    def _record_failure(self, attempt: int, exception: Exception):
        """Record a failed execution."""
        self._total_failures += 1
        
        # Record attempt
        retry_attempt = RetryAttempt(
            attempt_number=attempt,
            delay=0.0,
            exception=exception,
            success=False
        )
        self._record_attempt(retry_attempt)
        
        # Update circuit breaker
        if self.circuit_config:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if (self._circuit_state == CircuitState.CLOSED and 
                self._failure_count >= self.circuit_config.failure_threshold):
                self._circuit_state = CircuitState.OPEN
                self.logger.warning("Circuit breaker moved to OPEN state")
            
            elif self._circuit_state == CircuitState.HALF_OPEN:
                self._circuit_state = CircuitState.OPEN
                self.logger.warning("Circuit breaker moved back to OPEN state")
    
    def _record_attempt(self, attempt: RetryAttempt):
        """Record a retry attempt."""
        self._retry_history.append(attempt)
        
        # Trim history if too long
        if len(self._retry_history) > self._max_history:
            self._retry_history = self._retry_history[-self._max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics."""
        success_rate = (self._total_successes / self._total_attempts 
                       if self._total_attempts > 0 else 0)
        
        return {
            'total_attempts': self._total_attempts,
            'total_successes': self._total_successes,
            'total_failures': self._total_failures,
            'success_rate': success_rate,
            'circuit_state': self._circuit_state.value if self.circuit_config else None,
            'failure_count': self._failure_count,
            'recent_attempts': len(self._retry_history)
        }
    
    def get_recent_attempts(self, limit: int = 10) -> List[RetryAttempt]:
        """Get recent retry attempts."""
        return self._retry_history[-limit:] if self._retry_history else []
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        if self.circuit_config:
            self._circuit_state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self.logger.info("Circuit breaker manually reset to CLOSED state")
    
    def clear_statistics(self):
        """Clear retry statistics and history."""
        self._total_attempts = 0
        self._total_successes = 0
        self._total_failures = 0
        self._retry_history.clear()


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Decorator for automatic retry
def retry(max_attempts: int = 3,
          strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
          base_delay: float = 1.0,
          max_delay: float = 60.0,
          jitter: bool = True,
          retryable_exceptions: Optional[List[Type[Exception]]] = None,
          logger: Optional[logging.Logger] = None):
    """
    Decorator for automatic retry with configurable strategy.
    
    Args:
        max_attempts: Maximum retry attempts
        strategy: Retry strategy to use
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add jitter
        retryable_exceptions: List of exceptions that should trigger retry
        logger: Logger for retry events
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions or [RateLimitError, TimeoutError, APIError]
            )
            
            retry_manager = RetryManager(config=config, logger=logger)
            return retry_manager.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Convenience functions
def with_exponential_backoff(func: Callable,
                           max_attempts: int = 3,
                           base_delay: float = 1.0,
                           max_delay: float = 60.0) -> Any:
    """Execute function with exponential backoff retry."""
    retry_manager = RetryManager()
    return retry_manager.retry_with_backoff(
        func=func,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay
    )


def with_circuit_breaker(func: Callable,
                        failure_threshold: int = 5,
                        recovery_timeout: float = 60.0) -> Any:
    """Execute function with circuit breaker protection."""
    circuit_config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    
    retry_manager = RetryManager(circuit_config=circuit_config)
    return retry_manager.retry(func)