"""
Error Handling Toolkit

This module provides comprehensive error handling capabilities for AI applications,
including custom exceptions, retry mechanisms, and error recovery strategies.
"""

from .exception_types import (
    AIException,
    ModelError,
    ParseError,
    ConfigError,
    APIError,
    TimeoutError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    ErrorSeverity,
    ErrorCategory
)

from .error_handler import ErrorHandler
from .retry_manager import (
    RetryManager,
    RetryConfig,
    CircuitBreakerConfig,
    RetryStrategy,
    CircuitState,
    retry,
    with_exponential_backoff,
    with_circuit_breaker
)

__all__ = [
    # Exception types
    'AIException',
    'ModelError', 
    'ParseError',
    'ConfigError',
    'APIError',
    'TimeoutError',
    'RateLimitError',
    'AuthenticationError',
    'ValidationError',
    'ErrorSeverity',
    'ErrorCategory',
    
    # Error handling
    'ErrorHandler',
    'RetryManager',
    'RetryConfig',
    'CircuitBreakerConfig',
    'RetryStrategy',
    'CircuitState',
    
    # Decorators and utilities
    'retry',
    'with_exponential_backoff',
    'with_circuit_breaker',
]