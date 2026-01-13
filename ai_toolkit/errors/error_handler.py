"""
Unified error handler for AI toolkit operations.

This module provides centralized error handling capabilities including
error classification, logging, recovery strategies, and user-friendly
error reporting.
"""

import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Callable, Union, Type
from enum import Enum
from dataclasses import dataclass, field

from .exception_types import (
    AIException, ModelError, ParseError, ConfigError, APIError,
    TimeoutError, RateLimitError, AuthenticationError, ValidationError,
    ErrorSeverity, ErrorCategory
)


class ErrorAction(Enum):
    """Actions to take when handling errors."""
    IGNORE = "ignore"
    LOG = "log"
    RETRY = "retry"
    FALLBACK = "fallback"
    RAISE = "raise"
    NOTIFY = "notify"


@dataclass
class ErrorRule:
    """Rule for handling specific types of errors."""
    error_type: Type[Exception]
    action: ErrorAction
    max_retries: int = 0
    retry_delay: float = 1.0
    fallback_handler: Optional[Callable] = None
    custom_handler: Optional[Callable] = None
    log_level: int = logging.ERROR
    
    def matches(self, error: Exception) -> bool:
        """Check if this rule matches the given error."""
        return isinstance(error, self.error_type)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float = field(default_factory=time.time)
    attempt: int = 1
    max_attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_final_attempt(self) -> bool:
        """Check if this is the final attempt."""
        return self.attempt >= self.max_attempts


class ErrorHandler:
    """
    Unified error handler for AI toolkit operations.
    
    Provides centralized error handling with configurable rules,
    logging, retry mechanisms, and recovery strategies.
    """
    
    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 default_action: ErrorAction = ErrorAction.RAISE,
                 enable_stack_trace: bool = True):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance for error logging
            default_action: Default action for unhandled errors
            enable_stack_trace: Whether to include stack traces in logs
        """
        self.logger = logger or logging.getLogger(__name__)
        self.default_action = default_action
        self.enable_stack_trace = enable_stack_trace
        
        # Error handling rules
        self._rules: List[ErrorRule] = []
        
        # Error statistics
        self._error_counts: Dict[str, int] = {}
        self._last_errors: List[Dict[str, Any]] = []
        self._max_error_history = 100
        
        # Setup default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default error handling rules."""
        # Rate limit errors - retry with backoff
        self.add_rule(ErrorRule(
            error_type=RateLimitError,
            action=ErrorAction.RETRY,
            max_retries=3,
            retry_delay=5.0,
            log_level=logging.WARNING
        ))
        
        # Timeout errors - retry with shorter timeout
        self.add_rule(ErrorRule(
            error_type=TimeoutError,
            action=ErrorAction.RETRY,
            max_retries=2,
            retry_delay=2.0,
            log_level=logging.WARNING
        ))
        
        # Authentication errors - don't retry, log as error
        self.add_rule(ErrorRule(
            error_type=AuthenticationError,
            action=ErrorAction.RAISE,
            log_level=logging.ERROR
        ))
        
        # API errors - conditional retry based on status code
        self.add_rule(ErrorRule(
            error_type=APIError,
            action=ErrorAction.RETRY,
            max_retries=2,
            retry_delay=1.0,
            custom_handler=self._handle_api_error
        ))
    
    def add_rule(self, rule: ErrorRule) -> None:
        """
        Add an error handling rule.
        
        Args:
            rule: Error handling rule to add
        """
        self._rules.append(rule)
    
    def remove_rule(self, error_type: Type[Exception]) -> bool:
        """
        Remove error handling rules for a specific error type.
        
        Args:
            error_type: Type of error to remove rules for
            
        Returns:
            True if any rules were removed
        """
        original_count = len(self._rules)
        self._rules = [rule for rule in self._rules if rule.error_type != error_type]
        return len(self._rules) < original_count
    
    def handle_error(self,
                    error: Exception,
                    context: Optional[ErrorContext] = None,
                    **kwargs) -> Any:
        """
        Handle an error according to configured rules.
        
        Args:
            error: Exception to handle
            context: Error context information
            **kwargs: Additional context information
            
        Returns:
            Result of error handling (if any)
            
        Raises:
            Exception: If error should be re-raised
        """
        # Create context if not provided
        if context is None:
            context = ErrorContext(
                operation=kwargs.get('operation', 'unknown'),
                metadata=kwargs
            )
        
        # Update error statistics
        self._update_error_stats(error, context)
        
        # Find matching rule
        rule = self._find_matching_rule(error)
        
        # Handle the error according to the rule
        if rule:
            return self._apply_rule(error, rule, context)
        else:
            return self._apply_default_action(error, context)
    
    def handle_api_error(self,
                        error: Exception,
                        endpoint: Optional[str] = None,
                        status_code: Optional[int] = None,
                        response_body: Optional[str] = None) -> Any:
        """
        Handle API-specific errors.
        
        Args:
            error: API error to handle
            endpoint: API endpoint that failed
            status_code: HTTP status code
            response_body: Response body content
            
        Returns:
            Result of error handling
        """
        context = ErrorContext(
            operation="api_call",
            metadata={
                'endpoint': endpoint,
                'status_code': status_code,
                'response_body': response_body[:200] if response_body else None
            }
        )
        
        # Convert to APIError if not already
        if not isinstance(error, APIError):
            error = APIError(
                message=str(error),
                status_code=status_code,
                endpoint=endpoint,
                response_body=response_body,
                original_error=error
            )
        
        return self.handle_error(error, context)
    
    def handle_parse_error(self,
                          error: Exception,
                          parser_type: Optional[str] = None,
                          input_data: Optional[str] = None,
                          expected_format: Optional[str] = None) -> Any:
        """
        Handle parsing-specific errors.
        
        Args:
            error: Parse error to handle
            parser_type: Type of parser that failed
            input_data: Input data that failed to parse
            expected_format: Expected data format
            
        Returns:
            Result of error handling
        """
        context = ErrorContext(
            operation="parsing",
            metadata={
                'parser_type': parser_type,
                'input_data': input_data[:200] if input_data else None,
                'expected_format': expected_format
            }
        )
        
        # Convert to ParseError if not already
        if not isinstance(error, ParseError):
            error = ParseError(
                message=str(error),
                parser_type=parser_type,
                input_data=input_data,
                expected_format=expected_format,
                original_error=error
            )
        
        return self.handle_error(error, context)
    
    def handle_timeout(self,
                      error: Exception,
                      operation: Optional[str] = None,
                      timeout_duration: Optional[float] = None) -> Any:
        """
        Handle timeout-specific errors.
        
        Args:
            error: Timeout error to handle
            operation: Operation that timed out
            timeout_duration: Timeout duration in seconds
            
        Returns:
            Result of error handling
        """
        context = ErrorContext(
            operation=operation or "timeout",
            metadata={
                'timeout_duration': timeout_duration
            }
        )
        
        # Convert to TimeoutError if not already
        if not isinstance(error, TimeoutError):
            error = TimeoutError(
                message=str(error),
                operation=operation,
                timeout_duration=timeout_duration,
                original_error=error
            )
        
        return self.handle_error(error, context)
    
    def _find_matching_rule(self, error: Exception) -> Optional[ErrorRule]:
        """Find the first rule that matches the error."""
        for rule in self._rules:
            if rule.matches(error):
                return rule
        return None
    
    def _apply_rule(self, error: Exception, rule: ErrorRule, context: ErrorContext) -> Any:
        """Apply an error handling rule."""
        # Log the error
        self._log_error(error, context, rule.log_level)
        
        # Apply custom handler if provided
        if rule.custom_handler:
            try:
                return rule.custom_handler(error, context, rule)
            except Exception as handler_error:
                self.logger.error(f"Custom error handler failed: {handler_error}")
        
        # Apply the rule action
        if rule.action == ErrorAction.IGNORE:
            return None
        
        elif rule.action == ErrorAction.LOG:
            return None
        
        elif rule.action == ErrorAction.RETRY:
            if context.attempt < rule.max_retries + 1:
                # Wait before retry
                if rule.retry_delay > 0:
                    time.sleep(rule.retry_delay * (context.attempt - 1))  # Exponential backoff
                raise RetryException(f"Retry attempt {context.attempt}")
            else:
                # Max retries exceeded
                self.logger.error(f"Max retries ({rule.max_retries}) exceeded for {type(error).__name__}")
                raise error
        
        elif rule.action == ErrorAction.FALLBACK:
            if rule.fallback_handler:
                return rule.fallback_handler(error, context)
            else:
                self.logger.warning(f"No fallback handler configured for {type(error).__name__}")
                raise error
        
        elif rule.action == ErrorAction.RAISE:
            raise error
        
        elif rule.action == ErrorAction.NOTIFY:
            # Could integrate with notification systems here
            self.logger.critical(f"Critical error notification: {error}")
            raise error
        
        else:
            raise error
    
    def _apply_default_action(self, error: Exception, context: ErrorContext) -> Any:
        """Apply the default action for unhandled errors."""
        self._log_error(error, context, logging.ERROR)
        
        if self.default_action == ErrorAction.RAISE:
            raise error
        elif self.default_action == ErrorAction.LOG:
            return None
        else:
            raise error
    
    def _handle_api_error(self, error: APIError, context: ErrorContext, rule: ErrorRule) -> Any:
        """Custom handler for API errors."""
        status_code = error.status_code
        
        # Don't retry certain status codes
        if status_code in [400, 401, 403, 404]:
            self.logger.error(f"Non-retryable API error {status_code}: {error.message}")
            raise error
        
        # For 429 (rate limit), use longer delay
        if status_code == 429:
            retry_after = error.context.get('retry_after', rule.retry_delay)
            time.sleep(retry_after)
            raise RetryException(f"Rate limited, retrying after {retry_after}s")
        
        # For 5xx errors, retry with exponential backoff
        if status_code and status_code >= 500:
            delay = rule.retry_delay * (2 ** (context.attempt - 1))
            time.sleep(delay)
            raise RetryException(f"Server error, retrying after {delay}s")
        
        # Default retry behavior
        raise RetryException("API error, retrying")
    
    def _log_error(self, error: Exception, context: ErrorContext, level: int):
        """Log an error with context information."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'operation': context.operation,
            'attempt': context.attempt,
            'timestamp': context.timestamp
        }
        
        # Add AI-specific error information
        if isinstance(error, AIException):
            error_info.update({
                'error_code': error.error_code,
                'severity': error.severity.value,
                'category': error.category.value,
                'context': error.context,
                'suggestions': error.suggestions
            })
        
        # Add metadata
        if context.metadata:
            error_info['metadata'] = context.metadata
        
        # Log the error
        log_message = f"Error in {context.operation}: {error}"
        
        if self.enable_stack_trace and level >= logging.ERROR:
            self.logger.log(level, log_message, exc_info=True, extra=error_info)
        else:
            self.logger.log(level, log_message, extra=error_info)
    
    def _update_error_stats(self, error: Exception, context: ErrorContext):
        """Update error statistics."""
        error_type = type(error).__name__
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        # Add to error history
        error_record = {
            'timestamp': context.timestamp,
            'error_type': error_type,
            'message': str(error),
            'operation': context.operation,
            'attempt': context.attempt
        }
        
        self._last_errors.append(error_record)
        
        # Trim history if too long
        if len(self._last_errors) > self._max_error_history:
            self._last_errors = self._last_errors[-self._max_error_history:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        total_errors = sum(self._error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_counts': self._error_counts.copy(),
            'recent_errors': len(self._last_errors),
            'error_types': list(self._error_counts.keys()),
            'most_common_error': max(self._error_counts.items(), key=lambda x: x[1])[0] if self._error_counts else None
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records."""
        return self._last_errors[-limit:] if self._last_errors else []
    
    def clear_statistics(self):
        """Clear error statistics and history."""
        self._error_counts.clear()
        self._last_errors.clear()
    
    def create_error_report(self) -> str:
        """Create a formatted error report."""
        stats = self.get_error_statistics()
        recent_errors = self.get_recent_errors(5)
        
        report_lines = [
            "=== Error Handler Report ===",
            f"Total Errors: {stats['total_errors']}",
            f"Error Types: {len(stats['error_types'])}",
            ""
        ]
        
        if stats['error_counts']:
            report_lines.append("Error Counts:")
            for error_type, count in sorted(stats['error_counts'].items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  {error_type}: {count}")
            report_lines.append("")
        
        if recent_errors:
            report_lines.append("Recent Errors:")
            for error in recent_errors:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(error['timestamp']))
                report_lines.append(f"  [{timestamp}] {error['error_type']}: {error['message']}")
        
        return "\n".join(report_lines)


class RetryException(Exception):
    """Exception used internally to signal retry attempts."""
    pass


# Convenience functions for common error handling patterns

def with_error_handling(handler: ErrorHandler,
                       operation: str = "operation",
                       max_attempts: int = 1):
    """
    Decorator for automatic error handling.
    
    Args:
        handler: ErrorHandler instance
        operation: Name of the operation for logging
        max_attempts: Maximum number of attempts
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                max_attempts=max_attempts
            )
            
            for attempt in range(1, max_attempts + 1):
                context.attempt = attempt
                try:
                    return func(*args, **kwargs)
                except RetryException:
                    if attempt < max_attempts:
                        continue
                    else:
                        raise
                except Exception as e:
                    try:
                        return handler.handle_error(e, context)
                    except RetryException:
                        if attempt < max_attempts:
                            continue
                        else:
                            raise e
                    except Exception:
                        raise
        
        return wrapper
    return decorator


def safe_execute(func: Callable,
                handler: ErrorHandler,
                operation: str = "safe_execute",
                default_return: Any = None,
                **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        handler: ErrorHandler instance
        operation: Operation name for logging
        default_return: Default return value on error
        **kwargs: Arguments to pass to the function
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(**kwargs)
    except Exception as e:
        try:
            return handler.handle_error(e, ErrorContext(operation=operation))
        except Exception:
            return default_return