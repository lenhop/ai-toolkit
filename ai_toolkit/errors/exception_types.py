"""
Custom exception types for AI toolkit operations.

This module defines a hierarchy of exceptions specific to AI operations,
providing detailed error information and context for better debugging.
"""

from typing import Any, Dict, Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    MODEL = "model"
    PARSING = "parsing"
    CONFIG = "config"
    API = "api"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class AIException(Exception):
    """
    Base exception class for all AI toolkit errors.
    
    Provides structured error information including severity,
    category, context, and suggestions for resolution.
    """
    
    def __init__(self,
                 message: str,
                 error_code: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 context: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None,
                 original_error: Optional[Exception] = None):
        """
        Initialize AI exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            severity: Error severity level
            category: Error category for classification
            context: Additional context information
            suggestions: List of suggested solutions
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.suggestions = suggestions or []
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context,
            'suggestions': self.suggestions,
            'original_error': str(self.original_error) if self.original_error else None
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        parts = [f"{self.__class__.__name__}: {self.message}"]
        
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        
        if self.severity != ErrorSeverity.MEDIUM:
            parts.append(f"Severity: {self.severity.value}")
        
        if self.context:
            parts.append(f"Context: {self.context}")
        
        if self.suggestions:
            parts.append(f"Suggestions: {', '.join(self.suggestions)}")
        
        return " | ".join(parts)


class ModelError(AIException):
    """Exception raised for model-related errors."""
    
    def __init__(self,
                 message: str,
                 model_name: Optional[str] = None,
                 provider: Optional[str] = None,
                 **kwargs):
        """
        Initialize model error.
        
        Args:
            message: Error message
            model_name: Name of the model that caused the error
            provider: Model provider (e.g., 'openai', 'anthropic')
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        if provider:
            context['provider'] = provider
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.MODEL
        
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.provider = provider


class ParseError(AIException):
    """Exception raised for parsing-related errors."""
    
    def __init__(self,
                 message: str,
                 parser_type: Optional[str] = None,
                 input_data: Optional[str] = None,
                 expected_format: Optional[str] = None,
                 **kwargs):
        """
        Initialize parse error.
        
        Args:
            message: Error message
            parser_type: Type of parser that failed
            input_data: Input data that failed to parse (truncated for safety)
            expected_format: Expected data format
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if parser_type:
            context['parser_type'] = parser_type
        if input_data:
            # Truncate input data for safety
            context['input_data'] = input_data[:200] + "..." if len(input_data) > 200 else input_data
        if expected_format:
            context['expected_format'] = expected_format
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.PARSING
        
        super().__init__(message, **kwargs)
        self.parser_type = parser_type
        self.input_data = input_data
        self.expected_format = expected_format


class ConfigError(AIException):
    """Exception raised for configuration-related errors."""
    
    def __init__(self,
                 message: str,
                 config_key: Optional[str] = None,
                 config_file: Optional[str] = None,
                 **kwargs):
        """
        Initialize config error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_file: Configuration file path
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_file:
            context['config_file'] = config_file
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.CONFIG
        
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_file = config_file


class APIError(AIException):
    """Exception raised for API-related errors."""
    
    def __init__(self,
                 message: str,
                 status_code: Optional[int] = None,
                 response_body: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 **kwargs):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            response_body: API response body (truncated for safety)
            endpoint: API endpoint that failed
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if status_code:
            context['status_code'] = status_code
        if response_body:
            # Truncate response body for safety
            context['response_body'] = response_body[:500] + "..." if len(response_body) > 500 else response_body
        if endpoint:
            context['endpoint'] = endpoint
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.API
        
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_body = response_body
        self.endpoint = endpoint


class TimeoutError(AIException):
    """Exception raised for timeout-related errors."""
    
    def __init__(self,
                 message: str,
                 timeout_duration: Optional[float] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        """
        Initialize timeout error.
        
        Args:
            message: Error message
            timeout_duration: Timeout duration in seconds
            operation: Operation that timed out
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if timeout_duration:
            context['timeout_duration'] = timeout_duration
        if operation:
            context['operation'] = operation
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.TIMEOUT
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.operation = operation


class RateLimitError(AIException):
    """Exception raised for rate limiting errors."""
    
    def __init__(self,
                 message: str,
                 retry_after: Optional[int] = None,
                 limit_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            limit_type: Type of rate limit (e.g., 'requests', 'tokens')
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if retry_after:
            context['retry_after'] = retry_after
        if limit_type:
            context['limit_type'] = limit_type
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.RATE_LIMIT
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        
        # Add default suggestions for rate limiting
        suggestions = kwargs.get('suggestions', [])
        if retry_after:
            suggestions.append(f"Wait {retry_after} seconds before retrying")
        suggestions.extend([
            "Implement exponential backoff",
            "Reduce request frequency",
            "Consider upgrading API plan"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.limit_type = limit_type


class AuthenticationError(AIException):
    """Exception raised for authentication-related errors."""
    
    def __init__(self,
                 message: str,
                 auth_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize authentication error.
        
        Args:
            message: Error message
            auth_type: Type of authentication (e.g., 'api_key', 'oauth')
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if auth_type:
            context['auth_type'] = auth_type
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.AUTHENTICATION
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        
        # Add default suggestions for authentication errors
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check API key validity",
            "Verify authentication credentials",
            "Ensure proper permissions",
            "Check for expired tokens"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)
        self.auth_type = auth_type


class ValidationError(AIException):
    """Exception raised for validation-related errors."""
    
    def __init__(self,
                 message: str,
                 field_name: Optional[str] = None,
                 field_value: Optional[Any] = None,
                 validation_rule: Optional[str] = None,
                 **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            validation_rule: Validation rule that was violated
            **kwargs: Additional arguments passed to AIException
        """
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if field_value is not None:
            # Convert to string and truncate for safety
            value_str = str(field_value)
            context['field_value'] = value_str[:100] + "..." if len(value_str) > 100 else value_str
        if validation_rule:
            context['validation_rule'] = validation_rule
        
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.VALIDATION
        
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


# Convenience functions for creating common exceptions

def create_model_error(message: str, model_name: str = None, provider: str = None) -> ModelError:
    """Create a model error with common suggestions."""
    suggestions = [
        "Check model name and availability",
        "Verify model configuration",
        "Ensure sufficient API credits",
        "Try a different model"
    ]
    return ModelError(
        message=message,
        model_name=model_name,
        provider=provider,
        suggestions=suggestions
    )


def create_parse_error(message: str, parser_type: str = None, input_data: str = None) -> ParseError:
    """Create a parse error with common suggestions."""
    suggestions = [
        "Check input data format",
        "Verify parser configuration",
        "Try a different parser type",
        "Clean or preprocess input data"
    ]
    return ParseError(
        message=message,
        parser_type=parser_type,
        input_data=input_data,
        suggestions=suggestions
    )


def create_api_error(message: str, status_code: int = None, endpoint: str = None) -> APIError:
    """Create an API error with status-specific suggestions."""
    suggestions = []
    
    if status_code == 401:
        suggestions.extend([
            "Check API key validity",
            "Verify authentication headers",
            "Ensure proper permissions"
        ])
    elif status_code == 403:
        suggestions.extend([
            "Check API permissions",
            "Verify account access level",
            "Contact API provider support"
        ])
    elif status_code == 429:
        suggestions.extend([
            "Implement rate limiting",
            "Add exponential backoff",
            "Reduce request frequency"
        ])
    elif status_code and status_code >= 500:
        suggestions.extend([
            "Retry the request",
            "Check API service status",
            "Contact API provider support"
        ])
    else:
        suggestions.extend([
            "Check request parameters",
            "Verify API endpoint",
            "Review API documentation"
        ])
    
    return APIError(
        message=message,
        status_code=status_code,
        endpoint=endpoint,
        suggestions=suggestions
    )