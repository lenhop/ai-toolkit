"""
Tests for custom exception types.
"""

import pytest
from ai_toolkit.errors.exception_types import (
    AIException, ModelError, ParseError, ConfigError, APIError,
    TimeoutError, RateLimitError, AuthenticationError, ValidationError,
    ErrorSeverity, ErrorCategory,
    create_model_error, create_parse_error, create_api_error
)


class TestErrorEnums:
    """Test error enums."""
    
    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_category_values(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.MODEL.value == "model"
        assert ErrorCategory.PARSING.value == "parsing"
        assert ErrorCategory.CONFIG.value == "config"
        assert ErrorCategory.API.value == "api"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"


class TestAIException:
    """Test AIException base class."""
    
    def test_basic_creation(self):
        """Test basic AIException creation."""
        error = AIException("Test error message")
        
        assert "Test error message" in str(error)
        assert error.message == "Test error message"
        assert error.error_code == "AIException"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.UNKNOWN
        assert error.context == {}
        assert error.suggestions == []
        assert error.original_error is None
    
    def test_full_creation(self):
        """Test AIException with all parameters."""
        original_error = ValueError("Original error")
        context = {"key": "value"}
        suggestions = ["Try this", "Try that"]
        
        error = AIException(
            message="Custom error",
            error_code="CUSTOM_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.API,
            context=context,
            suggestions=suggestions,
            original_error=original_error
        )
        
        assert error.message == "Custom error"
        assert error.error_code == "CUSTOM_001"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.API
        assert error.context == context
        assert error.suggestions == suggestions
        assert error.original_error == original_error
    
    def test_to_dict(self):
        """Test converting exception to dictionary."""
        error = AIException(
            message="Test error",
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            context={"test": "value"},
            suggestions=["Fix it"]
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "AIException"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["severity"] == "high"
        assert error_dict["category"] == "unknown"
        assert error_dict["context"] == {"test": "value"}
        assert error_dict["suggestions"] == ["Fix it"]
        assert error_dict["original_error"] is None
    
    def test_string_representation(self):
        """Test string representation of exception."""
        error = AIException(
            message="Test error",
            error_code="TEST_001",
            severity=ErrorSeverity.HIGH,
            context={"key": "value"},
            suggestions=["Fix it", "Try again"]
        )
        
        error_str = str(error)
        assert "AIException: Test error" in error_str
        assert "Code: TEST_001" in error_str
        assert "Severity: high" in error_str
        assert "Context: {'key': 'value'}" in error_str
        assert "Suggestions: Fix it, Try again" in error_str


class TestModelError:
    """Test ModelError exception."""
    
    def test_basic_creation(self):
        """Test basic ModelError creation."""
        error = ModelError("Model failed")
        
        assert error.message == "Model failed"
        assert error.category == ErrorCategory.MODEL
        assert error.model_name is None
        assert error.provider is None
    
    def test_with_model_info(self):
        """Test ModelError with model information."""
        error = ModelError(
            message="Model timeout",
            model_name="gpt-4",
            provider="openai"
        )
        
        assert error.model_name == "gpt-4"
        assert error.provider == "openai"
        assert error.context["model_name"] == "gpt-4"
        assert error.context["provider"] == "openai"
    
    def test_inheritance(self):
        """Test ModelError inheritance."""
        error = ModelError("Test")
        assert isinstance(error, AIException)
        assert isinstance(error, Exception)


class TestParseError:
    """Test ParseError exception."""
    
    def test_basic_creation(self):
        """Test basic ParseError creation."""
        error = ParseError("Parse failed")
        
        assert error.message == "Parse failed"
        assert error.category == ErrorCategory.PARSING
        assert error.parser_type is None
        assert error.input_data is None
        assert error.expected_format is None
    
    def test_with_parse_info(self):
        """Test ParseError with parsing information."""
        error = ParseError(
            message="JSON parse failed",
            parser_type="json",
            input_data='{"invalid": json}',
            expected_format="valid JSON"
        )
        
        assert error.parser_type == "json"
        assert error.input_data == '{"invalid": json}'
        assert error.expected_format == "valid JSON"
        assert error.context["parser_type"] == "json"
        assert error.context["input_data"] == '{"invalid": json}'
        assert error.context["expected_format"] == "valid JSON"
    
    def test_input_data_truncation(self):
        """Test input data truncation for long strings."""
        long_input = "x" * 300
        error = ParseError(
            message="Parse failed",
            input_data=long_input
        )
        
        # Should be truncated to 200 chars + "..."
        assert len(error.context["input_data"]) == 203
        assert error.context["input_data"].endswith("...")


class TestConfigError:
    """Test ConfigError exception."""
    
    def test_basic_creation(self):
        """Test basic ConfigError creation."""
        error = ConfigError("Config invalid")
        
        assert error.message == "Config invalid"
        assert error.category == ErrorCategory.CONFIG
        assert error.config_key is None
        assert error.config_file is None
    
    def test_with_config_info(self):
        """Test ConfigError with configuration information."""
        error = ConfigError(
            message="Missing API key",
            config_key="api_key",
            config_file="config.yaml"
        )
        
        assert error.config_key == "api_key"
        assert error.config_file == "config.yaml"
        assert error.context["config_key"] == "api_key"
        assert error.context["config_file"] == "config.yaml"


class TestAPIError:
    """Test APIError exception."""
    
    def test_basic_creation(self):
        """Test basic APIError creation."""
        error = APIError("API failed")
        
        assert error.message == "API failed"
        assert error.category == ErrorCategory.API
        assert error.status_code is None
        assert error.response_body is None
        assert error.endpoint is None
    
    def test_with_api_info(self):
        """Test APIError with API information."""
        error = APIError(
            message="Not found",
            status_code=404,
            response_body='{"error": "Not found"}',
            endpoint="/api/users"
        )
        
        assert error.status_code == 404
        assert error.response_body == '{"error": "Not found"}'
        assert error.endpoint == "/api/users"
        assert error.context["status_code"] == 404
        assert error.context["response_body"] == '{"error": "Not found"}'
        assert error.context["endpoint"] == "/api/users"
    
    def test_response_body_truncation(self):
        """Test response body truncation for long responses."""
        long_response = "x" * 600
        error = APIError(
            message="API failed",
            response_body=long_response
        )
        
        # Should be truncated to 500 chars + "..."
        assert len(error.context["response_body"]) == 503
        assert error.context["response_body"].endswith("...")


class TestTimeoutError:
    """Test TimeoutError exception."""
    
    def test_basic_creation(self):
        """Test basic TimeoutError creation."""
        error = TimeoutError("Operation timed out")
        
        assert error.message == "Operation timed out"
        assert error.category == ErrorCategory.TIMEOUT
        assert error.severity == ErrorSeverity.HIGH
        assert error.timeout_duration is None
        assert error.operation is None
    
    def test_with_timeout_info(self):
        """Test TimeoutError with timeout information."""
        error = TimeoutError(
            message="API call timed out",
            timeout_duration=30.0,
            operation="api_call"
        )
        
        assert error.timeout_duration == 30.0
        assert error.operation == "api_call"
        assert error.context["timeout_duration"] == 30.0
        assert error.context["operation"] == "api_call"


class TestRateLimitError:
    """Test RateLimitError exception."""
    
    def test_basic_creation(self):
        """Test basic RateLimitError creation."""
        error = RateLimitError("Rate limited")
        
        assert error.message == "Rate limited"
        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retry_after is None
        assert error.limit_type is None
    
    def test_with_rate_limit_info(self):
        """Test RateLimitError with rate limit information."""
        error = RateLimitError(
            message="Too many requests",
            retry_after=60,
            limit_type="requests"
        )
        
        assert error.retry_after == 60
        assert error.limit_type == "requests"
        assert error.context["retry_after"] == 60
        assert error.context["limit_type"] == "requests"
    
    def test_default_suggestions(self):
        """Test default suggestions for rate limit errors."""
        error = RateLimitError(
            message="Rate limited",
            retry_after=30
        )
        
        suggestions = error.suggestions
        assert "Wait 30 seconds before retrying" in suggestions
        assert "Implement exponential backoff" in suggestions
        assert "Reduce request frequency" in suggestions
        assert "Consider upgrading API plan" in suggestions


class TestAuthenticationError:
    """Test AuthenticationError exception."""
    
    def test_basic_creation(self):
        """Test basic AuthenticationError creation."""
        error = AuthenticationError("Authentication failed")
        
        assert error.message == "Authentication failed"
        assert error.category == ErrorCategory.AUTHENTICATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.auth_type is None
    
    def test_with_auth_info(self):
        """Test AuthenticationError with authentication information."""
        error = AuthenticationError(
            message="Invalid API key",
            auth_type="api_key"
        )
        
        assert error.auth_type == "api_key"
        assert error.context["auth_type"] == "api_key"
    
    def test_default_suggestions(self):
        """Test default suggestions for authentication errors."""
        error = AuthenticationError("Auth failed")
        
        suggestions = error.suggestions
        assert "Check API key validity" in suggestions
        assert "Verify authentication credentials" in suggestions
        assert "Ensure proper permissions" in suggestions
        assert "Check for expired tokens" in suggestions


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_basic_creation(self):
        """Test basic ValidationError creation."""
        error = ValidationError("Validation failed")
        
        assert error.message == "Validation failed"
        assert error.category == ErrorCategory.VALIDATION
        assert error.field_name is None
        assert error.field_value is None
        assert error.validation_rule is None
    
    def test_with_validation_info(self):
        """Test ValidationError with validation information."""
        error = ValidationError(
            message="Invalid email",
            field_name="email",
            field_value="invalid-email",
            validation_rule="email_format"
        )
        
        assert error.field_name == "email"
        assert error.field_value == "invalid-email"
        assert error.validation_rule == "email_format"
        assert error.context["field_name"] == "email"
        assert error.context["field_value"] == "invalid-email"
        assert error.context["validation_rule"] == "email_format"
    
    def test_field_value_truncation(self):
        """Test field value truncation for long values."""
        long_value = "x" * 150
        error = ValidationError(
            message="Value too long",
            field_value=long_value
        )
        
        # Should be truncated to 100 chars + "..."
        assert len(error.context["field_value"]) == 103
        assert error.context["field_value"].endswith("...")


class TestConvenienceFunctions:
    """Test convenience functions for creating exceptions."""
    
    def test_create_model_error(self):
        """Test create_model_error function."""
        error = create_model_error(
            message="Model failed",
            model_name="gpt-4",
            provider="openai"
        )
        
        assert isinstance(error, ModelError)
        assert error.message == "Model failed"
        assert error.model_name == "gpt-4"
        assert error.provider == "openai"
        assert len(error.suggestions) > 0
        assert "Check model name and availability" in error.suggestions
    
    def test_create_parse_error(self):
        """Test create_parse_error function."""
        error = create_parse_error(
            message="Parse failed",
            parser_type="json",
            input_data='{"invalid"}'
        )
        
        assert isinstance(error, ParseError)
        assert error.message == "Parse failed"
        assert error.parser_type == "json"
        assert error.input_data == '{"invalid"}'
        assert len(error.suggestions) > 0
        assert "Check input data format" in error.suggestions
    
    def test_create_api_error_401(self):
        """Test create_api_error with 401 status."""
        error = create_api_error(
            message="Unauthorized",
            status_code=401,
            endpoint="/api/users"
        )
        
        assert isinstance(error, APIError)
        assert error.message == "Unauthorized"
        assert error.status_code == 401
        assert error.endpoint == "/api/users"
        assert "Check API key validity" in error.suggestions
    
    def test_create_api_error_429(self):
        """Test create_api_error with 429 status."""
        error = create_api_error(
            message="Rate limited",
            status_code=429
        )
        
        assert isinstance(error, APIError)
        assert error.status_code == 429
        assert "Implement rate limiting" in error.suggestions
        assert "Add exponential backoff" in error.suggestions
    
    def test_create_api_error_500(self):
        """Test create_api_error with 500 status."""
        error = create_api_error(
            message="Server error",
            status_code=500
        )
        
        assert isinstance(error, APIError)
        assert error.status_code == 500
        assert "Retry the request" in error.suggestions
        assert "Check API service status" in error.suggestions