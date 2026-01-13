"""
Tests for ErrorHandler class.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch

from ai_toolkit.errors.error_handler import (
    ErrorHandler, ErrorRule, ErrorContext, ErrorAction, RetryException
)
from ai_toolkit.errors.exception_types import (
    AIException, ModelError, ParseError, APIError, RateLimitError,
    TimeoutError, AuthenticationError, ErrorSeverity
)


class TestErrorRule:
    """Test ErrorRule class."""
    
    def test_error_rule_creation(self):
        """Test creating an ErrorRule."""
        rule = ErrorRule(
            error_type=ValueError,
            action=ErrorAction.RETRY,
            max_retries=3,
            retry_delay=2.0
        )
        
        assert rule.error_type == ValueError
        assert rule.action == ErrorAction.RETRY
        assert rule.max_retries == 3
        assert rule.retry_delay == 2.0
    
    def test_rule_matches(self):
        """Test rule matching."""
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.RETRY)
        
        assert rule.matches(ValueError("test"))
        assert rule.matches(ValueError())
        assert not rule.matches(TypeError("test"))
        assert not rule.matches(Exception("test"))
    
    def test_rule_matches_inheritance(self):
        """Test rule matching with inheritance."""
        rule = ErrorRule(error_type=AIException, action=ErrorAction.RETRY)
        
        assert rule.matches(AIException("test"))
        assert rule.matches(ModelError("test"))
        assert rule.matches(ParseError("test"))
        assert not rule.matches(ValueError("test"))


class TestErrorContext:
    """Test ErrorContext class."""
    
    def test_error_context_creation(self):
        """Test creating an ErrorContext."""
        context = ErrorContext(
            operation="test_operation",
            attempt=2,
            max_attempts=3,
            metadata={"key": "value"}
        )
        
        assert context.operation == "test_operation"
        assert context.attempt == 2
        assert context.max_attempts == 3
        assert context.metadata == {"key": "value"}
        assert isinstance(context.timestamp, float)
    
    def test_is_final_attempt(self):
        """Test is_final_attempt property."""
        context = ErrorContext(operation="test", attempt=2, max_attempts=3)
        assert not context.is_final_attempt
        
        context.attempt = 3
        assert context.is_final_attempt
        
        context.attempt = 4
        assert context.is_final_attempt


class TestErrorHandler:
    """Test ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.handler = ErrorHandler(logger=self.logger)
    
    def test_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler(
            logger=self.logger,
            default_action=ErrorAction.LOG,
            enable_stack_trace=False
        )
        
        assert handler.logger == self.logger
        assert handler.default_action == ErrorAction.LOG
        assert handler.enable_stack_trace is False
    
    def test_default_rules_setup(self):
        """Test that default rules are set up."""
        # Should have default rules for common error types
        rules = self.handler._rules
        
        # Check for rate limit rule
        rate_limit_rules = [r for r in rules if r.error_type == RateLimitError]
        assert len(rate_limit_rules) > 0
        assert rate_limit_rules[0].action == ErrorAction.RETRY
        
        # Check for timeout rule
        timeout_rules = [r for r in rules if r.error_type == TimeoutError]
        assert len(timeout_rules) > 0
        assert timeout_rules[0].action == ErrorAction.RETRY
        
        # Check for auth rule
        auth_rules = [r for r in rules if r.error_type == AuthenticationError]
        assert len(auth_rules) > 0
        assert auth_rules[0].action == ErrorAction.RAISE
    
    def test_add_rule(self):
        """Test adding error handling rules."""
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        
        initial_count = len(self.handler._rules)
        self.handler.add_rule(rule)
        
        assert len(self.handler._rules) == initial_count + 1
        assert rule in self.handler._rules
    
    def test_remove_rule(self):
        """Test removing error handling rules."""
        # Add a rule first
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        # Remove it
        removed = self.handler.remove_rule(ValueError)
        assert removed is True
        
        # Check it's gone
        value_error_rules = [r for r in self.handler._rules if r.error_type == ValueError]
        assert len(value_error_rules) == 0
        
        # Try removing non-existent rule
        removed = self.handler.remove_rule(TypeError)
        assert removed is False
    
    def test_handle_error_with_log_action(self):
        """Test handling error with LOG action."""
        # Add a rule for ValueError
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        error = ValueError("test error")
        result = self.handler.handle_error(error)
        
        # Should return None and log the error
        assert result is None
        self.logger.log.assert_called()
    
    def test_handle_error_with_raise_action(self):
        """Test handling error with RAISE action."""
        # Add a rule for ValueError
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.RAISE)
        self.handler.add_rule(rule)
        
        error = ValueError("test error")
        
        with pytest.raises(ValueError):
            self.handler.handle_error(error)
    
    def test_handle_error_with_ignore_action(self):
        """Test handling error with IGNORE action."""
        # Add a rule for ValueError
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.IGNORE)
        self.handler.add_rule(rule)
        
        error = ValueError("test error")
        result = self.handler.handle_error(error)
        
        # Should return None and not log
        assert result is None
    
    def test_handle_error_with_retry_action(self):
        """Test handling error with RETRY action."""
        # Add a rule for ValueError
        rule = ErrorRule(
            error_type=ValueError, 
            action=ErrorAction.RETRY,
            max_retries=2,
            retry_delay=0.1
        )
        self.handler.add_rule(rule)
        
        error = ValueError("test error")
        context = ErrorContext(operation="test", attempt=1, max_attempts=3)
        
        # Should raise RetryException for retry
        with pytest.raises(RetryException):
            self.handler.handle_error(error, context)
    
    def test_handle_error_max_retries_exceeded(self):
        """Test handling error when max retries exceeded."""
        # Add a rule for ValueError
        rule = ErrorRule(
            error_type=ValueError,
            action=ErrorAction.RETRY,
            max_retries=2
        )
        self.handler.add_rule(rule)
        
        error = ValueError("test error")
        context = ErrorContext(operation="test", attempt=3, max_attempts=3)
        
        # Should raise original error when max retries exceeded
        with pytest.raises(ValueError):
            self.handler.handle_error(error, context)
    
    def test_handle_error_with_fallback_action(self):
        """Test handling error with FALLBACK action."""
        # Mock fallback handler
        fallback_handler = Mock(return_value="fallback_result")
        
        rule = ErrorRule(
            error_type=ValueError,
            action=ErrorAction.FALLBACK,
            fallback_handler=fallback_handler
        )
        self.handler.add_rule(rule)
        
        error = ValueError("test error")
        result = self.handler.handle_error(error)
        
        assert result == "fallback_result"
        fallback_handler.assert_called_once()
    
    def test_handle_error_with_custom_handler(self):
        """Test handling error with custom handler."""
        # Mock custom handler
        custom_handler = Mock(return_value="custom_result")
        
        rule = ErrorRule(
            error_type=ValueError,
            action=ErrorAction.LOG,
            custom_handler=custom_handler
        )
        self.handler.add_rule(rule)
        
        error = ValueError("test error")
        result = self.handler.handle_error(error)
        
        assert result == "custom_result"
        custom_handler.assert_called_once()
    
    def test_handle_api_error(self):
        """Test handling API-specific errors."""
        error = Exception("API failed")
        
        # Should handle the error (may raise RetryException for retry)
        try:
            result = self.handler.handle_api_error(
                error=error,
                endpoint="/api/test",
                status_code=500,
                response_body='{"error": "server error"}'
            )
        except Exception:
            pass  # Expected for retry scenarios
        
        # Should have logged the error
        self.logger.log.assert_called()
    
    def test_handle_parse_error(self):
        """Test handling parse-specific errors."""
        error = Exception("Parse failed")
        
        # Should handle the error (may raise ParseError for non-retryable)
        try:
            result = self.handler.handle_parse_error(
                error=error,
                parser_type="json",
                input_data='{"invalid": json}',
                expected_format="valid JSON"
            )
        except ParseError:
            pass  # Expected for non-retryable errors
        
        # Should have logged the error
        self.logger.log.assert_called()
    
    def test_handle_timeout(self):
        """Test handling timeout-specific errors."""
        error = Exception("Timeout")
        
        # Should handle the error (may raise RetryException for retry)
        try:
            result = self.handler.handle_timeout(
                error=error,
                operation="api_call",
                timeout_duration=30.0
            )
        except Exception:
            pass  # Expected for retry scenarios
        
        # Should have logged the error
        self.logger.log.assert_called()
    
    def test_error_statistics(self):
        """Test error statistics tracking."""
        # Handle some errors (catch exceptions as expected)
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        try:
            self.handler.handle_error(ValueError("error 1"))
        except:
            pass
        try:
            self.handler.handle_error(ValueError("error 2"))
        except:
            pass
        try:
            self.handler.handle_error(TypeError("error 3"))
        except:
            pass
        
        stats = self.handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["error_counts"]["ValueError"] == 2
        assert stats["error_counts"]["TypeError"] == 1
        assert stats["most_common_error"] == "ValueError"
    
    def test_recent_errors(self):
        """Test recent errors tracking."""
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        self.handler.handle_error(ValueError("recent error"))
        
        recent = self.handler.get_recent_errors(limit=1)
        
        assert len(recent) == 1
        assert recent[0]["error_type"] == "ValueError"
        assert recent[0]["message"] == "recent error"
    
    def test_clear_statistics(self):
        """Test clearing error statistics."""
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        self.handler.handle_error(ValueError("test"))
        
        # Check stats exist
        stats = self.handler.get_error_statistics()
        assert stats["total_errors"] > 0
        
        # Clear and check
        self.handler.clear_statistics()
        stats = self.handler.get_error_statistics()
        assert stats["total_errors"] == 0
    
    def test_create_error_report(self):
        """Test creating error report."""
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        self.handler.handle_error(ValueError("test error"))
        
        report = self.handler.create_error_report()
        
        assert "Error Handler Report" in report
        assert "Total Errors: 1" in report
        assert "ValueError: 1" in report
        assert "test error" in report
    
    def test_ai_exception_logging(self):
        """Test logging of AI-specific exceptions."""
        rule = ErrorRule(error_type=ModelError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        error = ModelError(
            message="Model failed",
            model_name="gpt-4",
            severity=ErrorSeverity.HIGH
        )
        
        self.handler.handle_error(error)
        
        # Check that AI-specific information was logged
        self.logger.log.assert_called()
        call_args = self.logger.log.call_args
        extra_info = call_args[1].get('extra', {})
        
        assert extra_info.get('error_code') == 'ModelError'
        assert extra_info.get('severity') == 'high'
        assert extra_info.get('category') == 'model'
    
    def test_custom_api_error_handler(self):
        """Test custom API error handler."""
        # Create API error with 429 status
        error = APIError("Rate limited", status_code=429)
        
        with patch('time.sleep') as mock_sleep:
            with pytest.raises(RetryException):
                self.handler._handle_api_error(error, ErrorContext("test"), Mock())
            
            # Should have slept for rate limit
            mock_sleep.assert_called()
    
    def test_non_retryable_api_errors(self):
        """Test non-retryable API errors."""
        # 401 should not be retried
        error = APIError("Unauthorized", status_code=401)
        
        with pytest.raises(APIError):
            self.handler._handle_api_error(error, ErrorContext("test"), Mock())
    
    def test_default_action_fallback(self):
        """Test default action when no rule matches."""
        # No rule for TypeError
        error = TypeError("unhandled error")
        
        with pytest.raises(TypeError):
            self.handler.handle_error(error)
        
        # Should have logged the error
        self.logger.log.assert_called()
    
    def test_error_context_auto_creation(self):
        """Test automatic error context creation."""
        rule = ErrorRule(error_type=ValueError, action=ErrorAction.LOG)
        self.handler.add_rule(rule)
        
        error = ValueError("test")
        result = self.handler.handle_error(error, operation="test_op")
        
        # Should work without explicit context
        assert result is None
        self.logger.log.assert_called()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        from ai_toolkit.errors.error_handler import safe_execute
        
        def success_func(x, y):
            return x + y
        
        handler = ErrorHandler()
        result = safe_execute(
            func=success_func,
            handler=handler,
            operation="add",
            x=1,
            y=2
        )
        
        assert result == 3
    
    def test_safe_execute_with_error(self):
        """Test safe_execute with error."""
        from ai_toolkit.errors.error_handler import safe_execute
        
        def failing_func():
            raise ValueError("test error")
        
        handler = ErrorHandler()
        result = safe_execute(
            func=failing_func,
            handler=handler,
            operation="fail",
            default_return="default"
        )
        
        assert result == "default"
    
    def test_with_error_handling_decorator(self):
        """Test with_error_handling decorator."""
        from ai_toolkit.errors.error_handler import with_error_handling
        
        handler = ErrorHandler()
        
        @with_error_handling(handler, operation="test_op", max_attempts=2)
        def test_func(should_fail=False):
            if should_fail:
                raise ValueError("test error")
            return "success"
        
        # Test success
        result = test_func(should_fail=False)
        assert result == "success"
        
        # Test failure
        with pytest.raises(ValueError):
            test_func(should_fail=True)