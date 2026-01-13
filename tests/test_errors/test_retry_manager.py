"""
Tests for RetryManager class.
"""

import pytest
import time
from unittest.mock import Mock, patch

from ai_toolkit.errors.retry_manager import (
    RetryManager, RetryConfig, CircuitBreakerConfig, RetryStrategy,
    CircuitState, RetryAttempt, CircuitBreakerOpenError,
    retry, with_exponential_backoff, with_circuit_breaker
)
from ai_toolkit.errors.exception_types import (
    RateLimitError, TimeoutError, APIError, AuthenticationError
)


class TestRetryConfig:
    """Test RetryConfig class."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.jitter_range == 0.1
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=2.0,
            max_delay=30.0,
            jitter=False
        )
        
        assert config.max_attempts == 5
        assert config.strategy == RetryStrategy.FIXED_DELAY
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.jitter is False


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig class."""
    
    def test_default_config(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3


class TestRetryAttempt:
    """Test RetryAttempt class."""
    
    def test_retry_attempt_creation(self):
        """Test creating a RetryAttempt."""
        exception = ValueError("test")
        attempt = RetryAttempt(
            attempt_number=2,
            delay=1.5,
            exception=exception,
            success=False
        )
        
        assert attempt.attempt_number == 2
        assert attempt.delay == 1.5
        assert attempt.exception == exception
        assert attempt.success is False
        assert isinstance(attempt.timestamp, float)


class TestRetryManager:
    """Test RetryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.retry_manager = RetryManager()
    
    def test_initialization(self):
        """Test RetryManager initialization."""
        config = RetryConfig(max_attempts=5)
        circuit_config = CircuitBreakerConfig(failure_threshold=3)
        
        manager = RetryManager(config=config, circuit_config=circuit_config)
        
        assert manager.config == config
        assert manager.circuit_config == circuit_config
        assert manager._circuit_state == CircuitState.CLOSED
    
    def test_successful_execution(self):
        """Test successful function execution."""
        def success_func():
            return "success"
        
        result = self.retry_manager.retry(success_func)
        assert result == "success"
        
        # Check statistics
        stats = self.retry_manager.get_statistics()
        assert stats["total_attempts"] == 1
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 0
    
    def test_retry_on_retryable_exception(self):
        """Test retry on retryable exceptions."""
        call_count = 0
        
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = self.retry_manager.retry(failing_then_success)
        
        assert result == "success"
        assert call_count == 3
        
        # Check statistics
        stats = self.retry_manager.get_statistics()
        assert stats["total_attempts"] == 3
        assert stats["total_successes"] == 1
    
    def test_no_retry_on_non_retryable_exception(self):
        """Test no retry on non-retryable exceptions."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Auth failed")
        
        with pytest.raises(AuthenticationError):
            self.retry_manager.retry(failing_func)
        
        assert call_count == 1  # Should not retry
    
    def test_max_attempts_exceeded(self):
        """Test behavior when max attempts exceeded."""
        def always_failing():
            raise RateLimitError("Always fails")
        
        config = RetryConfig(max_attempts=2)
        manager = RetryManager(config=config)
        
        with patch('time.sleep'):
            with pytest.raises(RateLimitError):
                manager.retry(always_failing)
        
        # Check statistics
        stats = manager.get_statistics()
        assert stats["total_attempts"] == 2
        assert stats["total_failures"] == 1
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        # Test delay calculation
        delay1 = manager._calculate_delay(1, config)
        delay2 = manager._calculate_delay(2, config)
        delay3 = manager._calculate_delay(3, config)
        
        assert delay1 == 1.0  # 1.0 * 2^0
        assert delay2 == 2.0  # 1.0 * 2^1
        assert delay3 == 4.0  # 1.0 * 2^2
    
    def test_fixed_delay(self):
        """Test fixed delay strategy."""
        config = RetryConfig(
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=2.0,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        delay1 = manager._calculate_delay(1, config)
        delay2 = manager._calculate_delay(2, config)
        delay3 = manager._calculate_delay(3, config)
        
        assert delay1 == 2.0
        assert delay2 == 2.0
        assert delay3 == 2.0
    
    def test_linear_backoff_delay(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=1.0,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        delay1 = manager._calculate_delay(1, config)
        delay2 = manager._calculate_delay(2, config)
        delay3 = manager._calculate_delay(3, config)
        
        assert delay1 == 1.0  # 1.0 * 1
        assert delay2 == 2.0  # 1.0 * 2
        assert delay3 == 3.0  # 1.0 * 3
    
    def test_fibonacci_backoff_delay(self):
        """Test Fibonacci backoff delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.FIBONACCI_BACKOFF,
            base_delay=1.0,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        delay1 = manager._calculate_delay(1, config)
        delay2 = manager._calculate_delay(2, config)
        delay3 = manager._calculate_delay(3, config)
        delay4 = manager._calculate_delay(4, config)
        
        assert delay1 == 1.0  # 1.0 * fib(1) = 1.0 * 1
        assert delay2 == 1.0  # 1.0 * fib(2) = 1.0 * 1
        assert delay3 == 2.0  # 1.0 * fib(3) = 1.0 * 2
        assert delay4 == 3.0  # 1.0 * fib(4) = 1.0 * 3
    
    def test_max_delay_limit(self):
        """Test maximum delay limit."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=10.0,
            max_delay=15.0,
            exponential_base=2.0,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        delay3 = manager._calculate_delay(3, config)  # Would be 40.0 without limit
        assert delay3 == 15.0
    
    def test_jitter_application(self):
        """Test jitter application to delays."""
        config = RetryConfig(
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=10.0,
            jitter=True,
            jitter_range=0.1
        )
        manager = RetryManager(config=config)
        
        # Generate multiple delays to test jitter
        delays = [manager._calculate_delay(1, config) for _ in range(10)]
        
        # All delays should be different due to jitter
        assert len(set(delays)) > 1
        
        # All delays should be within jitter range
        for delay in delays:
            assert 9.0 <= delay <= 11.0  # 10.0 ± 10%
    
    def test_custom_retry_condition(self):
        """Test custom retry condition."""
        def custom_condition(exception, attempt):
            # Only retry ValueError, and only once
            return isinstance(exception, ValueError) and attempt < 2
        
        config = RetryConfig(
            max_attempts=3,
            retry_condition=custom_condition
        )
        manager = RetryManager(config=config)
        
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Custom retry test")
        
        with patch('time.sleep'):
            with pytest.raises(ValueError):
                manager.retry(failing_func)
        
        assert call_count == 2  # Initial + 1 retry
    
    def test_custom_delay_calculator(self):
        """Test custom delay calculator."""
        def custom_delay(attempt, base_delay):
            return base_delay * attempt * 2  # Custom formula
        
        config = RetryConfig(
            base_delay=1.0,
            delay_calculator=custom_delay,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        delay1 = manager._calculate_delay(1, config)
        delay2 = manager._calculate_delay(2, config)
        
        assert delay1 == 2.0  # 1.0 * 1 * 2
        assert delay2 == 4.0  # 1.0 * 2 * 2
    
    def test_should_retry_method(self):
        """Test should_retry method."""
        # Should retry retryable exceptions
        assert self.retry_manager.should_retry(RateLimitError("test"))
        assert self.retry_manager.should_retry(TimeoutError("test"))
        assert self.retry_manager.should_retry(APIError("test"))
        
        # Should not retry non-retryable exceptions
        assert not self.retry_manager.should_retry(AuthenticationError("test"))
        assert not self.retry_manager.should_retry(ValueError("test"))
    
    def test_retry_with_backoff_convenience(self):
        """Test retry_with_backoff convenience method."""
        call_count = 0
        
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("Rate limited")
            return "success"
        
        with patch('time.sleep'):
            result = self.retry_manager.retry_with_backoff(
                func=failing_then_success,
                max_attempts=3,
                base_delay=0.5
            )
        
        assert result == "success"
        assert call_count == 2
    
    def test_retry_with_condition_convenience(self):
        """Test retry_with_condition convenience method."""
        def condition(exception, attempt):
            return isinstance(exception, ValueError)
        
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("test")
        
        with patch('time.sleep'):
            with pytest.raises(ValueError):
                self.retry_manager.retry_with_condition(
                    func=failing_func,
                    condition=condition,
                    max_attempts=2
                )
        
        assert call_count == 2


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        circuit_config = CircuitBreakerConfig(failure_threshold=2)
        manager = RetryManager(circuit_config=circuit_config)
        
        def success_func():
            return "success"
        
        # Should execute normally in closed state
        result = manager.retry(success_func)
        assert result == "success"
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        circuit_config = CircuitBreakerConfig(failure_threshold=2)
        config = RetryConfig(max_attempts=1)  # No retries
        manager = RetryManager(config=config, circuit_config=circuit_config)
        
        def failing_func():
            raise Exception("Always fails")
        
        # First failure
        with pytest.raises(Exception):
            manager.retry(failing_func)
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            manager.retry(failing_func)
        
        # Third attempt - should be blocked by open circuit
        with pytest.raises(CircuitBreakerOpenError):
            manager.retry(failing_func)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,  # Short timeout for testing
            success_threshold=1
        )
        config = RetryConfig(max_attempts=1)
        manager = RetryManager(config=config, circuit_config=circuit_config)
        
        def failing_func():
            raise Exception("Fails")
        
        def success_func():
            return "success"
        
        # Cause circuit to open
        with pytest.raises(Exception):
            manager.retry(failing_func)
        
        # Should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            manager.retry(success_func)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should now succeed and close circuit
        result = manager.retry(success_func)
        assert result == "success"
        
        # Should continue to work
        result = manager.retry(success_func)
        assert result == "success"
    
    def test_reset_circuit_breaker(self):
        """Test manually resetting circuit breaker."""
        circuit_config = CircuitBreakerConfig(failure_threshold=1)
        config = RetryConfig(max_attempts=1)
        manager = RetryManager(config=config, circuit_config=circuit_config)
        
        def failing_func():
            raise Exception("Fails")
        
        # Open the circuit
        with pytest.raises(Exception):
            manager.retry(failing_func)
        
        # Should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            manager.retry(failing_func)
        
        # Reset circuit
        manager.reset_circuit_breaker()
        
        # Should work again (but still fail)
        with pytest.raises(Exception):
            manager.retry(failing_func)


class TestRetryDecorator:
    """Test retry decorator."""
    
    def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        @retry(max_attempts=3)
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
    
    def test_retry_decorator_with_retries(self):
        """Test retry decorator with retries."""
        call_count = 0
        
        @retry(max_attempts=3, base_delay=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"
        
        with patch('time.sleep'):
            result = failing_then_success()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_decorator_max_attempts(self):
        """Test retry decorator respects max attempts."""
        @retry(max_attempts=2, base_delay=0.01)
        def always_failing():
            raise RateLimitError("Always fails")
        
        with patch('time.sleep'):
            with pytest.raises(RateLimitError):
                always_failing()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_with_exponential_backoff(self):
        """Test with_exponential_backoff function."""
        call_count = 0
        
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("Rate limited")
            return "success"
        
        with patch('time.sleep'):
            result = with_exponential_backoff(
                func=failing_then_success,
                max_attempts=3
            )
        
        assert result == "success"
        assert call_count == 2
    
    def test_with_circuit_breaker(self):
        """Test with_circuit_breaker function."""
        def success_func():
            return "success"
        
        result = with_circuit_breaker(
            func=success_func,
            failure_threshold=3
        )
        
        assert result == "success"


class TestRetryStatistics:
    """Test retry statistics and history."""
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        manager = RetryManager()
        
        # Successful execution
        manager.retry(lambda: "success")
        
        # Failed execution
        try:
            manager.retry(lambda: (_ for _ in ()).throw(AuthenticationError("auth failed")))
        except AuthenticationError:
            pass
        
        stats = manager.get_statistics()
        
        assert stats["total_attempts"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1
        assert stats["success_rate"] == 0.5
    
    def test_recent_attempts_tracking(self):
        """Test recent attempts tracking."""
        manager = RetryManager()
        
        # Execute some operations
        manager.retry(lambda: "success")
        
        recent = manager.get_recent_attempts(limit=5)
        
        assert len(recent) == 1
        assert recent[0].success is True
        assert recent[0].attempt_number == 1
    
    def test_clear_statistics(self):
        """Test clearing statistics."""
        manager = RetryManager()
        
        manager.retry(lambda: "success")
        
        # Check stats exist
        stats = manager.get_statistics()
        assert stats["total_attempts"] > 0
        
        # Clear and check
        manager.clear_statistics()
        stats = manager.get_statistics()
        assert stats["total_attempts"] == 0