#!/usr/bin/env python3
"""
Error Handling Toolkit Examples

This script demonstrates how to use the error handling toolkit
for robust error management in AI applications.
"""

import os
import sys
import time
import logging
from typing import Any, Dict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_toolkit.errors import (
    ErrorHandler, RetryManager, RetryConfig, CircuitBreakerConfig,
    AIException, ModelError, ParseError, APIError, RateLimitError,
    TimeoutError, AuthenticationError, ValidationError,
    ErrorSeverity, ErrorCategory, RetryStrategy,
    retry, with_exponential_backoff
)


def basic_exception_examples():
    """Demonstrate basic custom exception usage."""
    print("🚀 Basic Exception Examples")
    print("=" * 50)
    
    # Create different types of AI exceptions
    exceptions = [
        ModelError(
            message="Model inference failed",
            model_name="gpt-4",
            provider="openai",
            severity=ErrorSeverity.HIGH
        ),
        
        ParseError(
            message="Failed to parse JSON response",
            parser_type="json",
            input_data='{"incomplete": json',
            expected_format="valid JSON object"
        ),
        
        APIError(
            message="API request failed",
            status_code=429,
            endpoint="/api/completions",
            response_body='{"error": "rate_limit_exceeded"}'
        ),
        
        RateLimitError(
            message="Rate limit exceeded",
            retry_after=60,
            limit_type="requests_per_minute"
        ),
        
        TimeoutError(
            message="Request timed out",
            operation="model_inference",
            timeout_duration=30.0
        ),
        
        AuthenticationError(
            message="Invalid API key",
            auth_type="bearer_token"
        ),
        
        ValidationError(
            message="Invalid email format",
            field_name="email",
            field_value="invalid-email",
            validation_rule="email_format"
        )
    ]
    
    print("Creating and displaying custom exceptions:")
    for i, exc in enumerate(exceptions, 1):
        print(f"\n{i}. {type(exc).__name__}:")
        print(f"   Message: {exc.message}")
        print(f"   Category: {exc.category.value}")
        print(f"   Severity: {exc.severity.value}")
        
        if exc.context:
            print(f"   Context: {exc.context}")
        
        if exc.suggestions:
            print(f"   Suggestions: {', '.join(exc.suggestions[:2])}")
        
        # Convert to dictionary
        exc_dict = exc.to_dict()
        print(f"   Dict keys: {list(exc_dict.keys())}")


def error_handler_examples():
    """Demonstrate ErrorHandler usage."""
    print("\n🛡️ Error Handler Examples")
    print("=" * 50)
    
    # Create error handler with custom logger
    logger = logging.getLogger("error_handler_demo")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    
    error_handler = ErrorHandler(logger=logger, enable_stack_trace=False)
    
    print("1. Basic Error Handling:")
    
    # Test different error handling actions
    test_errors = [
        ValidationError("Invalid input", field_name="test_field"),
        RateLimitError("Rate limited", retry_after=30),
        AuthenticationError("Invalid credentials")
    ]
    
    for error in test_errors:
        try:
            error_handler.handle_error(error, operation="demo_operation")
        except Exception as e:
            print(f"   Handled {type(error).__name__}: Re-raised as expected")
    
    print("\n2. Custom Error Rules:")
    
    # Add custom rule for validation errors
    from ai_toolkit.errors.error_handler import ErrorRule, ErrorAction
    
    def validation_fallback(error, context):
        return {"error": "validation_failed", "field": error.field_name, "fallback": True}
    
    validation_rule = ErrorRule(
        error_type=ValidationError,
        action=ErrorAction.FALLBACK,
        fallback_handler=validation_fallback
    )
    
    error_handler.add_rule(validation_rule)
    
    # Test the custom rule
    validation_error = ValidationError("Invalid email", field_name="email")
    result = error_handler.handle_error(validation_error, operation="validation_test")
    print(f"   Validation fallback result: {result}")
    
    print("\n3. Error Statistics:")
    stats = error_handler.get_error_statistics()
    print(f"   Total errors: {stats['total_errors']}")
    print(f"   Error types: {stats['error_types']}")
    
    # Generate error report
    report = error_handler.create_error_report()
    print(f"\n4. Error Report Preview:")
    print("   " + "\n   ".join(report.split('\n')[:5]))


def retry_manager_examples():
    """Demonstrate RetryManager usage."""
    print("\n🔄 Retry Manager Examples")
    print("=" * 50)
    
    print("1. Basic Retry with Exponential Backoff:")
    
    attempt_count = 0
    
    def unreliable_function():
        nonlocal attempt_count
        attempt_count += 1
        print(f"   Attempt {attempt_count}")
        
        if attempt_count < 3:
            raise RateLimitError(f"Rate limited on attempt {attempt_count}")
        
        return f"Success after {attempt_count} attempts!"
    
    # Configure retry with exponential backoff
    config = RetryConfig(
        max_attempts=5,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.1,  # Short delay for demo
        max_delay=2.0,
        jitter=True
    )
    
    retry_manager = RetryManager(config=config)
    
    try:
        result = retry_manager.retry(unreliable_function)
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print("\n2. Different Retry Strategies:")
    
    strategies = [
        (RetryStrategy.FIXED_DELAY, "Fixed Delay"),
        (RetryStrategy.LINEAR_BACKOFF, "Linear Backoff"),
        (RetryStrategy.FIBONACCI_BACKOFF, "Fibonacci Backoff")
    ]
    
    for strategy, name in strategies:
        config = RetryConfig(
            strategy=strategy,
            base_delay=0.1,
            max_attempts=4,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        delays = []
        for attempt in range(1, 5):
            delay = manager._calculate_delay(attempt, config)
            delays.append(f"{delay:.2f}s")
        
        print(f"   {name}: {' -> '.join(delays)}")
    
    print("\n3. Conditional Retry:")
    
    def smart_retry_condition(exception, attempt):
        # Only retry rate limit and timeout errors
        if isinstance(exception, (RateLimitError, TimeoutError)):
            return attempt < 3
        return False
    
    conditional_config = RetryConfig(
        max_attempts=5,
        base_delay=0.1,
        retry_condition=smart_retry_condition
    )
    
    conditional_manager = RetryManager(config=conditional_config)
    
    # Test with different error types
    test_cases = [
        ("Rate Limit Error", RateLimitError("Rate limited")),
        ("Auth Error", AuthenticationError("Invalid token")),
        ("Timeout Error", TimeoutError("Timed out"))
    ]
    
    for test_name, error in test_cases:
        should_retry = conditional_manager.should_retry(error)
        print(f"   {test_name}: {'Will retry' if should_retry else 'Will not retry'}")


def circuit_breaker_examples():
    """Demonstrate circuit breaker functionality."""
    print("\n⚡ Circuit Breaker Examples")
    print("=" * 50)
    
    # Configure circuit breaker
    circuit_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=2.0,
        success_threshold=2
    )
    
    retry_config = RetryConfig(max_attempts=1)  # No retries for this demo
    manager = RetryManager(config=retry_config, circuit_config=circuit_config)
    
    failure_count = 0
    
    def flaky_service():
        nonlocal failure_count
        failure_count += 1
        
        # Fail first 5 times, then succeed
        if failure_count <= 5:
            raise APIError(f"Service failure #{failure_count}", status_code=503)
        
        return f"Service recovered! (call #{failure_count})"
    
    print("1. Demonstrating Circuit Breaker States:")
    
    # Cause failures to open circuit
    print("   🔴 Causing failures to open circuit:")
    for i in range(1, 4):
        try:
            manager.retry(flaky_service)
        except APIError:
            print(f"      Failure {i}: Service failed")
    
    # Circuit should now be open
    print("   ⚡ Circuit is now OPEN - blocking requests:")
    try:
        manager.retry(flaky_service)
    except Exception as e:
        print(f"      Blocked: {type(e).__name__}")
    
    # Wait for recovery timeout
    print("   ⏳ Waiting for recovery timeout...")
    time.sleep(2.5)
    
    # Circuit should now allow testing
    print("   🟡 Circuit is now HALF-OPEN - testing service:")
    try:
        result = manager.retry(flaky_service)
        print(f"      ✅ {result}")
    except Exception as e:
        print(f"      Still failing: {e}")
    
    # Show statistics
    stats = manager.get_statistics()
    print(f"\n2. Circuit Breaker Statistics:")
    print(f"   Circuit state: {stats['circuit_state']}")
    print(f"   Total attempts: {stats['total_attempts']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")


def decorator_examples():
    """Demonstrate retry decorators."""
    print("\n🎭 Decorator Examples")
    print("=" * 50)
    
    print("1. Basic Retry Decorator:")
    
    attempt_count = 0
    
    @retry(max_attempts=4, base_delay=0.1, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
    def decorated_function():
        nonlocal attempt_count
        attempt_count += 1
        print(f"   Decorated function attempt {attempt_count}")
        
        if attempt_count < 3:
            raise TimeoutError(f"Timeout on attempt {attempt_count}")
        
        return f"Decorated function succeeded after {attempt_count} attempts"
    
    try:
        result = decorated_function()
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   ❌ {e}")
    
    print("\n2. Convenience Function:")
    
    def another_unreliable_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise RateLimitError("Randomly rate limited")
        return "Lucky success!"
    
    try:
        result = with_exponential_backoff(
            func=another_unreliable_function,
            max_attempts=5,
            base_delay=0.1
        )
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   ❌ {e}")


def real_world_scenarios():
    """Demonstrate real-world error handling scenarios."""
    print("\n🌍 Real-World Scenarios")
    print("=" * 50)
    
    print("1. AI Model Inference with Fallback:")
    
    class ModelService:
        def __init__(self):
            self.error_handler = ErrorHandler()
            self._setup_fallback_rules()
        
        def _setup_fallback_rules(self):
            from ai_toolkit.errors.error_handler import ErrorRule, ErrorAction
            
            def fallback_to_smaller_model(error, context):
                print("      🔄 Primary model failed, using fallback model")
                return {
                    "model": "gpt-3.5-turbo",
                    "response": "Fallback response from smaller model",
                    "confidence": 0.8,
                    "fallback_reason": str(error)
                }
            
            fallback_rule = ErrorRule(
                error_type=ModelError,
                action=ErrorAction.FALLBACK,
                fallback_handler=fallback_to_smaller_model
            )
            
            self.error_handler.add_rule(fallback_rule)
        
        def generate_text(self, prompt: str, model: str = "gpt-4"):
            try:
                # Simulate model failure
                if model == "gpt-4":
                    raise ModelError(
                        "Model overloaded",
                        model_name=model,
                        provider="openai",
                        severity=ErrorSeverity.HIGH
                    )
                
                return {"model": model, "response": f"Generated text for: {prompt}"}
            
            except Exception as e:
                return self.error_handler.handle_error(e, operation="text_generation")
    
    model_service = ModelService()
    result = model_service.generate_text("Write a haiku about AI")
    print(f"   Model inference result: {result}")
    
    print("\n2. API Client with Comprehensive Error Handling:")
    
    class RobustAPIClient:
        def __init__(self):
            self.retry_manager = RetryManager(
                config=RetryConfig(
                    max_attempts=5,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    base_delay=0.5,
                    max_delay=10.0
                ),
                circuit_config=CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=30.0
                )
            )
        
        def make_api_call(self, endpoint: str, data: Dict[str, Any]):
            def api_request():
                # Simulate various API failures
                import random
                rand = random.random()
                
                if rand < 0.3:
                    raise RateLimitError("Rate limit exceeded", retry_after=5)
                elif rand < 0.5:
                    raise APIError("Server error", status_code=500, endpoint=endpoint)
                elif rand < 0.6:
                    raise TimeoutError("Request timeout", timeout_duration=10.0)
                else:
                    return {"status": "success", "data": f"Response from {endpoint}"}
            
            try:
                return self.retry_manager.retry(api_request)
            except Exception as e:
                return {"status": "error", "error": str(e), "type": type(e).__name__}
    
    api_client = RobustAPIClient()
    
    # Make several API calls
    for i in range(3):
        result = api_client.make_api_call(f"/api/endpoint_{i}", {"test": "data"})
        status = "✅" if result.get("status") == "success" else "❌"
        print(f"   API call {i+1}: {status} {result.get('status', 'unknown')}")
    
    print("\n3. Data Processing Pipeline with Error Recovery:")
    
    class DataProcessor:
        def __init__(self):
            self.error_handler = ErrorHandler()
            self.processed_count = 0
            self.error_count = 0
        
        def process_item(self, item_id: int, data: str):
            try:
                # Simulate various processing errors
                if item_id % 3 == 0:
                    raise ValidationError(f"Invalid data format for item {item_id}")
                elif item_id % 5 == 0:
                    raise ParseError(f"Parse error for item {item_id}", parser_type="json")
                elif item_id % 7 == 0:
                    raise TimeoutError(f"Processing timeout for item {item_id}")
                
                # Successful processing
                self.processed_count += 1
                return f"Processed item {item_id}: {data}"
            
            except Exception as e:
                self.error_count += 1
                try:
                    # Try to handle the error
                    self.error_handler.handle_error(e, operation=f"process_item_{item_id}")
                    return f"Item {item_id}: Error handled gracefully"
                except Exception:
                    return f"Item {item_id}: Processing failed"
        
        def process_batch(self, items):
            results = []
            for item_id, data in items:
                result = self.process_item(item_id, data)
                results.append(result)
            
            return results
    
    processor = DataProcessor()
    
    # Process a batch of items
    test_items = [(i, f"data_{i}") for i in range(1, 16)]
    results = processor.process_batch(test_items)
    
    print(f"   Batch processing results:")
    print(f"   Successfully processed: {processor.processed_count}/{len(test_items)}")
    print(f"   Errors encountered: {processor.error_count}")
    
    # Show some results
    for i, result in enumerate(results[:5]):
        status = "✅" if "Processed item" in result else "⚠️"
        print(f"      {status} {result}")


def run_all_examples():
    """Run all error handling examples."""
    print("🎯 AI Toolkit Error Handling Examples")
    print("=" * 60)
    
    examples = [
        basic_exception_examples,
        error_handler_examples,
        retry_manager_examples,
        circuit_breaker_examples,
        decorator_examples,
        real_world_scenarios,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            if i > 1:
                print(f"\n{'='*20} Example {i} {'='*20}")
            example_func()
        except KeyboardInterrupt:
            print("\n\n⏹️ Examples interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in example: {e}")
    
    print(f"\n🎉 Error Handling Examples Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Custom exception types with rich context")
    print("   ✅ Configurable error handling rules")
    print("   ✅ Multiple retry strategies with backoff")
    print("   ✅ Circuit breaker for service protection")
    print("   ✅ Conditional retry logic")
    print("   ✅ Error statistics and reporting")
    print("   ✅ Decorator-based retry mechanisms")
    print("   ✅ Real-world integration patterns")


if __name__ == "__main__":
    run_all_examples()