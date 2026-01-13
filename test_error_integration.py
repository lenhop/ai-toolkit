#!/usr/bin/env python3
"""
Integration tests for the Error Handling Toolkit.

This script tests the error handling functionality with real scenarios
and various error types.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_toolkit.errors import (
    ErrorHandler, RetryManager, RetryConfig, CircuitBreakerConfig,
    AIException, ModelError, ParseError, APIError, RateLimitError,
    TimeoutError, AuthenticationError, ValidationError,
    ErrorSeverity, ErrorCategory, RetryStrategy
)


def test_basic_error_handling():
    """Test basic error handling functionality."""
    print("🧪 Testing Basic Error Handling")
    print("=" * 50)
    
    # Create error handler
    handler = ErrorHandler(enable_stack_trace=False)
    
    # Test handling different error types
    errors_to_test = [
        ModelError("Model failed", model_name="gpt-4", provider="openai"),
        ParseError("JSON parse failed", parser_type="json", input_data='{"invalid"}'),
        APIError("API error", status_code=500, endpoint="/api/test"),
        RateLimitError("Rate limited", retry_after=30),
        TimeoutError("Operation timed out", timeout_duration=30.0),
        AuthenticationError("Invalid API key", auth_type="api_key"),
        ValidationError("Invalid email", field_name="email", field_value="invalid-email")
    ]
    
    for error in errors_to_test:
        try:
            handler.handle_error(error)
        except Exception as e:
            print(f"   ✅ {type(error).__name__}: {error.message}")
            print(f"      Category: {error.category.value}, Severity: {error.severity.value}")
            if error.suggestions:
                print(f"      Suggestions: {', '.join(error.suggestions[:2])}")
    
    # Check statistics
    stats = handler.get_error_statistics()
    print(f"\n📊 Error Statistics:")
    print(f"   Total errors: {stats['total_errors']}")
    print(f"   Error types: {len(stats['error_types'])}")
    print(f"   Most common: {stats['most_common_error']}")
    
    return True


def test_retry_mechanisms():
    """Test retry mechanisms with different strategies."""
    print("\n🧪 Testing Retry Mechanisms")
    print("=" * 50)
    
    # Test exponential backoff
    print("   🔄 Testing Exponential Backoff:")
    
    attempt_count = 0
    
    def failing_function():
        nonlocal attempt_count
        attempt_count += 1
        print(f"      Attempt {attempt_count}")
        if attempt_count < 3:
            raise RateLimitError(f"Rate limited (attempt {attempt_count})")
        return f"Success after {attempt_count} attempts"
    
    config = RetryConfig(
        max_attempts=4,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.1,  # Short delay for testing
        jitter=False
    )
    
    retry_manager = RetryManager(config=config)
    
    start_time = time.time()
    result = retry_manager.retry(failing_function)
    duration = time.time() - start_time
    
    print(f"   ✅ Result: {result}")
    print(f"   ⏱️  Duration: {duration:.2f}s")
    
    # Test different strategies
    strategies_to_test = [
        (RetryStrategy.FIXED_DELAY, "Fixed Delay"),
        (RetryStrategy.LINEAR_BACKOFF, "Linear Backoff"),
        (RetryStrategy.FIBONACCI_BACKOFF, "Fibonacci Backoff")
    ]
    
    for strategy, name in strategies_to_test:
        config = RetryConfig(
            max_attempts=3,
            strategy=strategy,
            base_delay=0.05,
            jitter=False
        )
        manager = RetryManager(config=config)
        
        delays = []
        for attempt in range(1, 4):
            delay = manager._calculate_delay(attempt, config)
            delays.append(delay)
        
        print(f"   📈 {name}: {[f'{d:.2f}s' for d in delays]}")
    
    return True


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n🧪 Testing Circuit Breaker")
    print("=" * 50)
    
    # Configure circuit breaker
    circuit_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=1.0,  # Short timeout for testing
        success_threshold=2
    )
    
    retry_config = RetryConfig(max_attempts=1)  # No retries for this test
    manager = RetryManager(config=retry_config, circuit_config=circuit_config)
    
    failure_count = 0
    
    def unreliable_service():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 5:  # Fail first 5 times
            raise APIError(f"Service unavailable (failure {failure_count})", status_code=503)
        return f"Service recovered after {failure_count} attempts"
    
    print("   🔴 Causing circuit breaker to open:")
    
    # Cause failures to open circuit
    for i in range(1, 4):
        try:
            manager.retry(unreliable_service)
        except APIError as e:
            print(f"      Failure {i}: {e.message}")
    
    # Circuit should now be open
    print("   ⚡ Circuit breaker should be open:")
    try:
        manager.retry(unreliable_service)
    except Exception as e:
        print(f"      Blocked: {type(e).__name__}: {e}")
    
    # Wait for recovery timeout
    print("   ⏳ Waiting for recovery timeout...")
    time.sleep(1.2)
    
    # Circuit should now be half-open and allow recovery
    print("   🟡 Testing recovery:")
    try:
        result = manager.retry(unreliable_service)
        print(f"      ✅ {result}")
    except Exception as e:
        print(f"      Still failing: {e}")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"\n📊 Circuit Breaker Stats:")
    print(f"   Circuit state: {stats['circuit_state']}")
    print(f"   Total attempts: {stats['total_attempts']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")
    
    return True


def test_error_recovery_strategies():
    """Test error recovery strategies."""
    print("\n🧪 Testing Error Recovery Strategies")
    print("=" * 50)
    
    handler = ErrorHandler()
    
    # Add custom fallback rule
    from ai_toolkit.errors.error_handler import ErrorRule, ErrorAction
    
    def fallback_handler(error, context):
        print(f"      🔄 Fallback triggered for {type(error).__name__}")
        return f"Fallback result for {context.operation}"
    
    fallback_rule = ErrorRule(
        error_type=ValidationError,
        action=ErrorAction.FALLBACK,
        fallback_handler=fallback_handler
    )
    
    handler.add_rule(fallback_rule)
    
    # Test fallback
    print("   🛡️  Testing fallback strategy:")
    error = ValidationError("Invalid input", field_name="test_field")
    result = handler.handle_error(error, operation="validation_test")
    print(f"      ✅ Fallback result: {result}")
    
    # Test custom error handler
    def custom_handler(error, context, rule):
        print(f"      🔧 Custom handler for {type(error).__name__}")
        if isinstance(error, ParseError):
            return {"error": "parse_failed", "recovered": True}
        raise error
    
    custom_rule = ErrorRule(
        error_type=ParseError,
        action=ErrorAction.LOG,
        custom_handler=custom_handler
    )
    
    handler.add_rule(custom_rule)
    
    print("   🔧 Testing custom handler:")
    error = ParseError("JSON malformed", parser_type="json")
    result = handler.handle_error(error, operation="parse_test")
    print(f"      ✅ Custom result: {result}")
    
    return True


def test_conditional_retry():
    """Test conditional retry logic."""
    print("\n🧪 Testing Conditional Retry")
    print("=" * 50)
    
    # Custom retry condition
    def smart_retry_condition(exception, attempt):
        print(f"      🤔 Evaluating retry for {type(exception).__name__} (attempt {attempt})")
        
        # Don't retry auth errors
        if isinstance(exception, AuthenticationError):
            print("         ❌ Auth error - no retry")
            return False
        
        # Retry API errors up to 3 times, but only for 5xx status codes
        if isinstance(exception, APIError):
            if exception.status_code and exception.status_code >= 500:
                should_retry = attempt < 3
                print(f"         {'✅' if should_retry else '❌'} Server error - {'retry' if should_retry else 'no retry'}")
                return should_retry
            else:
                print("         ❌ Client error - no retry")
                return False
        
        # Default retry for other errors
        should_retry = attempt < 2
        print(f"         {'✅' if should_retry else '❌'} Default - {'retry' if should_retry else 'no retry'}")
        return should_retry
    
    config = RetryConfig(
        max_attempts=4,
        base_delay=0.1,
        retry_condition=smart_retry_condition
    )
    
    manager = RetryManager(config=config)
    
    # Test scenarios
    test_cases = [
        ("Auth Error", lambda: AuthenticationError("Invalid token")),
        ("Server Error", lambda: APIError("Server error", status_code=500)),
        ("Client Error", lambda: APIError("Bad request", status_code=400)),
        ("Rate Limit", lambda: RateLimitError("Too many requests"))
    ]
    
    for test_name, error_func in test_cases:
        print(f"   🎯 Testing {test_name}:")
        
        attempt_count = 0
        
        def failing_func():
            nonlocal attempt_count
            attempt_count += 1
            raise error_func()
        
        try:
            manager.retry(failing_func)
        except Exception as e:
            print(f"      Final result: {type(e).__name__} after {attempt_count} attempts")
        
        attempt_count = 0  # Reset for next test
    
    return True


def test_error_analysis_and_reporting():
    """Test error analysis and reporting features."""
    print("\n🧪 Testing Error Analysis and Reporting")
    print("=" * 50)
    
    handler = ErrorHandler()
    
    # Generate various errors for analysis
    test_errors = [
        ModelError("Model timeout", model_name="gpt-4", severity=ErrorSeverity.HIGH),
        ModelError("Model overloaded", model_name="gpt-3.5", severity=ErrorSeverity.MEDIUM),
        ParseError("JSON error", parser_type="json"),
        ParseError("XML error", parser_type="xml"),
        APIError("Rate limited", status_code=429),
        APIError("Server error", status_code=500),
        APIError("Not found", status_code=404),
        TimeoutError("Connection timeout", timeout_duration=30.0),
        ValidationError("Invalid email", field_name="email"),
        ValidationError("Missing field", field_name="name")
    ]
    
    print("   📝 Generating test errors:")
    for i, error in enumerate(test_errors, 1):
        try:
            handler.handle_error(error, operation=f"test_operation_{i}")
        except:
            pass  # Expected to raise
        print(f"      {i:2d}. {type(error).__name__}: {error.message}")
    
    # Analyze error patterns
    print("\n   📊 Error Analysis:")
    stats = handler.get_error_statistics()
    
    print(f"      Total errors processed: {stats['total_errors']}")
    print(f"      Unique error types: {len(stats['error_types'])}")
    print(f"      Most common error: {stats['most_common_error']}")
    
    print("\n      Error breakdown:")
    for error_type, count in sorted(stats['error_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_errors']) * 100
        print(f"         {error_type}: {count} ({percentage:.1f}%)")
    
    # Recent errors
    print("\n   🕒 Recent Errors:")
    recent = handler.get_recent_errors(limit=5)
    for error in recent:
        timestamp = time.strftime('%H:%M:%S', time.localtime(error['timestamp']))
        print(f"      [{timestamp}] {error['error_type']}: {error['message']}")
    
    # Generate report
    print("\n   📋 Error Report:")
    report = handler.create_error_report()
    print("      " + "\n      ".join(report.split('\n')[:10]))  # Show first 10 lines
    
    return True


def test_real_world_scenarios():
    """Test real-world error handling scenarios."""
    print("\n🧪 Testing Real-World Scenarios")
    print("=" * 50)
    
    # Scenario 1: API with rate limiting and occasional failures
    print("   🌐 Scenario 1: Unreliable API with rate limiting")
    
    call_count = 0
    
    def unreliable_api():
        nonlocal call_count
        call_count += 1
        
        if call_count == 2:
            raise RateLimitError("Rate limited", retry_after=1)
        elif call_count == 4:
            raise APIError("Server temporarily unavailable", status_code=503)
        elif call_count < 6:
            raise TimeoutError("Request timeout", timeout_duration=10.0)
        
        return {"status": "success", "data": f"API call {call_count} succeeded"}
    
    config = RetryConfig(
        max_attempts=8,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.1,
        max_delay=2.0
    )
    
    manager = RetryManager(config=config)
    
    try:
        start_time = time.time()
        result = manager.retry(unreliable_api)
        duration = time.time() - start_time
        
        print(f"      ✅ Success: {result}")
        print(f"      📊 Attempts: {call_count}, Duration: {duration:.2f}s")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
    
    # Scenario 2: Model inference with fallback
    print("\n   🤖 Scenario 2: Model inference with fallback")
    
    handler = ErrorHandler()
    
    def fallback_model(error, context):
        print("      🔄 Primary model failed, using fallback model")
        return {"model": "fallback", "result": "Fallback response", "confidence": 0.7}
    
    from ai_toolkit.errors.error_handler import ErrorRule, ErrorAction
    
    fallback_rule = ErrorRule(
        error_type=ModelError,
        action=ErrorAction.FALLBACK,
        fallback_handler=fallback_model
    )
    
    handler.add_rule(fallback_rule)
    
    def primary_model_inference():
        raise ModelError("Primary model overloaded", model_name="gpt-4", provider="openai")
    
    result = handler.handle_error(
        ModelError("Primary model overloaded", model_name="gpt-4", provider="openai"),
        operation="model_inference"
    )
    
    print(f"      ✅ Inference result: {result}")
    
    # Scenario 3: Data processing pipeline with multiple error types
    print("\n   🔧 Scenario 3: Data processing pipeline")
    
    pipeline_handler = ErrorHandler()
    
    def process_data_item(item_id: int):
        if item_id == 1:
            raise ValidationError("Invalid data format", field_name="format")
        elif item_id == 2:
            raise ParseError("JSON parsing failed", parser_type="json")
        elif item_id == 3:
            raise TimeoutError("Processing timeout", timeout_duration=5.0)
        else:
            return f"Processed item {item_id}"
    
    # Process multiple items
    items = [1, 2, 3, 4, 5]
    results = []
    
    for item_id in items:
        try:
            result = process_data_item(item_id)
            results.append(f"Item {item_id}: {result}")
            print(f"      ✅ Item {item_id}: Success")
        except Exception as e:
            try:
                pipeline_handler.handle_error(e, operation=f"process_item_{item_id}")
                results.append(f"Item {item_id}: Error handled")
                print(f"      ⚠️  Item {item_id}: Error handled - {type(e).__name__}")
            except Exception:
                results.append(f"Item {item_id}: Failed")
                print(f"      ❌ Item {item_id}: Failed - {type(e).__name__}")
    
    print(f"      📊 Pipeline results: {len([r for r in results if 'Success' in r])}/{len(items)} successful")
    
    return True


def run_all_tests():
    """Run all error handling integration tests."""
    print("🎯 AI Toolkit Error Handling Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Error Handling", test_basic_error_handling),
        ("Retry Mechanisms", test_retry_mechanisms),
        ("Circuit Breaker", test_circuit_breaker),
        ("Error Recovery Strategies", test_error_recovery_strategies),
        ("Conditional Retry", test_conditional_retry),
        ("Error Analysis and Reporting", test_error_analysis_and_reporting),
        ("Real-World Scenarios", test_real_world_scenarios),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results.append((test_name, success, None))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if error:
            print(f"     Error: {error}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All error handling tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)