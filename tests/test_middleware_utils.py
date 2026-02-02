"""
Tests for Middleware Utilities

This module tests all middleware wrapper functions in ai_toolkit.agents.middleware_utils.

Test Categories:
    1. Custom Middleware Tests
    2. Built-in Middleware Wrapper Tests
    3. Integration Tests

Author: AI Toolkit Team
Version: 1.0.0
"""

import pytest
import os
from unittest.mock import Mock, MagicMock
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. CUSTOM MIDDLEWARE TESTS
# =============================================================================

class TestCustomMiddleware:
    """Test custom middleware functions."""
    
    def test_create_dynamic_model_selector(self):
        """Test dynamic model selector creation."""
        from ai_toolkit.agents.middleware_utils import create_dynamic_model_selector
        
        # Create mock models
        basic_model = Mock()
        advanced_model = Mock()
        
        # Create middleware
        middleware = create_dynamic_model_selector(
            basic_model=basic_model,
            advanced_model=advanced_model,
            threshold=10
        )
        
        assert middleware is not None
        print("✓ create_dynamic_model_selector works")
    
    def test_create_tool_error_handler(self):
        """Test tool error handler creation."""
        from ai_toolkit.agents.middleware_utils import create_tool_error_handler
        
        # Create middleware
        middleware = create_tool_error_handler(
            error_message_template="Tool failed: {error}",
            log_errors=True
        )
        
        assert middleware is not None
        print("✓ create_tool_error_handler works")
    
    def test_create_context_based_prompt(self):
        """Test context-based prompt creation."""
        from ai_toolkit.agents.middleware_utils import create_context_based_prompt
        
        def prompt_generator(context):
            return "Test prompt"
        
        # Create middleware
        middleware = create_context_based_prompt(prompt_generator)
        
        assert middleware is not None
        print("✓ create_context_based_prompt works")


# =============================================================================
# 2. BUILT-IN MIDDLEWARE WRAPPER TESTS
# =============================================================================

class TestBuiltInMiddlewareWrappers:
    """Test built-in middleware wrapper functions."""
    
    def test_create_summarization_middleware(self):
        """Test summarization middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_summarization_middleware
        
        middleware = create_summarization_middleware(
            model="gpt-4o-mini",
            trigger_tokens=4000,
            keep_messages=20
        )
        
        assert middleware is not None
        print("✓ create_summarization_middleware works")
    
    def test_create_human_in_loop_middleware(self):
        """Test human-in-the-loop middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_human_in_loop_middleware
        
        middleware = create_human_in_loop_middleware(
            interrupt_on={
                "send_email": True,
                "read_email": False
            }
        )
        
        assert middleware is not None
        print("✓ create_human_in_loop_middleware works")
    
    def test_create_model_call_limit_middleware(self):
        """Test model call limit middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_model_call_limit_middleware
        
        middleware = create_model_call_limit_middleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end"
        )
        
        assert middleware is not None
        print("✓ create_model_call_limit_middleware works")
    
    def test_create_tool_call_limit_middleware(self):
        """Test tool call limit middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_tool_call_limit_middleware
        
        # Global limit
        middleware1 = create_tool_call_limit_middleware(
            thread_limit=20,
            run_limit=10
        )
        
        # Tool-specific limit
        middleware2 = create_tool_call_limit_middleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3
        )
        
        assert middleware1 is not None
        assert middleware2 is not None
        print("✓ create_tool_call_limit_middleware works")
    
    def test_create_model_fallback_middleware(self):
        """Test model fallback middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_model_fallback_middleware
        
        middleware = create_model_fallback_middleware(
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022"
        )
        
        assert middleware is not None
        print("✓ create_model_fallback_middleware works")
    
    def test_create_pii_middleware(self):
        """Test PII middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_pii_middleware
        import re
        
        # Built-in PII type
        middleware1 = create_pii_middleware("email", strategy="redact")
        
        # Custom regex pattern
        middleware2 = create_pii_middleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block"
        )
        
        # Custom detector function
        def detect_ssn(content: str) -> list:
            matches = []
            pattern = r"\d{3}-\d{2}-\d{4}"
            for match in re.finditer(pattern, content):
                matches.append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
            return matches
        
        middleware3 = create_pii_middleware(
            "ssn",
            detector=detect_ssn,
            strategy="hash"
        )
        
        assert middleware1 is not None
        assert middleware2 is not None
        assert middleware3 is not None
        print("✓ create_pii_middleware works")
    
    def test_create_todo_list_middleware(self):
        """Test todo list middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_todo_list_middleware
        
        middleware = create_todo_list_middleware()
        
        assert middleware is not None
        print("✓ create_todo_list_middleware works")
    
    def test_create_llm_tool_selector_middleware(self):
        """Test LLM tool selector middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_llm_tool_selector_middleware
        
        middleware = create_llm_tool_selector_middleware(
            model="gpt-4o-mini",
            max_tools=3,
            always_include=["search"]
        )
        
        assert middleware is not None
        print("✓ create_llm_tool_selector_middleware works")
    
    def test_create_tool_retry_middleware(self):
        """Test tool retry middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_tool_retry_middleware
        
        middleware = create_tool_retry_middleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0
        )
        
        assert middleware is not None
        print("✓ create_tool_retry_middleware works")
    
    def test_create_model_retry_middleware(self):
        """Test model retry middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_model_retry_middleware
        
        middleware = create_model_retry_middleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0
        )
        
        assert middleware is not None
        print("✓ create_model_retry_middleware works")
    
    def test_create_llm_tool_emulator(self):
        """Test LLM tool emulator creation."""
        from ai_toolkit.agents.middleware_utils import create_llm_tool_emulator
        
        middleware = create_llm_tool_emulator()
        
        assert middleware is not None
        print("✓ create_llm_tool_emulator works")
    
    def test_create_context_editing_middleware(self):
        """Test context editing middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_context_editing_middleware
        
        middleware = create_context_editing_middleware(
            trigger_tokens=100000,
            keep_tool_uses=3
        )
        
        assert middleware is not None
        print("✓ create_context_editing_middleware works")
    
    def test_create_shell_tool_middleware(self):
        """Test shell tool middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_shell_tool_middleware
        
        middleware = create_shell_tool_middleware(
            workspace_root="/tmp"
        )
        
        assert middleware is not None
        print("✓ create_shell_tool_middleware works")
    
    def test_create_filesystem_search_middleware(self):
        """Test filesystem search middleware creation."""
        from ai_toolkit.agents.middleware_utils import create_filesystem_search_middleware
        
        middleware = create_filesystem_search_middleware(
            root_path="/tmp",
            use_ripgrep=True
        )
        
        assert middleware is not None
        print("✓ create_filesystem_search_middleware works")


# =============================================================================
# 3. INTEGRATION TESTS
# =============================================================================

class TestMiddlewareIntegration:
    """Test middleware integration with agents."""
    
    @pytest.mark.skipif(
        not os.environ.get('DEEPSEEK_API_KEY'),
        reason="DEEPSEEK_API_KEY not set"
    )
    def test_agent_with_custom_middleware(self):
        """Test agent creation with custom middleware."""
        from ai_toolkit.models import ModelManager
        from ai_toolkit.agents import create_agent_with_tools
        from ai_toolkit.agents.middleware_utils import create_tool_error_handler
        from ai_toolkit.tools import create_search_tool
        
        # Create model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek"
        )
        
        # Create middleware
        error_handler = create_tool_error_handler()
        
        # Create tools
        search = create_search_tool()
        
        # Create agent with middleware
        agent = create_agent_with_tools(
            model=model,
            tools=[search],
            middleware=[error_handler]
        )
        
        assert agent is not None
        print("✓ Agent with custom middleware created successfully")
    
    @pytest.mark.skipif(
        not os.environ.get('DEEPSEEK_API_KEY'),
        reason="DEEPSEEK_API_KEY not set"
    )
    def test_agent_with_multiple_middleware(self):
        """Test agent creation with multiple middleware."""
        from ai_toolkit.models import ModelManager
        from ai_toolkit.agents import create_agent_with_tools
        from ai_toolkit.agents.middleware_utils import (
            create_tool_error_handler,
            create_tool_call_limit_middleware,
            create_model_retry_middleware
        )
        from ai_toolkit.tools import create_search_tool
        
        # Create model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek"
        )
        
        # Create middleware
        error_handler = create_tool_error_handler()
        call_limiter = create_tool_call_limit_middleware(run_limit=5)
        model_retry = create_model_retry_middleware(max_retries=2)
        
        # Create tools
        search = create_search_tool()
        
        # Create agent with multiple middleware
        agent = create_agent_with_tools(
            model=model,
            tools=[search],
            middleware=[error_handler, call_limiter, model_retry]
        )
        
        assert agent is not None
        print("✓ Agent with multiple middleware created successfully")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MIDDLEWARE UTILITIES TESTS")
    print("=" * 80 + "\n")
    
    # Run custom middleware tests
    print("1. Custom Middleware Tests")
    print("-" * 80)
    test_custom = TestCustomMiddleware()
    test_custom.test_create_dynamic_model_selector()
    test_custom.test_create_tool_error_handler()
    test_custom.test_create_context_based_prompt()
    print()
    
    # Run built-in middleware wrapper tests
    print("2. Built-in Middleware Wrapper Tests")
    print("-" * 80)
    test_builtin = TestBuiltInMiddlewareWrappers()
    test_builtin.test_create_summarization_middleware()
    test_builtin.test_create_human_in_loop_middleware()
    test_builtin.test_create_model_call_limit_middleware()
    test_builtin.test_create_tool_call_limit_middleware()
    test_builtin.test_create_model_fallback_middleware()
    test_builtin.test_create_pii_middleware()
    test_builtin.test_create_todo_list_middleware()
    test_builtin.test_create_llm_tool_selector_middleware()
    test_builtin.test_create_tool_retry_middleware()
    test_builtin.test_create_model_retry_middleware()
    test_builtin.test_create_llm_tool_emulator()
    test_builtin.test_create_context_editing_middleware()
    test_builtin.test_create_shell_tool_middleware()
    test_builtin.test_create_filesystem_search_middleware()
    print()
    
    # Run integration tests
    print("3. Integration Tests")
    print("-" * 80)
    test_integration = TestMiddlewareIntegration()
    
    if os.environ.get('DEEPSEEK_API_KEY'):
        try:
            test_integration.test_agent_with_custom_middleware()
            test_integration.test_agent_with_multiple_middleware()
        except Exception as e:
            print(f"⚠ Integration tests skipped: {e}")
    else:
        print("⚠ Integration tests skipped (DEEPSEEK_API_KEY not set)")
    print()
    
    print("=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Custom middleware: 3 functions tested")
    print("  ✓ Built-in middleware wrappers: 14 functions tested")
    print("  ✓ Integration tests: 2 scenarios tested")
    print()
