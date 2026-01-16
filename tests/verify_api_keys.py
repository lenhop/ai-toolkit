"""
Script to verify API keys for AI model providers.

This script checks if API keys are configured correctly and tests
actual API connections to verify they work.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_toolkit.config.env_loader import EnvLoader
from ai_toolkit.config.config_manager import ConfigManager
from ai_toolkit.models.model_manager import ModelManager
from ai_toolkit.models.model_config import ModelConfig


def check_env_vars() -> Dict[str, Tuple[bool, Optional[str]]]:
    """
    Check if required environment variables are set.
    
    Returns:
        Dictionary mapping provider names to (is_set, value_preview) tuples
    """
    loader = EnvLoader(auto_load=True)
    
    providers = ['deepseek', 'qwen', 'glm']
    results = {}
    
    for provider in providers:
        try:
            api_key = loader.get_api_key(provider, required=False)
            if api_key:
                # Show first 10 and last 4 characters for security
                preview = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
                results[provider] = (True, preview)
            else:
                results[provider] = (False, None)
        except Exception as e:
            results[provider] = (False, f"Error: {str(e)}")
    
    return results


def test_deepseek_connection(api_key: str) -> Tuple[bool, str]:
    """
    Test DeepSeek API connection.
    
    Args:
        api_key: DeepSeek API key
        
    Returns:
        Tuple of (success, message)
    """
    try:
        from langchain_community.chat_models import ChatOpenAI
        
        model = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=100,
            request_timeout=30
        )
        
        # Make a simple test call
        response = model.invoke("Say 'Hello' in one word.")
        content = response.content if hasattr(response, 'content') else str(response)
        
        return True, f"Success: {content[:50]}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_qwen_connection(api_key: str) -> Tuple[bool, str]:
    """
    Test Qwen API connection.
    
    Args:
        api_key: Qwen API key
        
    Returns:
        Tuple of (success, message)
    """
    try:
        from langchain_community.chat_models import ChatOpenAI
        
        model = ChatOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=100,
            request_timeout=30
        )
        
        # Make a simple test call
        response = model.invoke("Say 'Hello' in one word.")
        content = response.content if hasattr(response, 'content') else str(response)
        
        return True, f"Success: {content[:50]}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_glm_connection(api_key: str) -> Tuple[bool, str]:
    """
    Test GLM API connection.
    
    Args:
        api_key: GLM API key
        
    Returns:
        Tuple of (success, message)
    """
    try:
        from zhipuai import ZhipuAI
        
        client = ZhipuAI(api_key=api_key)
        
        # Make a simple test call
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        return True, f"Success: {content[:50]}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_with_model_manager(provider: str) -> Tuple[bool, str]:
    """
    Test API connection using ModelManager.
    
    Args:
        provider: Provider name ('deepseek', 'qwen', 'glm')
        
    Returns:
        Tuple of (success, message)
    """
    try:
        manager = ModelManager()
        
        # Create model instance
        model = manager.create_model(provider)
        
        # Make a simple test call
        response = model.invoke("Say 'Hello' in one word.")
        content = response.content if hasattr(response, 'content') else str(response)
        
        return True, f"Success: {content[:50]}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main verification function."""
    print("=" * 80)
    print("API Key Verification for AI Model Providers")
    print("=" * 80)
    print()
    
    # Step 1: Check environment variables
    print("Step 1: Checking environment variables...")
    print("-" * 80)
    env_results = check_env_vars()
    
    all_keys_present = True
    for provider, (is_set, preview) in env_results.items():
        status = "✓" if is_set else "✗"
        if is_set:
            print(f"{status} {provider.upper():12s} API key: {preview}")
        else:
            print(f"{status} {provider.upper():12s} API key: NOT FOUND")
            all_keys_present = False
    
    print()
    
    if not all_keys_present:
        print("⚠️  Warning: Some API keys are missing from environment variables.")
        print("   Please check your .env file or environment configuration.")
        print()
    
    # Step 2: Test API connections
    print("Step 2: Testing API connections...")
    print("-" * 80)
    
    loader = EnvLoader(auto_load=True)
    test_results = {}
    
    # Test DeepSeek
    print("Testing DeepSeek...", end=" ", flush=True)
    deepseek_key = loader.get_api_key('deepseek', required=False)
    if deepseek_key:
        success, message = test_deepseek_connection(deepseek_key)
        status = "✓" if success else "✗"
        print(f"{status} {message}")
        test_results['deepseek'] = (success, message)
    else:
        print("✗ Skipped (no API key)")
        test_results['deepseek'] = (False, "No API key")
    
    # Test Qwen
    print("Testing Qwen...", end=" ", flush=True)
    qwen_key = loader.get_api_key('qwen', required=False)
    if qwen_key:
        success, message = test_qwen_connection(qwen_key)
        status = "✓" if success else "✗"
        print(f"{status} {message}")
        test_results['qwen'] = (success, message)
    else:
        print("✗ Skipped (no API key)")
        test_results['qwen'] = (False, "No API key")
    
    # Test GLM
    print("Testing GLM...", end=" ", flush=True)
    glm_key = loader.get_api_key('glm', required=False)
    if glm_key:
        success, message = test_glm_connection(glm_key)
        status = "✓" if success else "✗"
        print(f"{status} {message}")
        test_results['glm'] = (success, message)
    else:
        print("✗ Skipped (no API key)")
        test_results['glm'] = (False, "No API key")
    
    print()
    
    # Step 3: Test with ModelManager
    print("Step 3: Testing with ModelManager...")
    print("-" * 80)
    
    manager_results = {}
    for provider in ['deepseek', 'qwen', 'glm']:
        if env_results[provider][0]:  # Only test if key exists
            print(f"Testing {provider} with ModelManager...", end=" ", flush=True)
            success, message = test_with_model_manager(provider)
            status = "✓" if success else "✗"
            print(f"{status} {message}")
            manager_results[provider] = (success, message)
        else:
            print(f"Skipping {provider} (no API key)")
            manager_results[provider] = (False, "No API key")
    
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_tests_passed = True
    for provider in ['deepseek', 'qwen', 'glm']:
        env_ok = env_results[provider][0]
        api_ok = test_results[provider][0] if provider in test_results else False
        manager_ok = manager_results[provider][0] if provider in manager_results else False
        
        if env_ok:
            if api_ok and manager_ok:
                status = "✓"
            elif api_ok or manager_ok:
                status = "⚠"
                all_tests_passed = False
            else:
                status = "✗"
                all_tests_passed = False
        else:
            status = "✗"
            all_tests_passed = False
        
        print(f"{status} {provider.upper():12s} - Env: {'✓' if env_ok else '✗'}, "
              f"API: {'✓' if api_ok else '✗'}, Manager: {'✓' if manager_ok else '✗'}")
    
    print()
    
    if all_tests_passed:
        print("✓ All API keys are configured and working correctly!")
        return 0
    else:
        print("✗ Some API keys are missing or not working correctly.")
        print("  Please check your configuration and API key validity.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
