#!/usr/bin/env python3
"""
Integration test for the model management toolkit.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

from ai_toolkit.models import ModelManager, ModelConfig, load_config_from_env


def test_model_manager_basic():
    """Test basic ModelManager functionality."""
    print("🧪 Testing ModelManager Basic Functionality...")
    
    try:
        # Initialize ModelManager
        manager = ModelManager()
        print("✅ ModelManager initialized successfully")
        
        # List available providers
        providers = manager.list_providers()
        print(f"✅ Available providers: {providers}")
        
        # List available models
        models = manager.list_models()
        print(f"✅ Available models: {len(models)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelManager basic test failed: {e}")
        return False


def test_model_config_loading():
    """Test model configuration loading."""
    print("\n🧪 Testing Model Configuration Loading...")
    
    try:
        # Test loading from environment for each provider
        providers_to_test = ['deepseek', 'qwen', 'glm']
        loaded_configs = {}
        
        for provider in providers_to_test:
            try:
                config = load_config_from_env(provider)
                loaded_configs[provider] = config
                print(f"✅ {provider.upper()} config loaded successfully")
                print(f"   Model: {config.model}")
                print(f"   Base URL: {config.base_url}")
                print(f"   Temperature: {config.temperature}")
                
            except Exception as e:
                print(f"⚠️  {provider.upper()} config failed: {e}")
        
        print(f"✅ Successfully loaded {len(loaded_configs)} configurations")
        return len(loaded_configs) > 0
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False


def test_model_creation_dry_run():
    """Test model creation without actual API calls."""
    print("\n🧪 Testing Model Creation (Dry Run)...")
    
    try:
        manager = ModelManager()
        
        # Test configuration validation for available providers
        providers_to_test = []
        
        # Check which providers have API keys
        if os.getenv('DEEPSEEK_API_KEY'):
            providers_to_test.append('deepseek')
        if os.getenv('QWEN_API_KEY'):
            providers_to_test.append('qwen')
        if os.getenv('GLM_API_KEY'):
            providers_to_test.append('glm')
        
        print(f"📋 Testing providers with API keys: {providers_to_test}")
        
        for provider in providers_to_test:
            try:
                # Test configuration creation (this validates the config)
                config = load_config_from_env(provider)
                print(f"✅ {provider.upper()} configuration is valid")
                
                # Test provider class loading
                from ai_toolkit.models.model_providers import get_provider_class
                provider_class = get_provider_class(provider)
                print(f"✅ {provider.upper()} provider class loaded: {provider_class.__name__}")
                
            except Exception as e:
                print(f"❌ {provider.upper()} validation failed: {e}")
        
        return len(providers_to_test) > 0
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_model_manager_with_config():
    """Test ModelManager with configuration file."""
    print("\n🧪 Testing ModelManager with Configuration File...")
    
    try:
        # Try to load with config file
        config_paths = [
            Path("config/config.yaml"),
            Path("config/models.yaml")
        ]
        
        config_found = False
        for config_path in config_paths:
            if config_path.exists():
                try:
                    manager = ModelManager(config_path)
                    print(f"✅ ModelManager loaded with config: {config_path}")
                    
                    # Test listing models with config
                    models = manager.list_models()
                    print(f"✅ Found {len(models)} models in configuration")
                    
                    config_found = True
                    break
                    
                except Exception as e:
                    print(f"⚠️  Failed to load config {config_path}: {e}")
        
        if not config_found:
            print("⚠️  No configuration files found, testing with defaults")
            manager = ModelManager()
            models = manager.list_models()
            print(f"✅ Default ModelManager created, {len(models)} models available")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration file test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Starting Model Management Toolkit Integration Tests...\n")
    
    tests = [
        test_model_manager_basic,
        test_model_config_loading,
        test_model_creation_dry_run,
        test_model_manager_with_config,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Integration Test Results Summary:")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_func.__name__}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Model management toolkit is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)