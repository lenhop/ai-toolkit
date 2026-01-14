#!/usr/bin/env python3
"""
Integration tests for the Configuration Management Toolkit.

This script tests the configuration management functionality
with real scenarios and file operations.
"""

import os
import sys
import tempfile
import yaml
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_toolkit.config import (
    ConfigManager,
    ConfigValidator,
    RequiredRule,
    TypeRule,
    RangeRule,
    PatternRule,
    ChoiceRule,
    EnvLoader
)


def test_config_manager_basic():
    """Test basic ConfigManager functionality."""
    print("🧪 Testing Basic ConfigManager")
    print("=" * 50)
    
    # Create sample config
    config_data = {
        'app': {
            'name': 'AI Toolkit',
            'version': '1.0.0',
            'debug': True
        },
        'models': {
            'deepseek': {
                'api_key': 'sk-test123',
                'model': 'deepseek-chat',
                'temperature': 0.7
            }
        }
    }
    
    # Test YAML config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        yaml_path = Path(f.name)
    
    manager = ConfigManager(config_path=yaml_path)
    
    print("1. Loading YAML configuration:")
    print(f"   Config path: {yaml_path}")
    print(f"   App name: {manager.get_config('app.name')}")
    print(f"   Model: {manager.get_config('models.deepseek.model')}")
    
    # Test nested key access
    print("\n2. Nested key access:")
    print(f"   Temperature: {manager.get_config('models.deepseek.temperature')}")
    print(f"   Debug mode: {manager.get_config('app.debug')}")
    
    # Test setting values
    print("\n3. Setting configuration values:")
    manager.set_config('app.environment', 'production')
    manager.set_config('models.deepseek.max_tokens', 1000)
    print(f"   Environment: {manager.get_config('app.environment')}")
    print(f"   Max tokens: {manager.get_config('models.deepseek.max_tokens')}")
    
    # Test getting all keys
    print("\n4. All configuration keys:")
    keys = manager.get_all_keys()
    print(f"   Total keys: {len(keys)}")
    print(f"   Sample keys: {keys[:5]}")
    
    yaml_path.unlink()
    return True


def test_env_var_substitution():
    """Test environment variable substitution."""
    print("\n🧪 Testing Environment Variable Substitution")
    print("=" * 50)
    
    # Set environment variables
    os.environ['TEST_API_KEY'] = 'sk-real-key-123'
    os.environ['TEST_PORT'] = '8080'
    
    config_data = {
        'api': {
            'key': '${TEST_API_KEY}',
            'url': 'http://localhost:${TEST_PORT}',
            'timeout': '${TEST_TIMEOUT:30}'  # With default
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        yaml_path = Path(f.name)
    
    manager = ConfigManager(config_path=yaml_path)
    
    print("1. Environment variable substitution:")
    print(f"   API key: {manager.get_config('api.key')}")
    print(f"   API URL: {manager.get_config('api.url')}")
    print(f"   Timeout (with default): {manager.get_config('api.timeout')}")
    
    # Cleanup
    del os.environ['TEST_API_KEY']
    del os.environ['TEST_PORT']
    yaml_path.unlink()
    
    return True


def test_config_validation():
    """Test configuration validation."""
    print("\n🧪 Testing Configuration Validation")
    print("=" * 50)
    
    validator = ConfigValidator()
    
    # Test model configuration validation
    print("1. Model configuration validation:")
    
    valid_config = {
        'api_key': 'sk-test123456789',
        'model': 'gpt-4',
        'temperature': 0.7,
        'max_tokens': 1000
    }
    
    result = validator.validate_model_config(valid_config)
    print(f"   Valid config: {result}")
    
    invalid_config = {
        'api_key': '',  # Empty
        'temperature': 3.0  # Out of range
    }
    
    result = validator.validate_model_config(invalid_config)
    print(f"   Invalid config: {result}")
    if not result:
        print(f"   Errors: {validator.get_errors()}")
    
    # Test custom validation rules
    print("\n2. Custom validation rules:")
    validator.clear_rules()
    
    config = {
        'port': 8080,
        'host': 'localhost',
        'environment': 'production'
    }
    
    validator.add_rule('port', TypeRule(int))
    validator.add_rule('port', RangeRule(1, 65535))
    validator.add_rule('environment', ChoiceRule(['development', 'staging', 'production']))
    
    result = validator.validate(config)
    print(f"   Validation result: {result}")
    
    # Test API key validation
    print("\n3. API key validation:")
    api_keys = {
        'openai': 'sk-1234567890abcdef',
        'anthropic': 'sk-ant-1234567890',
        'invalid': 'short'  # Too short
    }
    
    result = validator.validate_api_keys(api_keys)
    print(f"   Validation result: {result}")
    if not result:
        print(f"   Errors: {validator.get_errors()}")
    
    return True


def test_env_loader():
    """Test environment variable loading."""
    print("\n🧪 Testing Environment Variable Loading")
    print("=" * 50)
    
    # Create .env file
    env_content = """
# Test environment file
APP_NAME=AI Toolkit
APP_VERSION=1.0.0
DEBUG=true
PORT=8080
MAX_WORKERS=4
FEATURES=feature1,feature2,feature3
DEEPSEEK_API_KEY=sk-deepseek-test
OPENAI_API_KEY=sk-openai-test
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content)
        env_path = Path(f.name)
    
    loader = EnvLoader(env_file=env_path)
    
    print("1. Loading environment variables:")
    print(f"   APP_NAME: {loader.get_str('APP_NAME')}")
    print(f"   PORT: {loader.get_int('PORT')}")
    print(f"   DEBUG: {loader.get_bool('DEBUG')}")
    print(f"   FEATURES: {loader.get_list('FEATURES')}")
    
    # Test API key retrieval
    print("\n2. API key retrieval:")
    deepseek_key = loader.get_api_key('deepseek')
    openai_key = loader.get_api_key('openai')
    print(f"   DeepSeek API key: {deepseek_key}")
    print(f"   OpenAI API key: {openai_key}")
    
    # Test batch loading
    print("\n3. Batch loading with mapping:")
    mapping = {
        'name': 'APP_NAME',
        'version': 'APP_VERSION',
        'port': 'PORT'
    }
    
    config = loader.load_from_env(mapping)
    print(f"   Loaded config: {config}")
    
    # Test fallback keys
    print("\n4. Fallback key retrieval:")
    value = loader.get_with_fallback(
        ['NONEXISTENT_KEY', 'APP_NAME', 'ANOTHER_KEY'],
        default='fallback'
    )
    print(f"   Value with fallback: {value}")
    
    env_path.unlink()
    return True


def test_config_merge():
    """Test merging multiple configuration files."""
    print("\n🧪 Testing Configuration Merging")
    print("=" * 50)
    
    # Create base config
    base_config = {
        'app': {
            'name': 'AI Toolkit',
            'version': '1.0.0'
        },
        'database': {
            'host': 'localhost',
            'port': 5432
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(base_config, f)
        base_path = Path(f.name)
    
    # Create override config
    override_config = {
        'app': {
            'environment': 'production',
            'debug': False
        },
        'database': {
            'host': 'prod-db.example.com'
        },
        'cache': {
            'enabled': True,
            'ttl': 3600
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(override_config, f)
        override_path = Path(f.name)
    
    # Merge configs
    manager = ConfigManager(auto_load=False)
    merged = manager.merge_configs(base_path, override_path)
    
    print("1. Merged configuration:")
    print(f"   App name (from base): {merged['app']['name']}")
    print(f"   App environment (from override): {merged['app']['environment']}")
    print(f"   DB host (overridden): {merged['database']['host']}")
    print(f"   DB port (from base): {merged['database']['port']}")
    print(f"   Cache enabled (new): {merged['cache']['enabled']}")
    
    base_path.unlink()
    override_path.unlink()
    
    return True


def test_config_save_and_load():
    """Test saving and loading configuration."""
    print("\n🧪 Testing Configuration Save and Load")
    print("=" * 50)
    
    manager = ConfigManager(auto_load=False)
    
    # Create configuration
    config = {
        'app': {
            'name': 'Test App',
            'version': '2.0.0'
        },
        'settings': {
            'theme': 'dark',
            'language': 'en'
        }
    }
    
    manager.from_dict(config)
    
    # Save as YAML
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        yaml_path = Path(f.name)
    
    manager.save_config(yaml_path)
    print(f"1. Saved configuration to: {yaml_path}")
    
    # Load and verify
    new_manager = ConfigManager(config_path=yaml_path)
    print(f"   Loaded app name: {new_manager.get_config('app.name')}")
    print(f"   Loaded theme: {new_manager.get_config('settings.theme')}")
    
    # Save as JSON
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_path = Path(f.name)
    
    manager.save_config(json_path)
    print(f"\n2. Saved configuration to: {json_path}")
    
    # Load and verify
    json_manager = ConfigManager(config_path=json_path)
    print(f"   Loaded version: {json_manager.get_config('app.version')}")
    print(f"   Loaded language: {json_manager.get_config('settings.language')}")
    
    yaml_path.unlink()
    json_path.unlink()
    
    return True


def test_integrated_workflow():
    """Test integrated workflow with all components."""
    print("\n🧪 Testing Integrated Workflow")
    print("=" * 50)
    
    # 1. Create .env file
    env_content = """
DEEPSEEK_API_KEY=sk-deepseek-real-key
QWEN_API_KEY=sk-qwen-real-key
GLM_API_KEY=sk-glm-real-key
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content)
        env_path = Path(f.name)
    
    # 2. Load environment variables
    env_loader = EnvLoader(env_file=env_path)
    print("1. Loaded environment variables")
    
    # 3. Create configuration with env vars
    config_data = {
        'models': {
            'deepseek': {
                'api_key': '${DEEPSEEK_API_KEY}',
                'model': 'deepseek-chat',
                'temperature': 0.7
            },
            'qwen': {
                'api_key': '${QWEN_API_KEY}',
                'model': 'qwen-turbo',
                'temperature': 0.8
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)
    
    # 4. Load configuration
    config_manager = ConfigManager(config_path=config_path)
    print("2. Loaded configuration with env substitution")
    print(f"   DeepSeek API key: {config_manager.get_config('models.deepseek.api_key')}")
    
    # 5. Validate configuration
    validator = ConfigValidator()
    
    deepseek_config = config_manager.get_config('models.deepseek')
    is_valid = validator.validate_model_config(deepseek_config)
    print(f"3. Configuration validation: {is_valid}")
    
    # 6. Update and save
    config_manager.set_config('models.deepseek.max_tokens', 2000)
    
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        updated_path = Path(f.name)
    
    config_manager.save_config(updated_path)
    print(f"4. Saved updated configuration to: {updated_path}")
    
    # Cleanup
    env_path.unlink()
    config_path.unlink()
    updated_path.unlink()
    
    return True


def run_all_tests():
    """Run all configuration management integration tests."""
    print("🎯 AI Toolkit Configuration Management Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Basic ConfigManager", test_config_manager_basic),
        ("Environment Variable Substitution", test_env_var_substitution),
        ("Configuration Validation", test_config_validation),
        ("Environment Variable Loading", test_env_loader),
        ("Configuration Merging", test_config_merge),
        ("Configuration Save and Load", test_config_save_and_load),
        ("Integrated Workflow", test_integrated_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
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
        print("🎉 All configuration management tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
