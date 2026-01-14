#!/usr/bin/env python3
"""
Configuration Management Toolkit Examples

This script demonstrates how to use the configuration management toolkit
for loading, validating, and managing application configurations.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_toolkit.config import (
    ConfigManager,
    ConfigValidator,
    RequiredRule,
    TypeRule,
    RangeRule,
    PatternRule,
    ChoiceRule,
    CustomRule,
    EnvLoader
)


def basic_config_manager_examples():
    """Demonstrate basic ConfigManager functionality."""
    print("📁 Basic ConfigManager Examples")
    print("=" * 50)
    
    # Example 1: Loading configuration from YAML
    print("1. Loading Configuration from YAML:")
    
    config_data = {
        'application': {
            'name': 'AI Toolkit Demo',
            'version': '1.0.0',
            'debug': True,
            'port': 8080
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'ai_toolkit_db',
            'pool_size': 10
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)
    
    manager = ConfigManager(config_path=config_path)
    
    print(f"   Config file: {config_path}")
    print(f"   App name: {manager.get_config('application.name')}")
    print(f"   App version: {manager.get_config('application.version')}")
    print(f"   Database host: {manager.get_config('database.host')}")
    print(f"   Log level: {manager.get_config('logging.level')}")
    
    # Example 2: Accessing nested configuration
    print("\n2. Accessing Nested Configuration:")
    app_config = manager.get_config('application')
    print(f"   Application config: {app_config}")
    print(f"   Debug mode: {manager.get_config('application.debug')}")
    print(f"   Port: {manager.get_config('application.port')}")
    
    # Example 3: Setting configuration values
    print("\n3. Setting Configuration Values:")
    manager.set_config('application.environment', 'production')
    manager.set_config('application.max_connections', 100)
    manager.set_config('cache.enabled', True)
    manager.set_config('cache.ttl', 3600)
    
    print(f"   Environment: {manager.get_config('application.environment')}")
    print(f"   Max connections: {manager.get_config('application.max_connections')}")
    print(f"   Cache enabled: {manager.get_config('cache.enabled')}")
    print(f"   Cache TTL: {manager.get_config('cache.ttl')}")
    
    # Example 4: Getting all configuration keys
    print("\n4. Getting All Configuration Keys:")
    all_keys = manager.get_all_keys()
    print(f"   Total keys: {len(all_keys)}")
    print(f"   Application keys: {[k for k in all_keys if k.startswith('application')]}")
    print(f"   Database keys: {[k for k in all_keys if k.startswith('database')]}")
    
    config_path.unlink()


def env_var_substitution_examples():
    """Demonstrate environment variable substitution."""
    print("\n🔐 Environment Variable Substitution Examples")
    print("=" * 50)
    
    # Set environment variables
    os.environ['DB_PASSWORD'] = 'secret_password_123'
    os.environ['API_KEY'] = 'sk-api-key-xyz'
    os.environ['REDIS_PORT'] = '6379'
    
    # Example 1: Basic substitution
    print("1. Basic Environment Variable Substitution:")
    
    config_data = {
        'database': {
            'password': '${DB_PASSWORD}',
            'connection_string': 'postgresql://user:${DB_PASSWORD}@localhost/db'
        },
        'api': {
            'key': '${API_KEY}',
            'base_url': 'https://api.example.com'
        },
        'redis': {
            'host': 'localhost',
            'port': '${REDIS_PORT}'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)
    
    manager = ConfigManager(config_path=config_path)
    
    print(f"   DB password: {manager.get_config('database.password')}")
    print(f"   Connection string: {manager.get_config('database.connection_string')}")
    print(f"   API key: {manager.get_config('api.key')}")
    print(f"   Redis port: {manager.get_config('redis.port')}")
    
    # Example 2: Substitution with defaults
    print("\n2. Environment Variable Substitution with Defaults:")
    
    config_with_defaults = {
        'server': {
            'host': '${SERVER_HOST:0.0.0.0}',
            'port': '${SERVER_PORT:8000}',
            'workers': '${WORKERS:4}'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_with_defaults, f)
        defaults_path = Path(f.name)
    
    defaults_manager = ConfigManager(config_path=defaults_path)
    
    print(f"   Server host (default): {defaults_manager.get_config('server.host')}")
    print(f"   Server port (default): {defaults_manager.get_config('server.port')}")
    print(f"   Workers (default): {defaults_manager.get_config('server.workers')}")
    
    # Cleanup
    del os.environ['DB_PASSWORD']
    del os.environ['API_KEY']
    del os.environ['REDIS_PORT']
    config_path.unlink()
    defaults_path.unlink()


def config_validation_examples():
    """Demonstrate configuration validation."""
    print("\n✅ Configuration Validation Examples")
    print("=" * 50)
    
    validator = ConfigValidator()
    
    # Example 1: Model configuration validation
    print("1. Model Configuration Validation:")
    
    valid_model_config = {
        'api_key': 'sk-1234567890abcdef',
        'model': 'gpt-4',
        'temperature': 0.7,
        'max_tokens': 2000
    }
    
    is_valid = validator.validate_model_config(valid_model_config)
    print(f"   Valid model config: {is_valid}")
    
    invalid_model_config = {
        'api_key': '',  # Empty
        'temperature': 3.0,  # Out of range
        'max_tokens': 'invalid'  # Wrong type
    }
    
    is_valid = validator.validate_model_config(invalid_model_config)
    print(f"   Invalid model config: {is_valid}")
    if not is_valid:
        print(f"   Errors: {len(validator.get_errors())} validation errors")
    
    # Example 2: Custom validation rules
    print("\n2. Custom Validation Rules:")
    validator.clear_rules()
    
    config = {
        'server': {
            'port': 8080,
            'host': 'localhost',
            'protocol': 'https'
        },
        'email': 'admin@example.com',
        'environment': 'production'
    }
    
    # Add validation rules
    validator.add_rule('server.port', TypeRule(int))
    validator.add_rule('server.port', RangeRule(1, 65535))
    validator.add_rule('server.protocol', ChoiceRule(['http', 'https']))
    validator.add_rule('email', PatternRule(r'^[\w\.-]+@[\w\.-]+\.\w+$'))
    validator.add_rule('environment', ChoiceRule(['development', 'staging', 'production']))
    
    is_valid = validator.validate(config)
    print(f"   Validation result: {is_valid}")
    print(f"   Server port: {config['server']['port']} (valid range: 1-65535)")
    print(f"   Protocol: {config['server']['protocol']} (allowed: http, https)")
    print(f"   Email: {config['email']} (valid format)")
    print(f"   Environment: {config['environment']} (allowed values)")
    
    # Example 3: API key validation
    print("\n3. API Key Validation:")
    
    api_keys = {
        'openai': 'sk-1234567890abcdef1234567890',
        'anthropic': 'sk-ant-1234567890abcdef',
        'deepseek': 'sk-deepseek-1234567890'
    }
    
    is_valid = validator.validate_api_keys(api_keys)
    print(f"   API keys validation: {is_valid}")
    print(f"   Number of keys: {len(api_keys)}")
    
    # Example 4: URL validation
    print("\n4. URL Validation:")
    
    urls = [
        'https://api.openai.com/v1',
        'http://localhost:8080',
        'https://192.168.1.1:3000'
    ]
    
    for url in urls:
        is_valid = validator.validate_url(url)
        print(f"   {url}: {'✓ Valid' if is_valid else '✗ Invalid'}")


def env_loader_examples():
    """Demonstrate environment variable loading."""
    print("\n🌍 Environment Variable Loading Examples")
    print("=" * 50)
    
    # Create .env file
    env_content = """
# Application settings
APP_NAME=AI Toolkit
APP_VERSION=1.0.0
APP_DEBUG=true
APP_PORT=8080

# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_toolkit
DB_USER=admin
DB_PASSWORD=secret123

# API keys
OPENAI_API_KEY=sk-openai-test-key
DEEPSEEK_API_KEY=sk-deepseek-test-key
ANTHROPIC_API_KEY=sk-ant-test-key

# Feature flags
FEATURES=caching,logging,monitoring
MAX_WORKERS=4
TIMEOUT=30.5
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content)
        env_path = Path(f.name)
    
    loader = EnvLoader(env_file=env_path)
    
    # Example 1: Basic type-safe loading
    print("1. Type-Safe Environment Variable Loading:")
    print(f"   APP_NAME (string): {loader.get_str('APP_NAME')}")
    print(f"   APP_PORT (int): {loader.get_int('APP_PORT')}")
    print(f"   APP_DEBUG (bool): {loader.get_bool('APP_DEBUG')}")
    print(f"   TIMEOUT (float): {loader.get_float('TIMEOUT')}")
    print(f"   FEATURES (list): {loader.get_list('FEATURES')}")
    
    # Example 2: API key retrieval
    print("\n2. API Key Retrieval:")
    openai_key = loader.get_api_key('openai')
    deepseek_key = loader.get_api_key('deepseek')
    anthropic_key = loader.get_api_key('anthropic')
    
    print(f"   OpenAI: {openai_key}")
    print(f"   DeepSeek: {deepseek_key}")
    print(f"   Anthropic: {anthropic_key}")
    
    # Example 3: Batch loading with mapping
    print("\n3. Batch Loading with Mapping:")
    db_mapping = {
        'host': 'DB_HOST',
        'port': 'DB_PORT',
        'name': 'DB_NAME',
        'user': 'DB_USER',
        'password': 'DB_PASSWORD'
    }
    
    db_config = loader.load_from_env(db_mapping)
    print(f"   Database config: {db_config}")
    
    # Example 4: Fallback keys
    print("\n4. Fallback Key Retrieval:")
    # Try multiple possible key names
    api_url = loader.get_with_fallback(
        ['API_URL', 'SERVICE_URL', 'BASE_URL'],
        default='https://api.default.com'
    )
    print(f"   API URL (with fallback): {api_url}")
    
    # Example 5: Required variables validation
    print("\n5. Required Variables Validation:")
    required_vars = ['APP_NAME', 'APP_VERSION', 'DB_HOST']
    is_valid = loader.validate_required_vars(required_vars)
    print(f"   All required variables present: {is_valid}")
    
    env_path.unlink()


def config_merge_examples():
    """Demonstrate configuration merging."""
    print("\n🔀 Configuration Merging Examples")
    print("=" * 50)
    
    # Example 1: Merging base and environment-specific configs
    print("1. Merging Base and Environment-Specific Configs:")
    
    # Base configuration
    base_config = {
        'app': {
            'name': 'AI Toolkit',
            'version': '1.0.0',
            'features': {
                'caching': True,
                'logging': True
            }
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'pool_size': 5
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(base_config, f)
        base_path = Path(f.name)
    
    # Production overrides
    prod_config = {
        'app': {
            'debug': False,
            'features': {
                'monitoring': True
            }
        },
        'database': {
            'host': 'prod-db.example.com',
            'pool_size': 20,
            'ssl': True
        },
        'cache': {
            'enabled': True,
            'ttl': 3600
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(prod_config, f)
        prod_path = Path(f.name)
    
    # Merge configurations
    manager = ConfigManager(auto_load=False)
    merged = manager.merge_configs(base_path, prod_path)
    
    print(f"   App name (from base): {merged['app']['name']}")
    print(f"   App debug (from prod): {merged['app']['debug']}")
    print(f"   Caching (from base): {merged['app']['features']['caching']}")
    print(f"   Monitoring (from prod): {merged['app']['features']['monitoring']}")
    print(f"   DB host (overridden): {merged['database']['host']}")
    print(f"   DB pool size (overridden): {merged['database']['pool_size']}")
    print(f"   Cache enabled (new): {merged['cache']['enabled']}")
    
    # Example 2: Merging multiple configuration layers
    print("\n2. Merging Multiple Configuration Layers:")
    
    # Development overrides
    dev_config = {
        'app': {
            'debug': True
        },
        'database': {
            'host': 'localhost'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(dev_config, f)
        dev_path = Path(f.name)
    
    # Merge: base -> prod -> dev
    dev_manager = ConfigManager(auto_load=False)
    dev_merged = dev_manager.merge_configs(base_path, prod_path, dev_path)
    
    print(f"   Final debug setting: {dev_merged['app']['debug']}")
    print(f"   Final DB host: {dev_merged['database']['host']}")
    print(f"   Final pool size: {dev_merged['database']['pool_size']}")
    
    # Cleanup
    base_path.unlink()
    prod_path.unlink()
    dev_path.unlink()


def integrated_workflow_example():
    """Demonstrate integrated workflow."""
    print("\n🔄 Integrated Workflow Example")
    print("=" * 50)
    
    print("Scenario: Setting up AI model configuration with validation")
    
    # Step 1: Create .env file with API keys
    print("\n1. Loading API keys from environment:")
    env_content = """
DEEPSEEK_API_KEY=sk-deepseek-production-key
QWEN_API_KEY=sk-qwen-production-key
GLM_API_KEY=sk-glm-production-key
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content)
        env_path = Path(f.name)
    
    env_loader = EnvLoader(env_file=env_path)
    print("   ✓ Environment variables loaded")
    
    # Step 2: Create configuration with env var substitution
    print("\n2. Creating configuration with environment substitution:")
    config_data = {
        'models': {
            'deepseek': {
                'api_key': '${DEEPSEEK_API_KEY}',
                'model': 'deepseek-chat',
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'qwen': {
                'api_key': '${QWEN_API_KEY}',
                'model': 'qwen-turbo',
                'temperature': 0.8,
                'max_tokens': 1500
            },
            'glm': {
                'api_key': '${GLM_API_KEY}',
                'model': 'glm-4',
                'temperature': 0.9,
                'max_tokens': 1000
            }
        },
        'settings': {
            'retry_attempts': 3,
            'timeout': 30,
            'log_level': 'INFO'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)
    
    manager = ConfigManager(config_path=config_path)
    print("   ✓ Configuration loaded with API keys substituted")
    
    # Step 3: Validate model configurations
    print("\n3. Validating model configurations:")
    validator = ConfigValidator()
    
    for model_name in ['deepseek', 'qwen', 'glm']:
        model_config = manager.get_config(f'models.{model_name}')
        is_valid = validator.validate_model_config(model_config)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"   {model_name}: {status}")
    
    # Step 4: Update configuration
    print("\n4. Updating configuration:")
    manager.set_config('models.deepseek.max_tokens', 4000)
    manager.set_config('settings.cache_enabled', True)
    print(f"   ✓ Updated DeepSeek max_tokens: {manager.get_config('models.deepseek.max_tokens')}")
    print(f"   ✓ Enabled caching: {manager.get_config('settings.cache_enabled')}")
    
    # Step 5: Save updated configuration
    print("\n5. Saving updated configuration:")
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        updated_path = Path(f.name)
    
    manager.save_config(updated_path)
    print(f"   ✓ Configuration saved to: {updated_path}")
    
    # Step 6: Verify saved configuration
    print("\n6. Verifying saved configuration:")
    verify_manager = ConfigManager(config_path=updated_path)
    print(f"   DeepSeek max_tokens: {verify_manager.get_config('models.deepseek.max_tokens')}")
    print(f"   Cache enabled: {verify_manager.get_config('settings.cache_enabled')}")
    print("   ✓ Configuration verified successfully")
    
    # Cleanup
    env_path.unlink()
    config_path.unlink()
    updated_path.unlink()


def run_all_examples():
    """Run all configuration management examples."""
    print("🎯 AI Toolkit Configuration Management Examples")
    print("=" * 60)
    
    examples = [
        basic_config_manager_examples,
        env_var_substitution_examples,
        config_validation_examples,
        env_loader_examples,
        config_merge_examples,
        integrated_workflow_example,
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
    
    print(f"\n🎉 Configuration Management Examples Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Loading and saving YAML/JSON configurations")
    print("   ✅ Environment variable substitution")
    print("   ✅ Configuration validation with custom rules")
    print("   ✅ Type-safe environment variable loading")
    print("   ✅ Configuration merging and layering")
    print("   ✅ API key management")
    print("   ✅ Integrated workflow with all components")


if __name__ == "__main__":
    run_all_examples()
