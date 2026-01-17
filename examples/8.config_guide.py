#!/usr/bin/env python3
"""
Configuration Guide - Simple config management

Shows how to load and use configuration in AI Toolkit.

Examples:
1. Load environment variables
2. Load YAML configuration
3. Access nested config values
4. Save configuration

Philosophy: Keep it simple!
- Use .env for secrets
- Use YAML for structured config
- Use standard Python dict access
"""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_toolkit.config import load_env, load_config, save_config, get_nested


def example_1_load_env():
    """Example 1: Load environment variables."""
    print("=" * 60)
    print("Example 1: Load Environment Variables")
    print("=" * 60)
    
    # Load .env file
    env_file = project_root / '.env'
    if load_env(env_file):
        print(f"✅ Loaded environment from: {env_file}")
    else:
        print(f"⚠️  No .env file found at: {env_file}")
    
    # Access environment variables
    api_keys = {
        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
        'QWEN_API_KEY': os.getenv('QWEN_API_KEY'),
        'GLM_API_KEY': os.getenv('GLM_API_KEY'),
    }
    
    print("\n🔑 API Keys found:")
    for key, value in api_keys.items():
        if value and not value.startswith('your_'):
            print(f"   ✅ {key}: {value[:10]}...")
        else:
            print(f"   ❌ {key}: Not set")


def example_2_load_yaml():
    """Example 2: Load YAML configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Load YAML Configuration")
    print("=" * 60)
    
    # Load config file
    config_file = project_root / 'config' / 'config.yaml'
    
    if not config_file.exists():
        print(f"⚠️  Config file not found: {config_file}")
        return
    
    config = load_config(config_file)
    print(f"✅ Loaded config from: {config_file}")
    
    # Access config values
    print("\n📋 Configuration:")
    print(f"   App name: {config.get('app', {}).get('name', 'N/A')}")
    print(f"   Version: {config.get('app', {}).get('version', 'N/A')}")
    
    # Access model configs
    if 'models' in config:
        print("\n🤖 Models configured:")
        for provider, settings in config['models'].items():
            model_name = settings.get('model', 'N/A')
            print(f"   {provider}: {model_name}")


def example_3_nested_access():
    """Example 3: Safe nested config access."""
    print("\n" + "=" * 60)
    print("Example 3: Safe Nested Access")
    print("=" * 60)
    
    # Sample config
    config = {
        'models': {
            'deepseek': {
                'model': 'deepseek-chat',
                'temperature': 0.7,
                'max_tokens': 4096
            },
            'qwen': {
                'model': 'qwen-turbo',
                'temperature': 0.8
            }
        }
    }
    
    # Safe access with get_nested
    print("\n✅ Safe nested access:")
    
    # Existing path
    model = get_nested(config, 'models', 'deepseek', 'model')
    print(f"   DeepSeek model: {model}")
    
    # Non-existing path with default
    missing = get_nested(config, 'models', 'gpt4', 'model', default='not-configured')
    print(f"   GPT-4 model: {missing}")
    
    # Deep nesting
    temp = get_nested(config, 'models', 'deepseek', 'temperature', default=0.7)
    print(f"   Temperature: {temp}")


def example_4_save_config():
    """Example 4: Save configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Save Configuration")
    print("=" * 60)
    
    # Create sample config
    my_config = {
        'model': 'deepseek-chat',
        'temperature': 0.7,
        'max_tokens': 4096,
        'system_prompt': 'You are a helpful assistant.',
        'tools': ['calculator', 'search', 'code_interpreter']
    }
    
    # Save to file
    output_file = project_root / 'my_config.yaml'
    save_config(my_config, output_file)
    print(f"✅ Saved config to: {output_file}")
    
    # Load it back
    loaded = load_config(output_file)
    print(f"✅ Loaded config back:")
    print(f"   Model: {loaded['model']}")
    print(f"   Temperature: {loaded['temperature']}")
    print(f"   Tools: {', '.join(loaded['tools'])}")
    
    # Clean up
    output_file.unlink()
    print(f"🗑️  Cleaned up: {output_file}")


def run_all_examples():
    """Run all configuration examples."""
    print("\n🎯 AI Toolkit Configuration Guide")
    print("Simple and practical config management\n")
    
    examples = [
        example_1_load_env,
        example_2_load_yaml,
        example_3_nested_access,
        example_4_save_config,
    ]
    
    for example in examples:
        try:
            example()
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Configuration Guide Complete!")
    print("=" * 60)
    print("\n💡 Key Takeaways:")
    print("   1. Use load_env() for .env files")
    print("   2. Use load_config() for YAML/JSON files")
    print("   3. Use get_nested() for safe nested access")
    print("   4. Keep it simple - standard Python dicts!")


if __name__ == "__main__":
    run_all_examples()
