#!/usr/bin/env python3
"""
Example usage of the AI Toolkit Model Management system.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'  # Changed from '../.env' to './.env'
load_dotenv(dotenv_path=env_path)

from ai_toolkit.models import ModelManager, ModelConfig, load_config_from_env


def basic_usage_example():
    """Demonstrate basic model management usage."""
    print("🚀 Basic Model Management Example")
    print("=" * 50)
    
    # Initialize ModelManager
    manager = ModelManager()
    
    # List available providers
    providers = manager.list_providers()
    print(f"📋 Available providers: {providers}")
    
    # List available models
    models = manager.list_models()
    print(f"📋 Available models: {len(models)}")
    for model in models[:3]:  # Show first 3
        print(f"   - {model['provider']}: {model['model']} ({model['status']})")
    
    print()


def configuration_example():
    """Demonstrate configuration loading."""
    print("⚙️  Configuration Management Example")
    print("=" * 50)
    
    # Load configuration from environment for each provider
    providers_to_test = ['deepseek', 'qwen', 'glm']
    
    for provider in providers_to_test:
        try:
            config = load_config_from_env(provider)
            print(f"✅ {provider.upper()} Configuration:")
            print(f"   Model: {config.model}")
            print(f"   Base URL: {config.base_url}")
            print(f"   Temperature: {config.temperature}")
            print(f"   Max Tokens: {config.max_tokens}")
            print()
            
        except Exception as e:
            print(f"❌ {provider.upper()} configuration failed: {e}")
            print()


def model_creation_example():
    """Demonstrate model creation (without actual API calls)."""
    print("🔧 Model Creation Example")
    print("=" * 50)
    
    manager = ModelManager()
    
    # Check which providers have API keys
    available_providers = []
    if os.getenv('DEEPSEEK_API_KEY'):
        available_providers.append('deepseek')
    if os.getenv('QWEN_API_KEY'):
        available_providers.append('qwen')
    if os.getenv('GLM_API_KEY'):
        available_providers.append('glm')
    
    print(f"📋 Providers with API keys: {available_providers}")
    
    for provider in available_providers:
        try:
            print(f"\n🔍 Testing {provider.upper()} model creation...")
            
            # This would create the actual model, but we'll just validate the config
            config = load_config_from_env(provider)
            print(f"✅ Configuration valid for {provider}")
            print(f"   Ready to create: {config.model}")
            
            # Get model info
            info = manager.get_model_info(provider)
            if info:
                print(f"✅ Model info available")
            else:
                print(f"ℹ️  Model not yet cached")
            
        except Exception as e:
            print(f"❌ {provider} model creation test failed: {e}")
    
    print()


def advanced_usage_example():
    """Demonstrate advanced model management features."""
    print("🎯 Advanced Model Management Example")
    print("=" * 50)
    
    manager = ModelManager()
    
    # Custom model configuration
    try:
        custom_config = ModelConfig(
            api_key="sk-example123",
            base_url="https://api.example.com",
            model="custom-model",
            temperature=0.5,
            max_tokens=2048
        )
        print("✅ Custom configuration created successfully")
        print(f"   Model: {custom_config.model}")
        print(f"   Temperature: {custom_config.temperature}")
        print(f"   Max Tokens: {custom_config.max_tokens}")
        
    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")
    
    # Model caching demonstration
    print(f"\n📊 Current cache status:")
    print(f"   Cached models: {len(manager._models)}")
    print(f"   Cached providers: {len(manager._providers)}")
    
    # Clear cache
    manager.clear_cache()
    print(f"✅ Cache cleared")
    print(f"   Cached models: {len(manager._models)}")
    
    print()


def configuration_file_example():
    """Demonstrate loading from configuration files."""
    print("📁 Configuration File Example")
    print("=" * 50)
    
    # Try to load from config files
    config_paths = [
        Path("config/config.yaml"),  # Changed from "../config/config.yaml"
        Path("config/models.yaml")   # Changed from "../config/models.yaml"
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                manager = ModelManager(config_path)
                print(f"✅ Loaded configuration from: {config_path}")
                
                models = manager.list_models()
                print(f"   Found {len(models)} models in configuration")
                
                # Show configured models
                for model in models:
                    if model['status'] == 'configured':
                        print(f"   - {model['provider']}: {model['model']}")
                
                break
                
            except Exception as e:
                print(f"❌ Failed to load {config_path}: {e}")
        else:
            print(f"⚠️  Configuration file not found: {config_path}")
    
    print()


def main():
    """Run all examples."""
    print("🎓 AI Toolkit Model Management Examples")
    print("=" * 60)
    print()
    
    examples = [
        basic_usage_example,
        configuration_example,
        model_creation_example,
        advanced_usage_example,
        configuration_file_example,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example {example.__name__} failed: {e}")
            print()
    
    print("🎉 Model Management Examples Complete!")
    print()
    print("💡 Next Steps:")
    print("   - Add credits to your API accounts to test actual model calls")
    print("   - Explore the prompt management and output parsing modules")
    print("   - Check out the integration tests for more advanced usage")


if __name__ == "__main__":
    main()