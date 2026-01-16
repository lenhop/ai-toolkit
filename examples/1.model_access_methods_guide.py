#!/usr/bin/env python3
"""
AI Model Access Methods Guide

This comprehensive guide demonstrates different methods to create and access AI models:
1. LangChain Provider-Specific Packages (ChatTongyi, ChatZhipuAI, etc.)
2. OpenAI-Compatible Interface (ChatOpenAI)
3. Native SDK (Direct API client)
4. REST API (HTTP requests)
5. LangChain init_chat_model (Universal interface)

Each method has its own advantages and use cases.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Method 1: LangChain Provider-Specific Packages
# =============================================================================

def method1_langchain_provider_packages():
    """
    Method 1: Using LangChain Provider-Specific Packages
    
    Advantages:
    - Native integration with LangChain ecosystem
    - Provider-specific features and optimizations
    - Type hints and IDE support
    - Consistent LangChain interface
    
    Disadvantages:
    - Need to install provider-specific packages
    - Different import for each provider
    - May lag behind provider API updates
    
    Key Parameters:
    - api_key: Your API key for the provider
    - model: Model name/identifier
    - temperature: Randomness (0.0-2.0, default 0.7)
    - max_tokens: Maximum response length
    - streaming: Enable token-by-token streaming
    """
    print("=" * 80)
    print("METHOD 1: LangChain Provider-Specific Packages")
    print("=" * 80)
    
    # Example 1.1: Qwen (Tongyi) using ChatTongyi
    print("\n1.1 Qwen (Tongyi Qianwen) - ChatTongyi")
    print("-" * 40)
    
    try:
        from langchain_community.chat_models.tongyi import ChatTongyi
        
        print("Package: langchain-community")
        print("Import: from langchain_community.chat_models.tongyi import ChatTongyi")
        print("\nKey Parameters:")
        print("  - dashscope_api_key: Qwen API key (or DASHSCOPE_API_KEY env var)")
        print("  - model: 'qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-long'")
        print("  - temperature: 0.0-2.0 (default 0.7)")
        print("  - top_p: 0.0-1.0 (nucleus sampling)")
        print("  - streaming: True/False")
        
        # Create model instance
        qwen_model = ChatTongyi(
            dashscope_api_key=os.getenv("QWEN_API_KEY"),
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=1000,
            streaming=False
        )
        
        print("\n✅ Model created successfully")
        print(f"   Model: {qwen_model.model_name}")
        print(f"   Type: {type(qwen_model).__name__}")
        
        # Example usage (commented to avoid API calls)
        # response = qwen_model.invoke("Hello, how are you?")
        # print(f"   Response: {response.content}")
        
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("   Install: pip install langchain-community")
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    # Example 1.2: GLM (Zhipu AI) using ChatZhipuAI
    print("\n1.2 GLM (Zhipu AI) - ChatZhipuAI")
    print("-" * 40)
    
    try:
        from langchain_community.chat_models import ChatZhipuAI
        
        print("Package: langchain-community")
        print("Import: from langchain_community.chat_models import ChatZhipuAI")
        print("\nKey Parameters:")
        print("  - api_key: GLM API key (or ZHIPUAI_API_KEY env var)")
        print("  - model: 'glm-4', 'glm-4-air', 'glm-3-turbo'")
        print("  - temperature: 0.0-1.0")
        print("  - top_p: 0.0-1.0")
        
        glm_model = ChatZhipuAI(
            api_key=os.getenv("GLM_API_KEY"),
            model="glm-4",
            temperature=0.7,
            streaming=False
        )
        
        print("\n✅ Model created successfully")
        print(f"   Model: glm-4")
        print(f"   Type: {type(glm_model).__name__}")
        
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("   Install: pip install langchain-community")
    except Exception as e:
        print(f"⚠️  Error: {e}")


# =============================================================================
# Method 2: OpenAI-Compatible Interface (ChatOpenAI)
# =============================================================================

def method2_openai_compatible():
    """
    Method 2: Using ChatOpenAI with OpenAI-Compatible APIs
    
    Advantages:
    - Single interface for multiple providers
    - Well-documented and widely used
    - Easy to switch between providers
    - Supports most OpenAI features
    
    Disadvantages:
    - Provider-specific features may not be available
    - Requires OpenAI-compatible API endpoint
    
    Key Parameters:
    - api_key: API key for the provider
    - base_url: Custom API endpoint URL
    - model: Model identifier
    - temperature: Randomness (0.0-2.0)
    - max_tokens: Maximum response length
    - timeout: Request timeout in seconds
    - max_retries: Number of retry attempts
    """
    print("\n" + "=" * 80)
    print("METHOD 2: OpenAI-Compatible Interface (ChatOpenAI)")
    print("=" * 80)
    
    try:
        from langchain_community.chat_models import ChatOpenAI
        
        print("\nPackage: langchain-community or langchain-openai")
        print("Import: from langchain_community.chat_models import ChatOpenAI")
        
        # Example 2.1: DeepSeek
        print("\n2.1 DeepSeek via ChatOpenAI")
        print("-" * 40)
        print("Key Configuration:")
        print("  - base_url: https://api.deepseek.com")
        print("  - model: 'deepseek-chat', 'deepseek-coder'")
        print("  - API key format: sk-...")
        
        deepseek_model = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000,
            timeout=60,
            max_retries=3
        )
        
        print("\n✅ DeepSeek model created")
        print(f"   Base URL: https://api.deepseek.com")
        print(f"   Model: deepseek-chat")
        
        # Example 2.2: Qwen
        print("\n2.2 Qwen via ChatOpenAI")
        print("-" * 40)
        print("Key Configuration:")
        print("  - base_url: https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("  - model: 'qwen-turbo', 'qwen-plus', 'qwen-max'")
        
        qwen_model = ChatOpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-turbo",
            temperature=0.8,
            max_tokens=1500
        )
        
        print("\n✅ Qwen model created")
        print(f"   Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1")
        print(f"   Model: qwen-turbo")
        
        # Example 2.3: GLM
        print("\n2.3 GLM via ChatOpenAI")
        print("-" * 40)
        print("Key Configuration:")
        print("  - base_url: https://open.bigmodel.cn/api/paas/v4")
        print("  - model: 'glm-4', 'glm-4-air', 'glm-3-turbo'")
        
        glm_model = ChatOpenAI(
            api_key=os.getenv("GLM_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="glm-4",
            temperature=0.9,
            max_tokens=1000
        )
        
        print("\n✅ GLM model created")
        print(f"   Base URL: https://open.bigmodel.cn/api/paas/v4")
        print(f"   Model: glm-4")
        
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("   Install: pip install langchain-openai")
    except Exception as e:
        print(f"⚠️  Error: {e}")


# =============================================================================
# Method 3: Native SDK (Direct API Client)
# =============================================================================

def method3_native_sdk():
    """
    Method 3: Using Provider's Native SDK
    
    Advantages:
    - Full access to provider-specific features
    - Latest API updates and features
    - Official support and documentation
    - Often more efficient
    
    Disadvantages:
    - Different API for each provider
    - Need to learn provider-specific SDK
    - Manual integration with LangChain if needed
    
    Key Concepts:
    - Direct API client instantiation
    - Provider-specific methods and parameters
    - Custom response handling
    """
    print("\n" + "=" * 80)
    print("METHOD 3: Native SDK (Direct API Client)")
    print("=" * 80)
    
    # Example 3.1: Zhipu AI (GLM) Native SDK
    print("\n3.1 Zhipu AI (GLM) Native SDK")
    print("-" * 40)
    
    try:
        from zhipuai import ZhipuAI
        
        print("Package: zhipuai")
        print("Install: pip install zhipuai")
        print("Import: from zhipuai import ZhipuAI")
        print("\nKey Methods:")
        print("  - client.chat.completions.create()")
        print("  - Parameters: model, messages, temperature, max_tokens, stream")
        
        # Create client
        client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
        
        print("\n✅ GLM client created")
        print(f"   Type: {type(client).__name__}")
        
        # Example usage (commented to avoid API calls)
        print("\nExample Usage:")
        print("""
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": "Hello!"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        """)
        
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("   Install: pip install zhipuai")
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    # Example 3.2: DashScope (Qwen) Native SDK
    print("\n3.2 DashScope (Qwen) Native SDK")
    print("-" * 40)
    
    try:
        import dashscope
        from dashscope import Generation
        
        print("Package: dashscope")
        print("Install: pip install dashscope")
        print("Import: from dashscope import Generation")
        print("\nKey Methods:")
        print("  - Generation.call()")
        print("  - Parameters: model, messages, api_key, result_format")
        
        # Set API key
        dashscope.api_key = os.getenv("QWEN_API_KEY")
        
        print("\n✅ DashScope configured")
        
        print("\nExample Usage:")
        print("""
        response = Generation.call(
            model='qwen-turbo',
            messages=[
                {'role': 'user', 'content': 'Hello!'}
            ],
            result_format='message'
        )
        content = response.output.choices[0].message.content
        """)
        
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("   Install: pip install dashscope")
    except Exception as e:
        print(f"⚠️  Error: {e}")


# =============================================================================
# Method 4: REST API (HTTP Requests)
# =============================================================================

def method4_rest_api():
    """
    Method 4: Using REST API with HTTP Requests
    
    Advantages:
    - No SDK dependencies
    - Full control over requests
    - Works in any environment
    - Easy to debug and monitor
    
    Disadvantages:
    - More code to write
    - Manual error handling
    - Need to handle authentication
    - No built-in retry logic
    
    Key Concepts:
    - HTTP POST requests
    - JSON payload formatting
    - Authentication headers
    - Response parsing
    """
    print("\n" + "=" * 80)
    print("METHOD 4: REST API (HTTP Requests)")
    print("=" * 80)
    
    try:
        import requests
        
        print("\nPackage: requests")
        print("Install: pip install requests")
        print("Import: import requests")
        
        # Example 4.1: DeepSeek REST API
        print("\n4.1 DeepSeek REST API")
        print("-" * 40)
        print("Endpoint: https://api.deepseek.com/chat/completions")
        print("Method: POST")
        print("Headers: Authorization: Bearer <api_key>")
        print("\nRequest Body:")
        print("""
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        """)
        
        print("\nExample Code:")
        print("""
        import requests
        
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        content = result['choices'][0]['message']['content']
        """)
        
        # Example 4.2: GLM REST API
        print("\n4.2 GLM (Zhipu AI) REST API")
        print("-" * 40)
        print("Endpoint: https://open.bigmodel.cn/api/paas/v4/chat/completions")
        print("Method: POST")
        print("Headers: Authorization: Bearer <api_key>")
        print("\nRequest Body:")
        print("""
        {
            "model": "glm-4",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        """)
        
        # Example 4.3: Qwen REST API
        print("\n4.3 Qwen (DashScope) REST API")
        print("-" * 40)
        print("Endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        print("Method: POST")
        print("Headers: Authorization: Bearer <api_key>")
        print("\nRequest Body:")
        print("""
        {
            "model": "qwen-turbo",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.8
        }
        """)
        
        print("\n✅ REST API examples provided")
        
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("   Install: pip install requests")


# =============================================================================
# Method 5: LangChain init_chat_model (Universal Interface)
# =============================================================================

def method5_init_chat_model():
    """
    Method 5: Using LangChain's init_chat_model (Universal Interface)
    
    Advantages:
    - Single unified interface for all providers
    - Automatic provider detection
    - Easy to switch providers
    - Consistent configuration
    
    Disadvantages:
    - Requires proper package installation
    - May not support all providers
    - Less control over provider-specific features
    
    Key Parameters:
    - model: Model identifier (can include provider prefix)
    - model_provider: Explicit provider name
    - api_key: API key for the provider
    - base_url: Custom endpoint (for compatible APIs)
    - temperature: Randomness parameter
    - configurable: Whether to return configurable model
    """
    print("\n" + "=" * 80)
    print("METHOD 5: LangChain init_chat_model (Universal Interface)")
    print("=" * 80)
    
    try:
        from langchain.chat_models import init_chat_model
        
        print("\nPackage: langchain")
        print("Install: pip install langchain")
        print("Import: from langchain.chat_models import init_chat_model")
        print("\nKey Features:")
        print("  - Automatic provider detection from model name")
        print("  - Unified configuration interface")
        print("  - Support for multiple providers")
        
        # Example 5.1: Using model name with provider prefix
        print("\n5.1 Using Model Name with Provider Prefix")
        print("-" * 40)
        print("Format: <provider>/<model-name>")
        print("Examples:")
        print("  - 'openai/gpt-4'")
        print("  - 'anthropic/claude-3-sonnet'")
        print("  - 'google/gemini-pro'")
        
        print("\nExample Code:")
        print("""
        model = init_chat_model(
            model="openai/gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        """)
        
        # Example 5.2: Using explicit model_provider
        print("\n5.2 Using Explicit model_provider Parameter")
        print("-" * 40)
        print("Specify provider explicitly:")
        
        print("\nExample Code:")
        print("""
        model = init_chat_model(
            model="gpt-4",
            model_provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )
        """)
        
        # Example 5.3: Custom base_url for compatible APIs
        print("\n5.3 Custom base_url for OpenAI-Compatible APIs")
        print("-" * 40)
        print("Use with providers that support OpenAI-compatible endpoints:")
        
        print("\nExample Code for DeepSeek:")
        print("""
        model = init_chat_model(
            model="deepseek-chat",
            model_provider="openai",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.7
        )
        """)
        
        print("\nExample Code for Qwen:")
        print("""
        model = init_chat_model(
            model="qwen-turbo",
            model_provider="openai",
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.8
        )
        """)
        
        # Example 5.4: Configurable models
        print("\n5.4 Configurable Models")
        print("-" * 40)
        print("Get a configurable model that can be updated:")
        
        print("\nExample Code:")
        print("""
        model = init_chat_model(
            model="gpt-4",
            configurable=True
        )
        
        # Later, reconfigure the model
        configured_model = model.with_config(
            temperature=0.9,
            max_tokens=2000
        )
        """)
        
        print("\n✅ init_chat_model examples provided")
        
    except ImportError as e:
        print(f"❌ Package not installed: {e}")
        print("   Install: pip install langchain")
    except Exception as e:
        print(f"⚠️  Error: {e}")


# =============================================================================
# Comparison and Recommendations
# =============================================================================

def comparison_and_recommendations():
    """
    Comparison of all methods and recommendations for different use cases.
    """
    print("\n" + "=" * 80)
    print("COMPARISON AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n📊 Method Comparison:")
    print("-" * 80)
    print(f"{'Method':<30} {'Ease of Use':<15} {'Flexibility':<15} {'Best For':<20}")
    print("-" * 80)
    print(f"{'1. Provider Packages':<30} {'Medium':<15} {'High':<15} {'LangChain apps':<20}")
    print(f"{'2. ChatOpenAI':<30} {'Easy':<15} {'Medium':<15} {'Multi-provider':<20}")
    print(f"{'3. Native SDK':<30} {'Medium':<15} {'Very High':<15} {'Advanced features':<20}")
    print(f"{'4. REST API':<30} {'Hard':<15} {'Very High':<15} {'Custom integration':<20}")
    print(f"{'5. init_chat_model':<30} {'Very Easy':<15} {'Medium':<15} {'Quick prototyping':<20}")
    
    print("\n💡 Recommendations:")
    print("-" * 80)
    
    print("\n🎯 Use Method 1 (Provider Packages) when:")
    print("   - Building LangChain applications")
    print("   - Need provider-specific optimizations")
    print("   - Want type hints and IDE support")
    print("   - Using single provider consistently")
    
    print("\n🎯 Use Method 2 (ChatOpenAI) when:")
    print("   - Working with OpenAI-compatible APIs")
    print("   - Need to switch between providers easily")
    print("   - Want familiar OpenAI interface")
    print("   - Building multi-provider applications")
    
    print("\n🎯 Use Method 3 (Native SDK) when:")
    print("   - Need provider-specific features")
    print("   - Want latest API capabilities")
    print("   - Building production applications")
    print("   - Performance is critical")
    
    print("\n🎯 Use Method 4 (REST API) when:")
    print("   - No SDK available for your language")
    print("   - Need full control over requests")
    print("   - Debugging API issues")
    print("   - Building custom integrations")
    
    print("\n🎯 Use Method 5 (init_chat_model) when:")
    print("   - Rapid prototyping")
    print("   - Testing different providers")
    print("   - Building provider-agnostic code")
    print("   - Simple use cases")
    
    print("\n📝 Configuration Best Practices:")
    print("-" * 80)
    print("1. Store API keys in environment variables (.env file)")
    print("2. Use configuration files for model parameters")
    print("3. Implement retry logic for production")
    print("4. Set appropriate timeouts")
    print("5. Monitor token usage and costs")
    print("6. Handle errors gracefully")
    print("7. Use streaming for long responses")
    print("8. Implement rate limiting")


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("AI MODEL ACCESS METHODS - COMPREHENSIVE GUIDE")
    print("=" * 80)
    print("\nThis guide demonstrates 5 different methods to create and access AI models.")
    print("Each method has its own advantages and use cases.")
    
    examples = [
        method1_langchain_provider_packages,
        method2_openai_compatible,
        method3_native_sdk,
        method4_rest_api,
        method5_init_chat_model,
        comparison_and_recommendations,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
    
    print("\n" + "=" * 80)
    print("GUIDE COMPLETE")
    print("=" * 80)
    print("\n📚 Additional Resources:")
    print("   - LangChain Documentation: https://python.langchain.com/docs/")
    print("   - DeepSeek API: https://platform.deepseek.com/api-docs/")
    print("   - Qwen API: https://help.aliyun.com/zh/dashscope/")
    print("   - GLM API: https://open.bigmodel.cn/dev/api")
    print("\n💡 Tip: Chec