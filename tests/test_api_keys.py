#!/usr/bin/env python3
"""
Test script to verify API keys for DeepSeek, Qwen, and GLM models.
"""

import os
from dotenv import load_dotenv
import asyncio
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Print loaded environment variables (masked)
print("🔧 Environment Variables Status:")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
qwen_key = os.getenv("QWEN_API_KEY") 
glm_key = os.getenv("GLM_API_KEY")

def mask_key(key):
    if not key or len(key) < 8:
        return "❌ Not set"
    return f"✅ Set ({key[:8]}...{key[-4:]})"

print(f"   DEEPSEEK_API_KEY: {mask_key(deepseek_key)}")
print(f"   QWEN_API_KEY: {mask_key(qwen_key)}")
print(f"   GLM_API_KEY: {mask_key(glm_key)}")
print()

def test_deepseek_api():
    """Test DeepSeek API key"""
    print("🔍 Testing DeepSeek API...")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DeepSeek API key not found")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Hello, please respond with 'DeepSeek API working!'"}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ DeepSeek API working! Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ DeepSeek API error: {str(e)}")
        return False

def test_qwen_api():
    """Test Qwen API key"""
    print("\n🔍 Testing Qwen API...")
    
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("❌ Qwen API key not found")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "user", "content": "Hello, please respond with 'Qwen API working!'"}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ Qwen API working! Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Qwen API error: {str(e)}")
        return False

def test_glm_api():
    """Test GLM API key"""
    print("\n🔍 Testing GLM API...")
    
    api_key = os.getenv("GLM_API_KEY")
    if not api_key or api_key == "your_glm_api_key_here":
        print("❌ GLM API key not found or not set")
        return False
    
    try:
        from zhipuai import ZhipuAI
        
        client = ZhipuAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="glm-4.6",
            messages=[
                {"role": "user", "content": "Hello, please respond with 'GLM API working!'"}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ GLM API working! Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ GLM API error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting API Key Tests...\n")
    
    results = {
        "deepseek": test_deepseek_api(),
        "qwen": test_qwen_api(),
        "glm": test_glm_api()
    }
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    working_apis = []
    failed_apis = []
    
    for api_name, status in results.items():
        if status:
            print(f"✅ {api_name.upper()}: Working")
            working_apis.append(api_name)
        else:
            print(f"❌ {api_name.upper()}: Failed")
            failed_apis.append(api_name)
    
    print(f"\n📈 Working APIs: {len(working_apis)}/3")
    if working_apis:
        print(f"   - {', '.join(working_apis)}")
    
    if failed_apis:
        print(f"🚨 Failed APIs: {len(failed_apis)}/3")
        print(f"   - {', '.join(failed_apis)}")
        print("\n💡 Tips for failed APIs:")
        print("   - Check if API keys are correctly set in .env file")
        print("   - Verify API keys are valid and not expired")
        print("   - Check network connectivity")
        print("   - Ensure sufficient API credits/quota")
    
    return len(working_apis) == 3

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)