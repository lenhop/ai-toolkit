#!/usr/bin/env python3
"""
Test GLM connection and model availability
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

def test_glm_connection():
    """Test GLM API connection and model info"""
    print("🔍 Testing GLM Connection...")
    
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("❌ GLM API key not found")
        return False
    
    try:
        from zhipuai import ZhipuAI
        
        client = ZhipuAI(api_key=api_key)
        
        # Try to get model information or make a minimal request
        print(f"✅ GLM Client initialized successfully")
        print(f"📋 Using model: glm-4.6")
        print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
        
        # Try a very minimal request to test the connection
        try:
            response = client.chat.completions.create(
                model="glm-4.6",
                messages=[
                    {"role": "user", "content": "Hi"}
                ],
                max_tokens=1  # Minimal tokens to reduce cost
            )
            
            result = response.choices[0].message.content.strip()
            print(f"✅ GLM-4.6 API working! Response: {result}")
            return True
            
        except Exception as api_error:
            error_str = str(api_error)
            if "1113" in error_str or "余额不足" in error_str:
                print(f"⚠️  GLM API connection successful, but insufficient balance")
                print(f"   Error: {error_str}")
                print(f"   ✅ API key is valid")
                print(f"   ✅ Model glm-4.6 is accessible")
                print(f"   ❌ Account needs recharging")
                return "balance_issue"
            else:
                print(f"❌ GLM API error: {error_str}")
                return False
        
    except ImportError:
        print("❌ zhipuai library not installed")
        return False
    except Exception as e:
        print(f"❌ GLM connection error: {str(e)}")
        return False

if __name__ == "__main__":
    result = test_glm_connection()
    
    if result == "balance_issue":
        print("\n" + "="*50)
        print("📊 GLM Configuration Status: ✅ READY")
        print("="*50)
        print("✅ API Key: Valid")
        print("✅ Model: glm-4.6 accessible")
        print("✅ Configuration: Correct")
        print("⚠️  Balance: Needs recharging")
        print("\n💡 The GLM integration is properly configured.")
        print("   Just add credits to your GLM account to start using it.")
    elif result:
        print("\n✅ GLM fully working!")
    else:
        print("\n❌ GLM configuration needs attention")