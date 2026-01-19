"""
Model Usage Patterns - Different ways to create models

Shows all the ways you can use ModelManager.
"""

import os
from dotenv import load_dotenv
from ai_toolkit.models import ModelManager

# Load environment variables
load_dotenv()

# Create manager
manager = ModelManager()

print("=" * 60)
print("Model Creation Patterns")
print("=" * 60)

# Pattern 1: Simplest - Auto-load from environment
print("\n[Pattern 1] Auto-load API key from environment")
print("Code: model = manager.create_model('deepseek')")
try:
    model = manager.create_model("deepseek")
    print("✓ Model created")
except Exception as e:
    print(f"✗ Error: {e}")

# Pattern 2: Provide API key directly
print("\n[Pattern 2] Provide API key directly")
print("Code: model = manager.create_model('deepseek', api_key='sk-...')")
manager.clear_cache()
try:
    model = manager.create_model(
        "deepseek",
        api_key=os.environ.get('DEEPSEEK_API_KEY')
    )
    print("✓ Model created")
except Exception as e:
    print(f"✗ Error: {e}")

# Pattern 3: Specify model variant
print("\n[Pattern 3] Specify model variant")
print("Code: model = manager.create_model('deepseek', model='deepseek-chat')")
manager.clear_cache()
try:
    model = manager.create_model("deepseek", model="deepseek-chat")
    print("✓ Model created")
except Exception as e:
    print(f"✗ Error: {e}")

# Pattern 4: Custom parameters
print("\n[Pattern 4] Custom parameters")
print("Code: model = manager.create_model('deepseek', temperature=0.5, max_tokens=1000)")
manager.clear_cache()
try:
    model = manager.create_model(
        "deepseek",
        temperature=0.5,
        max_tokens=1000
    )
    print("✓ Model created")
except Exception as e:
    print(f"✗ Error: {e}")

# Pattern 5: All parameters
print("\n[Pattern 5] All parameters together")
print("Code: model = manager.create_model(provider='deepseek', api_key='...', model='...', temperature=0.7, max_tokens=2000)")
manager.clear_cache()
try:
    model = manager.create_model(
        provider="deepseek",
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2000
    )
    print("✓ Model created")
except Exception as e:
    print(f"✗ Error: {e}")

# Pattern 6: Named parameters (recommended for clarity)
print("\n[Pattern 6] Named parameters (most readable)")
print("Code: model = manager.create_model(provider='deepseek', model='deepseek-chat', temperature=0.7)")
manager.clear_cache()
try:
    model = manager.create_model(
        provider="deepseek",
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2000
    )
    print("✓ Model created")
except Exception as e:
    print(f"✗ Error: {e}")

# Pattern 7: Different providers
print("\n[Pattern 7] Different providers")
for provider in ["deepseek", "qwen", "glm"]:
    print(f"  - {provider}: ", end="")
    manager.clear_cache()
    try:
        # Check if API key exists
        api_key = os.environ.get(f"{provider.upper()}_API_KEY")
        if api_key:
            model = manager.create_model(provider)
            print("✓")
        else:
            print(f"⊘ (no {provider.upper()}_API_KEY)")
    except Exception as e:
        print(f"✗ {e}")

print("\n" + "=" * 60)
print("Recommended Usage")
print("=" * 60)
print("""
# Best practice: Simple and clear
from ai_toolkit.models import ModelManager

manager = ModelManager()
model = manager.create_model("deepseek")

# Or with custom parameters
model = manager.create_model(
    provider="deepseek",
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2000
)

# Use the model
response = model.invoke("Hello!")
print(response.content)
""")
