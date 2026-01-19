"""
Simplified Model Manager Example

Shows how to use the new streamlined model management.
"""

from ai_toolkit.models import ModelManager

# Create manager
manager = ModelManager()

# Example 1: Create DeepSeek model (OpenAI-compatible)
print("=== Example 1: DeepSeek ===")
deepseek = manager.create_model("deepseek")
response = deepseek.invoke("Say hello in 5 words")
print(f"Response: {response.content}\n")

# Example 2: Create Qwen model (OpenAI-compatible)
print("=== Example 2: Qwen ===")
qwen = manager.create_model("qwen", model="qwen-turbo")
response = qwen.invoke("What is AI?")
print(f"Response: {response.content}\n")

# Example 3: Create GLM model (Native SDK)
print("=== Example 3: GLM ===")
glm = manager.create_model("glm", temperature=0.5)
response = glm.invoke("Explain Python in one sentence")
print(f"Response: {response.content}\n")

# Example 4: Get cached model
print("=== Example 4: Get Cached Model ===")
cached = manager.get_model("deepseek")
if cached:
    print("✓ Model retrieved from cache")
    response = cached.invoke("Quick test")
    print(f"Response: {response.content}\n")

# Example 5: List providers
print("=== Example 5: List Providers ===")
providers = manager.list_providers()
print(f"Supported providers: {providers}\n")

# Example 6: Streaming
print("=== Example 6: Streaming ===")
model = manager.create_model("deepseek")
print("Streaming response: ", end="", flush=True)
for chunk in model.stream("Count to 5"):
    print(chunk.content, end="", flush=True)
print("\n")
