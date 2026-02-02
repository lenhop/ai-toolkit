



"""
Quick Start Example - Simplified Model Usage

Shows the easiest way to get started with ai_toolkit.
"""

import sys
import os
import time
from dotenv import load_dotenv, dotenv_values
from ai_toolkit.models import ModelManager

# 1.Load environment variables
load_dotenv()
"""
指定 .env 路径: dotenv_path (非当前目录必用)。
覆盖同名变量: override=True (强制使用 .env 配置必用)。
调试加载问题: verbose=True (变量未生效必用)。
解决中文乱码: encoding="utf-8" (包含中文必用)。
"""
# list environment variables
# 1.environ, it's a dict
os.environ 
# 2.dotenv_values
# env_vars = dotenv_values()


# Create model manager
manager = ModelManager()

# Create a model (auto-loads API key from environment - recommended)
model = manager.create_model(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    provider="deepseek",           # Provider: deepseek, qwen, or glm
    model="deepseek-chat",         # Model variant (optional)
    temperature=0.7,               # Randomness: 0.0-2.0
    max_tokens=2000                # Max response length
)

# Alternative: Provide API key directly if needed
# model = manager.create_model(
#     provider="deepseek",
#     api_key=os.environ.get('DEEPSEEK_API_KEY'),
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=2000
# )

# Simple usage
print("=== Simple Chat ===")
response = model.invoke("What is Python?")
print(f"Response: {response.content}\n")

# Streaming usage
print("=== Streaming Chat ===")
print("Response: ", end="", flush=True)
for chunk in model.stream("Count from 1 to 5"):
    print(chunk.content, end="", flush=True)
print("\n")


# construct an AIMesage
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.__dict__)
    time.sleep(5)
    print(full.text)
    print()

print(full.content_blocks)


# Multi-turn conversation
print("=== Multi-turn Conversation ===")
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="My name is Alice"),
    AIMessage(content="Nice to meet you, Alice!"),
    HumanMessage(content="What's my name?")
]

response = model.invoke(messages)
print(f"Response: {response.content}\n")


# 并行提交问题
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)

# 先完行先返回来
for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)


