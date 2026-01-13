#!/usr/bin/env python3
"""
Debug script to check .env file loading
"""

import os
from dotenv import load_dotenv
from pathlib import Path

print("🔍 Debugging .env file loading...")

# Check if .env file exists
env_file = Path('.env')
print(f"📁 .env file exists: {env_file.exists()}")
print(f"📍 .env file path: {env_file.absolute()}")

if env_file.exists():
    print(f"📏 .env file size: {env_file.stat().st_size} bytes")
    
    # Read raw content
    with open(env_file, 'r') as f:
        content = f.read()
    print(f"📄 .env file content preview:")
    lines = content.split('\n')[:10]  # First 10 lines
    for i, line in enumerate(lines, 1):
        if line.strip() and not line.startswith('#'):
            # Mask API keys for security
            if '=' in line:
                key, value = line.split('=', 1)
                if 'API_KEY' in key:
                    masked_value = value[:8] + '*' * (len(value) - 8) if len(value) > 8 else '*' * len(value)
                    print(f"   {i}: {key}={masked_value}")
                else:
                    print(f"   {i}: {line}")
            else:
                print(f"   {i}: {line}")
        elif line.strip():
            print(f"   {i}: {line}")

# Try loading with explicit path
print(f"\n🔄 Loading .env file...")
result = load_dotenv(dotenv_path=env_file, verbose=True)
print(f"✅ load_dotenv result: {result}")

# Check environment variables after loading
print(f"\n🔍 Environment variables after loading:")
for key in ['DEEPSEEK_API_KEY', 'QWEN_API_KEY', 'GLM_API_KEY']:
    value = os.getenv(key)
    if value:
        masked = value[:8] + '*' * (len(value) - 8) if len(value) > 8 else '*' * len(value)
        print(f"   {key}: {masked}")
    else:
        print(f"   {key}: Not found")

# Try alternative loading method
print(f"\n🔄 Trying alternative loading method...")
from dotenv import dotenv_values
config = dotenv_values(".env")
print(f"📊 dotenv_values result: {len(config)} variables loaded")
for key in ['DEEPSEEK_API_KEY', 'QWEN_API_KEY', 'GLM_API_KEY']:
    if key in config:
        value = config[key]
        masked = value[:8] + '*' * (len(value) - 8) if len(value) > 8 else '*' * len(value)
        print(f"   {key}: {masked}")
    else:
        print(f"   {key}: Not found in config")