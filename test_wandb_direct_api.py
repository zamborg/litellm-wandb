#!/usr/bin/env python3
"""
Test direct W&B API call to understand the expected format
"""

import os
import openai

print("Testing direct W&B API call...\n")

# Create OpenAI client pointing to W&B
client = openai.OpenAI(
    api_key=os.environ['WANDB_API_KEY'],
    base_url="https://api.inference.wandb.ai/v1"
)

# Test with different model formats
test_models = [
    "meta-llama/Llama-3.1-8B-Instruct",  # Provider prefix
    "Llama-3.1-8B-Instruct",             # Without prefix
    "llama3.1:8b-instruct",              # Alternative format
]

for model in test_models:
    print(f"Testing model: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            project="wandb/inference-demo",
            max_tokens=10,
        )
        print(f"✓ Success! Response: {response.choices[0].message.content}")
        print(f"  Model in response: {response.model}\n")
        break
    except Exception as e:
        print(f"✗ Error: {e}\n")

# If successful, test what happens without project parameter
print("Testing without project parameter...")
try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )
    print("✓ Works without project parameter")
except Exception as e:
    print(f"✗ Error without project: {e}")