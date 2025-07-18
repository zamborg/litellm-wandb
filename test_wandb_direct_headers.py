#!/usr/bin/env python3
"""
Test W&B API directly with OpenAI client to verify header handling
"""

import os
import openai
import httpx

print("Testing W&B API with direct OpenAI client...\n")

api_key = os.environ.get('WANDB_API_KEY')
base_url = "https://api.inference.wandb.ai/v1"

# Test 1: Using extra_headers parameter
print("Test 1: Using extra_headers in create() call")
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url
)

try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
        extra_headers={
            "project": "wandb/inference-demo"
        }
    )
    print("✓ Success with extra_headers!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Using default_headers in client initialization
print("\n\nTest 2: Using default_headers in client initialization")
client2 = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
    default_headers={
        "project": "wandb/inference-demo"
    }
)

try:
    response = client2.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )
    print("✓ Success with default_headers!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Using httpx client directly to test raw request
print("\n\nTest 3: Using httpx directly to verify header format")
headers_to_test = [
    {"Authorization": f"Bearer {api_key}", "project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "x-wandb-project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "X-WandB-Project": "wandb/inference-demo"},
]

for headers in headers_to_test:
    print(f"\nTesting headers: {list(headers.keys())}")
    try:
        client = httpx.Client()
        response = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 10
            }
        )
        if response.status_code == 200:
            print("✓ Success!")
            break
        else:
            print(f"✗ Failed with status {response.status_code}: {response.json()}")
    except Exception as e:
        print(f"✗ Exception: {e}")