#!/usr/bin/env python3
"""
Test different authorization formats for W&B
"""

import os
import requests
import base64

api_key = os.environ['WANDB_API_KEY']

print("Testing W&B API with different auth formats...\n")

# Test different URL and auth combinations
tests = [
    {
        "name": "URL with project path",
        "url": "https://api.inference.wandb.ai/v1/wandb/inference-demo/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}"}
    },
    {
        "name": "Auth with project in bearer token",
        "url": "https://api.inference.wandb.ai/v1/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}:wandb/inference-demo"}
    },
    {
        "name": "Basic auth with project",
        "url": "https://api.inference.wandb.ai/v1/chat/completions",
        "headers": {"Authorization": f"Basic {base64.b64encode(f'{api_key}:wandb/inference-demo'.encode()).decode()}"}
    },
    {
        "name": "X-API-Key header",
        "url": "https://api.inference.wandb.ai/v1/chat/completions",
        "headers": {"X-API-Key": api_key, "X-Project": "wandb/inference-demo"}
    },
    {
        "name": "W&B specific headers",
        "url": "https://api.inference.wandb.ai/v1/chat/completions",
        "headers": {"X-WANDB-API-KEY": api_key, "X-WANDB-PROJECT": "wandb/inference-demo"}
    }
]

data = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
}

for test in tests:
    print(f"Test: {test['name']}")
    print(f"  URL: {test['url']}")
    try:
        response = requests.post(test['url'], headers=test['headers'], json=data)
        if response.status_code == 200:
            print(f"  ✓ Success! Response: {response.json()['choices'][0]['message']['content']}")
            print(f"  Working format found!\n")
            break
        else:
            print(f"  ✗ Error {response.status_code}: {response.json()}\n")
    except Exception as e:
        print(f"  ✗ Exception: {e}\n")

# Let's also check what headers W&B expects by making an OPTIONS request
print("Checking W&B API expectations with OPTIONS request...")
try:
    response = requests.options("https://api.inference.wandb.ai/v1/chat/completions")
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
except Exception as e:
    print(f"Error: {e}")