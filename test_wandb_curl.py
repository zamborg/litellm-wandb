#!/usr/bin/env python3
"""
Test W&B API with different header formats
"""

import os
import requests
import json

api_key = os.environ['WANDB_API_KEY']
base_url = "https://api.inference.wandb.ai/v1/chat/completions"

print("Testing W&B API with different header formats...\n")

# Test different header names
header_tests = [
    {"Authorization": f"Bearer {api_key}", "project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "Project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "X-Project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "X-WandB-Project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "wandb-project": "wandb/inference-demo"},
]

data = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
}

for i, headers in enumerate(header_tests):
    print(f"Test {i+1}: Headers = {list(headers.keys())}")
    try:
        response = requests.post(base_url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"✓ Success! Response: {response.json()['choices'][0]['message']['content']}")
            print(f"  Working header format: {headers}\n")
            break
        else:
            print(f"✗ Error {response.status_code}: {response.json()}\n")
    except Exception as e:
        print(f"✗ Exception: {e}\n")

# Also test if project can be in the request body
print("Testing with project in request body...")
headers = {"Authorization": f"Bearer {api_key}"}
data_with_project = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10,
    "project": "wandb/inference-demo"
}

try:
    response = requests.post(base_url, headers=headers, json=data_with_project)
    if response.status_code == 200:
        print(f"✓ Project in body works! Response: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"✗ Error {response.status_code}: {response.json()}")
except Exception as e:
    print(f"✗ Exception: {e}")