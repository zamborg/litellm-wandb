#!/usr/bin/env python3
"""
Test W&B API with various header case combinations
"""

import os
import httpx

api_key = os.environ.get('WANDB_API_KEY')
base_url = "https://api.inference.wandb.ai/v1"

print("Testing W&B API with case-sensitive header variations...\n")

# Test various case combinations
headers_to_test = [
    {"Authorization": f"Bearer {api_key}", "PROJECT": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "wandb-project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "WANDB-PROJECT": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "Wandb-Project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "W&B-Project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "x-wandb-project": "wandb/inference-demo"},
    {"Authorization": f"Bearer {api_key}", "X-WANDB-PROJECT": "wandb/inference-demo"},
]

data = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
}

for headers in headers_to_test:
    header_keys = [k for k in headers.keys() if k != "Authorization"]
    print(f"Testing header: {header_keys[0]}")
    
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                print(f"✓ SUCCESS! The correct header is: {header_keys[0]}")
                print(f"Response: {response.json()['choices'][0]['message']['content']}")
                break
            else:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                if "missing project header" in error_msg:
                    print(f"✗ Still missing project header")
                else:
                    print(f"✗ Different error: {error_msg}")
                    
    except Exception as e:
        print(f"✗ Exception: {e}")

# Let's also try sending project in the URL or as query parameter
print("\n\nTesting alternative approaches:")

# Test with project in URL path
print("\nTest: Project in URL path")
try:
    with httpx.Client() as client:
        response = client.post(
            f"{base_url}/wandb/inference-demo/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=data,
            timeout=30.0
        )
        if response.status_code == 200:
            print("✓ Success with project in URL!")
        else:
            print(f"✗ Failed: {response.status_code} - {response.json()}")
except Exception as e:
    print(f"✗ Exception: {e}")

# Test with project as query parameter
print("\nTest: Project as query parameter")
try:
    with httpx.Client() as client:
        response = client.post(
            f"{base_url}/chat/completions?project=wandb/inference-demo",
            headers={"Authorization": f"Bearer {api_key}"},
            json=data,
            timeout=30.0
        )
        if response.status_code == 200:
            print("✓ Success with project as query parameter!")
        else:
            print(f"✗ Failed: {response.status_code} - {response.json()}")
except Exception as e:
    print(f"✗ Exception: {e}")