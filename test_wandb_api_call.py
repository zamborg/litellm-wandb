#!/usr/bin/env python3
"""
Test W&B API integration with actual API calls
"""

import os
from litellm import completion, register_model

# Register W&B model
register_model({
    "wandb/Llama-3.1-8B-Instruct": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "wandb",
        "mode": "chat",
    }
})

print("Testing W&B API integration with actual calls...\n")

# Test 1: Basic completion
print("1. Testing basic completion...")
try:
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Say 'Hello from W&B!' in exactly 5 words."}],
        api_key=os.environ['WANDB_API_KEY'],
        project="wandb/inference-demo",
        max_tokens=50,
        temperature=0.7,
    )
    print("✓ Basic completion successful!")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model used: {response.model}")
    print(f"Tokens used: {response.usage.total_tokens}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Streaming completion
print("\n2. Testing streaming completion...")
try:
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        api_key=os.environ['WANDB_API_KEY'],
        project="wandb/inference-demo",
        stream=True,
        max_tokens=50,
    )
    print("✓ Streaming response: ", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: System message support
print("\n3. Testing system message support...")
try:
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        api_key=os.environ['WANDB_API_KEY'],
        project="wandb/inference-demo",
        max_tokens=100,
    )
    print("✓ System message support works!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Parameters passthrough
print("\n4. Testing various parameters...")
try:
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Generate a random number between 1 and 10"}],
        api_key=os.environ['WANDB_API_KEY'],
        project="wandb/inference-demo",
        temperature=0.9,
        top_p=0.95,
        max_tokens=20,
        presence_penalty=0.5,
        frequency_penalty=0.5,
    )
    print("✓ Parameters passed through successfully!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n✅ W&B API integration test complete!")