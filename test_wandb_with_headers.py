#!/usr/bin/env python3
"""
Test W&B API with project as header
"""

import os
import openai

print("Testing W&B API with project header...\n")

# Test 1: Direct OpenAI client with headers
client = openai.OpenAI(
    api_key=os.environ['WANDB_API_KEY'],
    base_url="https://api.inference.wandb.ai/v1",
    default_headers={
        "project": "wandb/inference-demo"
    }
)

try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Say 'Hello from W&B!'"}],
        max_tokens=20,
    )
    print("✓ Direct API call successful!")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

# Test 2: Now test with LiteLLM using headers
print("Testing with LiteLLM using headers...\n")

from litellm import completion, register_model

# Register the model with the correct provider name
register_model({
    "wandb/meta-llama/Llama-3.1-8B-Instruct": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "wandb",
        "mode": "chat",
    }
})

try:
    response = completion(
        model="wandb/meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Say 'Hello from LiteLLM!'"}],
        api_key=os.environ['WANDB_API_KEY'],
        headers={
            "project": "wandb/inference-demo"
        },
        max_tokens=20,
    )
    print("✓ LiteLLM call successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Error: {e}")
    
    # Try with custom headers approach
    print("\nTrying with api_base and headers...")
    try:
        response = completion(
            model="wandb/meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Say 'Hello from LiteLLM!'"}],
            api_key=os.environ['WANDB_API_KEY'],
            api_base="https://api.inference.wandb.ai/v1",
            headers={
                "project": "wandb/inference-demo"
            },
            max_tokens=20,
        )
        print("✓ LiteLLM call with api_base successful!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"✗ Error: {e}")