#!/usr/bin/env python3
"""
Test W&B integration with different header configurations
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from litellm import completion, register_model
import litellm

# Enable debug mode to see the actual requests
litellm.set_verbose = True

# Register W&B model
register_model({
    "wandb/meta-llama/Llama-3.1-8B-Instruct": {
        "max_tokens": 8192,
        "litellm_provider": "wandb",
        "mode": "chat",
    }
})

print("Testing W&B integration with headers...\n")

# Test configurations
test_configs = [
    {
        "name": "Headers with 'project' key",
        "headers": {
            "project": "wandb/inference-demo"
        }
    },
    {
        "name": "Headers with 'Project' key (capitalized)",
        "headers": {
            "Project": "wandb/inference-demo"
        }
    },
    {
        "name": "Headers with 'x-project' key",
        "headers": {
            "x-project": "wandb/inference-demo"
        }
    },
    {
        "name": "Headers with 'X-Project' key",
        "headers": {
            "X-Project": "wandb/inference-demo"
        }
    },
    {
        "name": "Headers with 'X-WandB-Project' key",
        "headers": {
            "X-WandB-Project": "wandb/inference-demo"
        }
    },
    {
        "name": "Multiple headers including project",
        "headers": {
            "project": "wandb/inference-demo",
            "X-Custom-Header": "test-value"
        }
    }
]

# Run tests
for config in test_configs:
    print(f"\nTest: {config['name']}")
    print(f"Headers: {config['headers']}")
    
    try:
        response = completion(
            model="wandb/meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Say 'Hello from W&B!'"}],
            api_key=os.environ.get('WANDB_API_KEY'),
            api_base="https://api.inference.wandb.ai/v1",
            headers=config['headers'],
            max_tokens=20,
        )
        print(f"✓ Success! Response: {response.choices[0].message.content}")
        print(f"  Headers were passed correctly!")
        break
    except Exception as e:
        error_msg = str(e)
        if "missing project header" in error_msg:
            print(f"✗ Failed: W&B still reports 'missing project header'")
        else:
            print(f"✗ Failed with error: {error_msg}")

# Also test what happens when we pass project as a parameter instead of header
print("\n\nTesting with 'project' as a parameter (not header):")
try:
    response = completion(
        model="wandb/meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Say 'Hello from W&B!'"}],
        api_key=os.environ.get('WANDB_API_KEY'),
        api_base="https://api.inference.wandb.ai/v1",
        project="wandb/inference-demo",  # As a parameter, not header
        max_tokens=20,
    )
    print("✓ Success when passed as parameter!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    error_msg = str(e)
    if "missing project header" in error_msg:
        print("✗ Failed: W&B requires project as header, not parameter")
    else:
        print(f"✗ Failed with error: {error_msg}")

print("\n\nConclusion:")
print("- W&B requires the 'project' to be passed in a specific way")
print("- The current implementation passes headers correctly via extra_headers")
print("- Need to determine the exact header name or parameter format W&B expects")