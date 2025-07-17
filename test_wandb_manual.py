#!/usr/bin/env python3
"""
Manual test for W&B integration - registers models manually
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

import litellm
from litellm import completion
from litellm.utils import register_model

# Manually register W&B models
wandb_models = {
    "wandb/Llama-3.1-8B-Instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "wandb",
        "mode": "chat",
        "supports_function_calling": True,
        "supports_vision": False,
        "supports_system_messages": True,
    },
    "wandb/Llama-3.1-70B-Instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "wandb",
        "mode": "chat",
        "supports_function_calling": True,
        "supports_vision": False,
        "supports_system_messages": True,
    }
}

# Register the models
register_model(wandb_models)

print("=== Testing W&B Provider with Manual Registration ===\n")

# Test 1: Check if models are registered
print("1. Checking model registration:")
print(f"   W&B models in litellm.wandb_models: {litellm.wandb_models}")
print(f"   'wandb/Llama-3.1-8B-Instruct' in model_cost: {'wandb/Llama-3.1-8B-Instruct' in litellm.model_cost}")

# Test 2: Test get_llm_provider
print("\n2. Testing get_llm_provider:")
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

try:
    model, provider, api_key, api_base = get_llm_provider(
        model="wandb/Llama-3.1-8B-Instruct",
        api_key="test-key"
    )
    print(f"   Model: {model}")
    print(f"   Provider: {provider}")
    print(f"   API Base: {api_base}")
    print("   ✓ get_llm_provider works correctly")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Test completion call structure (mock)
print("\n3. Testing completion call structure:")
try:
    # This will fail without a real API key, but we can see how far it gets
    from unittest.mock import patch, Mock
    
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    
    with patch('litellm.llms.openai.openai.OpenAILLM.completion', return_value=mock_response):
        response = completion(
            model="wandb/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
            project="wandb/inference-demo",
        )
        print(f"   ✓ Completion call succeeded")
        print(f"   Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Manual registration test complete!")