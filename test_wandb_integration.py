#!/usr/bin/env python3
"""
Test script to verify W&B integration with LiteLLM

This script tests that W&B is properly integrated as an OpenAI-compatible provider.
"""

import sys
import os

# Force use of local model cost file
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("."))

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_wandb_provider_detection():
    """Test that W&B is recognized as a provider"""
    print("Testing W&B provider detection...")
    
    # Check if wandb is in OpenAI-compatible providers
    assert "wandb" in litellm.openai_compatible_providers, "W&B not found in openai_compatible_providers"
    print("✓ W&B is in openai_compatible_providers")
    
    # Check if W&B endpoint is recognized
    assert "api.inference.wandb.ai/v1" in litellm.openai_compatible_endpoints, "W&B endpoint not found"
    print("✓ W&B endpoint is in openai_compatible_endpoints")
    
    # Test get_llm_provider function
    model, provider, api_key, api_base = get_llm_provider(
        model="wandb/Llama-3.1-8B-Instruct",
        api_key="test-key"
    )
    
    assert model == "Llama-3.1-8B-Instruct", f"Model mismatch: {model}"
    assert provider == "wandb", f"Provider mismatch: {provider}"
    assert api_base == "https://api.inference.wandb.ai/v1", f"API base mismatch: {api_base}"
    print("✓ get_llm_provider correctly identifies W&B models")
    
    print("\nAll provider detection tests passed! ✅")


def test_wandb_model_config():
    """Test that W&B models are in the configuration"""
    print("\nTesting W&B model configuration...")
    
    # Load model configuration
    import json
    with open("model_prices_and_context_window.json", "r") as f:
        model_config = json.load(f)
    
    # Check for W&B models
    wandb_models = ["wandb/Llama-3.1-8B-Instruct", "wandb/Llama-3.1-70B-Instruct", "wandb/Mistral-7B-Instruct-v0.1"]
    
    for model in wandb_models:
        assert model in model_config, f"Model {model} not found in configuration"
        assert model_config[model]["litellm_provider"] == "wandb", f"Provider mismatch for {model}"
        print(f"✓ {model} correctly configured")
    
    print("\nAll model configuration tests passed! ✅")


def test_parameter_passthrough():
    """Test that project parameter would be passed through"""
    print("\nTesting parameter passthrough...")
    
    from litellm.utils import get_optional_params
    
    # Simulate getting optional params for W&B
    non_default_params = {
        "temperature": 0.7,
        "max_tokens": 100,
        "project": "wandb/inference-demo",  # W&B-specific parameter
    }
    
    optional_params = get_optional_params(
        custom_llm_provider="wandb",
        model="wandb/Llama-3.1-8B-Instruct",
        non_default_params=non_default_params,
        optional_params={},
    )
    
    # Check that all parameters are included (OpenAI-compatible providers pass all params)
    assert "temperature" in optional_params, "temperature not in optional_params"
    assert "max_tokens" in optional_params, "max_tokens not in optional_params"
    # project might be in extra_body for OpenAI-compatible providers
    assert "project" in optional_params or "project" in optional_params.get("extra_body", {}), "project parameter not passed through"
    
    print("✓ Parameters are passed through correctly")
    print("\nParameter passthrough test passed! ✅")


def main():
    """Run all tests"""
    print("=== W&B Integration Test Suite ===\n")
    
    try:
        test_wandb_provider_detection()
        test_wandb_model_config()
        test_parameter_passthrough()
        
        print("\n🎉 All tests passed! W&B integration is working correctly.")
        print("\nTo use W&B with LiteLLM:")
        print("1. Set your WANDB_API_KEY environment variable")
        print("2. Use model names like 'wandb/Llama-3.1-8B-Instruct'")
        print("3. Always include the 'project' parameter in your requests")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()