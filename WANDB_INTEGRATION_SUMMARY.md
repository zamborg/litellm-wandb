# W&B (Weights & Biases) Integration Summary

This document summarizes the changes made to integrate W&B's inference API into LiteLLM.

## Overview

W&B's inference API is OpenAI-compatible with an additional required `project` parameter. The integration allows users to use W&B models through LiteLLM's unified interface.

## Changes Made

### 1. **Provider Configuration**

#### File: `litellm/constants.py`
- Added W&B API endpoint to `openai_compatible_endpoints`:
  ```python
  "api.inference.wandb.ai/v1"
  ```
- Added "wandb" to `openai_compatible_providers` list

### 2. **Model Registration**

#### File: `litellm/__init__.py`
- Added `wandb_models: List = []` to store W&B model names
- Updated `add_known_models()` function to populate `wandb_models` list

#### File: `litellm/utils.py`
- Updated `register_model()` function to handle "wandb" provider

### 3. **Provider Detection**

#### File: `litellm/litellm_core_utils/get_llm_provider_logic.py`
- Added W&B-specific logic in `_get_openai_compatible_provider_info()`:
  ```python
  elif custom_llm_provider == "wandb":
      api_base = (
          api_base
          or get_secret("WANDB_API_BASE")
          or "https://api.inference.wandb.ai/v1"
      )
      dynamic_api_key = api_key or get_secret_str("WANDB_API_KEY")
  ```
- Added endpoint detection for W&B:
  ```python
  elif endpoint == "api.inference.wandb.ai/v1":
      custom_llm_provider = "wandb"
      dynamic_api_key = get_secret_str("WANDB_API_KEY")
  ```
- Added model list check:
  ```python
  elif model in litellm.wandb_models:
      custom_llm_provider = "wandb"
  ```

### 4. **Model Configuration**

#### File: `model_prices_and_context_window.json`
Added three W&B models with their configurations:
- `wandb/Llama-3.1-8B-Instruct`
- `wandb/Llama-3.1-70B-Instruct`
- `wandb/Mistral-7B-Instruct-v0.1`

Each model includes:
- Token limits (8192 for all)
- Provider: "wandb"
- Mode: "chat"
- Capabilities (function calling, system messages)
- Note about required `project` parameter

### 5. **Documentation**

#### File: `docs/wandb_provider.md`
Comprehensive documentation including:
- Quick start example
- Configuration options
- Required parameters (especially `project`)
- Environment variables
- Advanced usage examples
- Proxy configuration
- Error handling

#### File: `examples/wandb_example.py`
Working examples demonstrating:
- Basic completion
- Streaming responses
- Multi-turn conversations
- Custom parameters
- Error handling

### 6. **Tests**

#### File: `tests/test_wandb_provider.py`
Unit tests covering:
- Provider detection
- Model registration
- Parameter passing
- API key/base configuration
- Proxy configuration structure

## Usage

### Basic Example
```python
from litellm import completion

response = completion(
    model="wandb/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="your-wandb-api-key",
    project="wandb/inference-demo",  # Required!
)
```

### With Environment Variables
```bash
export WANDB_API_KEY="your-api-key"
export WANDB_API_BASE="https://api.inference.wandb.ai/v1"  # Optional
```

```python
response = completion(
    model="wandb/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    project="wandb/inference-demo",  # Still required!
)
```

## Key Points

1. **OpenAI Compatibility**: W&B is treated as an OpenAI-compatible provider, so all standard OpenAI parameters work.

2. **Project Parameter**: The `project` parameter is **required** for all W&B API calls. This is passed through LiteLLM's parameter system.

3. **No Custom Implementation**: Since W&B is OpenAI-compatible, no custom provider implementation was needed. The integration uses the existing OpenAI client.

4. **Cost Tracking**: Currently set to 0.0 for all W&B models. Users should update these values based on their W&B pricing.

## Testing the Integration

1. Install dependencies:
   ```bash
   pip install litellm
   ```

2. Set environment variables:
   ```bash
   export WANDB_API_KEY="your-key"
   ```

3. Run a test:
   ```python
   from litellm import completion
   
   response = completion(
       model="wandb/Llama-3.1-8B-Instruct",
       messages=[{"role": "user", "content": "Test"}],
       project="wandb/inference-demo"
   )
   print(response)
   ```

## Future Improvements

1. Add more W&B models as they become available
2. Update pricing information when available
3. Add support for W&B-specific features if they extend beyond OpenAI compatibility
4. Add integration tests with actual W&B API calls