# W&B (Weights & Biases) Provider

LiteLLM supports W&B's inference API for chat completions. W&B's API is OpenAI-compatible with an additional required `project` parameter.

## Quick Start

```python
from litellm import completion

response = completion(
    model="wandb/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    api_key="<your-wandb-api-key>",
    project="wandb/inference-demo",  # Required parameter
)

print(response.choices[0].message.content)
```

## Configuration

### Environment Variables

Set your W&B credentials as environment variables:

```bash
export WANDB_API_KEY="your-api-key"
export WANDB_API_BASE="https://api.inference.wandb.ai/v1"  # Optional, this is the default
```

### Using with Environment Variables

```python
from litellm import completion
import os

# API key will be read from WANDB_API_KEY env var
response = completion(
    model="wandb/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    project="wandb/inference-demo",  # Still required
)
```

## Required Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `api_key` | string | Your W&B API key | Yes (or set WANDB_API_KEY) |
| `project` | string | W&B project name (e.g., "wandb/inference-demo") | Yes |
| `model` | string | Model name with "wandb/" prefix | Yes |

## Supported Models

- `wandb/Llama-3.1-8B-Instruct`
- `wandb/Llama-3.1-70B-Instruct`
- `wandb/Mistral-7B-Instruct-v0.1`
- Any other model available on your W&B instance

## Advanced Usage

### Custom API Base

```python
response = completion(
    model="wandb/custom-model",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="your-key",
    api_base="https://custom.wandb.ai/v1",
    project="your-org/your-project",
)
```

### Streaming Responses

```python
response = completion(
    model="wandb/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a story"}],
    api_key="your-key",
    project="wandb/inference-demo",
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### With Additional Parameters

```python
response = completion(
    model="wandb/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    api_key="your-key",
    project="wandb/inference-demo",
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
)
```

## Proxy Configuration

### config.yaml

```yaml
model_list:
  - model_name: wandb-llama-8b
    litellm_params:
      model: wandb/Llama-3.1-8B-Instruct
      api_base: https://api.inference.wandb.ai/v1
      api_key: os.environ/WANDB_API_KEY
      project: wandb/inference-demo
  
  - model_name: wandb-llama-70b
    litellm_params:
      model: wandb/Llama-3.1-70B-Instruct
      api_base: https://api.inference.wandb.ai/v1
      api_key: os.environ/WANDB_API_KEY
      project: wandb/inference-demo
```

### Using with Proxy

```bash
# Start the proxy
litellm --config config.yaml

# Make a request
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{
    "model": "wandb-llama-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Error Handling

```python
from litellm import completion
from litellm.exceptions import APIError

try:
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="invalid-key",
        project="wandb/inference-demo",
    )
except APIError as e:
    print(f"W&B API error: {e}")
```

## Notes

1. **Project Parameter**: The `project` parameter is required for all W&B inference API calls. This is a W&B-specific requirement.

2. **OpenAI Compatibility**: W&B's API is OpenAI-compatible, so most OpenAI parameters work as expected (temperature, max_tokens, etc.).

3. **Cost Tracking**: Currently, W&B models have zero cost in LiteLLM's pricing. Update the costs in your configuration if needed.

## Related Resources

- [W&B Inference Documentation](https://docs.wandb.ai/guides/inference)
- [W&B API Reference](https://docs.wandb.ai/ref/api)
- [LiteLLM Providers Overview](https://docs.litellm.ai/docs/providers)