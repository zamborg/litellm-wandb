from litellm import completion
import os
api_key=os.getenv('WANDB_API_KEY')

response = completion(
    model="wandb/meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    api_key=api_key,
    provider_specific_header = {
        "extra_headers": {
            "OpenAI-Project": "wandb/zubin-dump",  # or whatever header name W&B expect
        },
        "custom_llm_provider": "wandb"
    }
)
print(response)