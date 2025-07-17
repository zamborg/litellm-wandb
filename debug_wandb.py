#!/usr/bin/env python3
import os
import sys

# Force use of local model cost file
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

sys.path.insert(0, os.path.abspath("."))

import litellm

# Debug: Check model cost loading
print("Total models in model_cost:", len(litellm.model_cost))
print("Model cost URL:", litellm.model_cost_map_url)

# Debug: Check if wandb models are loaded
print("\nW&B models in litellm.wandb_models:", litellm.wandb_models)
print("\nW&B models in model_cost:")
for model, config in litellm.model_cost.items():
    if config.get("litellm_provider") == "wandb":
        print(f"  - {model}: {config}")

print("\nChecking if 'wandb' is in openai_compatible_providers:", "wandb" in litellm.openai_compatible_providers)

# Check some specific models
print("\nChecking specific models:")
print("'wandb/Llama-3.1-8B-Instruct' in model_cost:", "wandb/Llama-3.1-8B-Instruct" in litellm.model_cost)