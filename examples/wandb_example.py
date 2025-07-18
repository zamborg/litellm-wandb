#!/usr/bin/env python3
"""
W&B (Weights & Biases) Provider Example

This example demonstrates how to use W&B's inference API with LiteLLM.

Requirements:
- Set WANDB_API_KEY environment variable
- Have access to a W&B project with inference enabled
"""

import os
from litellm import completion, register_model

# Register W&B models since they may not be in the remote model list yet
register_model({
    "wandb/Llama-3.1-8B-Instruct": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "wandb",
        "mode": "chat",
    }
})

def basic_completion_example():
    """Basic example of using W&B with LiteLLM"""
    print("=== Basic W&B Completion Example ===\n")
    
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        api_key=os.getenv("WANDB_API_KEY"),  # Or pass directly
        project="wandb/inference-demo",  # Required parameter
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")
    print()


def streaming_example():
    """Example of streaming responses from W&B"""
    print("=== Streaming W&B Example ===\n")
    
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Write a haiku about machine learning"}],
        api_key=os.getenv("WANDB_API_KEY"),
        project="wandb/inference-demo",
        stream=True,
    )
    
    print("Streaming response: ", end="")
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def conversation_example():
    """Example of multi-turn conversation with W&B"""
    print("=== W&B Conversation Example ===\n")
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are neural networks?"},
    ]
    
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=messages,
        api_key=os.getenv("WANDB_API_KEY"),
        project="wandb/inference-demo",
        temperature=0.7,
        max_tokens=200,
    )
    
    print(f"Assistant: {response.choices[0].message.content}\n")
    
    # Continue the conversation
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    messages.append({"role": "user", "content": "Can you explain backpropagation?"})
    
    response2 = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=messages,
        api_key=os.getenv("WANDB_API_KEY"),
        project="wandb/inference-demo",
        temperature=0.7,
        max_tokens=200,
    )
    
    print(f"Assistant: {response2.choices[0].message.content}")
    print()


def custom_parameters_example():
    """Example using custom parameters with W&B"""
    print("=== W&B with Custom Parameters Example ===\n")
    
    response = completion(
        model="wandb/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Generate a creative product name for an AI-powered coffee maker"}],
        api_key=os.getenv("WANDB_API_KEY"),
        project="wandb/inference-demo",
        temperature=0.9,  # Higher temperature for creativity
        max_tokens=100,
        top_p=0.95,
        frequency_penalty=0.5,  # Reduce repetition
        presence_penalty=0.5,   # Encourage diversity
    )
    
    print(f"Creative response: {response.choices[0].message.content}")
    print()


def error_handling_example():
    """Example of error handling with W&B"""
    print("=== W&B Error Handling Example ===\n")
    
    from litellm.exceptions import APIError, BadRequestError
    
    try:
        # This will fail if project is not provided
        response = completion(
            model="wandb/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            api_key=os.getenv("WANDB_API_KEY"),
            # project parameter intentionally missing
        )
    except (APIError, BadRequestError) as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        print("Note: W&B requires a 'project' parameter to be specified")
    print()


def main():
    """Run all examples"""
    print("Note: This example registers W&B models locally.")
    print("In production, W&B models will be available once the model list is updated.\n")
    
    # Check if API key is set
    if not os.getenv("WANDB_API_KEY"):
        print("Error: Please set WANDB_API_KEY environment variable")
        print("export WANDB_API_KEY='your-wandb-api-key'")
        return
    
    try:
        basic_completion_example()
        streaming_example()
        conversation_example()
        custom_parameters_example()
        error_handling_example()
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()