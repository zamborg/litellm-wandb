"""
W&B (Weights & Biases) Inference Integration for LiteLLM

Provides text completion support using W&B's OpenAI-compatible inference endpoints.
"""

from .completion.handler import WandbTextCompletion

__all__ = ["WandbTextCompletion"] 