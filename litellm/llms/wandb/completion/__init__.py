"""
W&B Text Completion Module

Provides text completion handlers and transformations for W&B Inference.
"""

from .handler import WandbTextCompletion
from .transformation import WandbTextCompletionConfig

__all__ = ["WandbTextCompletion", "WandbTextCompletionConfig"] 