"""
Support for W&B (Weights & Biases) model family 
"""

from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig


class WandbTextCompletionConfig(OpenAITextCompletionConfig):
    """
    Configuration for W&B (Weights & Biases) Inference text completion API interface.
    
    W&B Inference uses OpenAI-compatible endpoints with an additional required parameter:
    - `project` (string): Required parameter in the format 'team/project' for usage tracking
    
    Since W&B is OpenAI-compatible, we inherit all OpenAI text completion parameters and behavior.
    """

    def get_supported_openai_params(self, model: str) -> list:
        # W&B supports the same OpenAI parameters plus project
        base_params = super().get_supported_openai_params(model)
        base_params.append("project")  # W&B specific parameter
        return base_params



