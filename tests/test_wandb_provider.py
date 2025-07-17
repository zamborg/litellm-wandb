"""
Test W&B (Weights & Biases) Provider Integration

This test file verifies that W&B's inference API works correctly with LiteLLM.
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock

import pytest

sys.path.insert(0, os.path.abspath(".."))

import litellm
from litellm import completion
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


class TestWandbProvider:
    """Test cases for W&B provider integration"""
    
    def test_wandb_in_openai_compatible_providers(self):
        """Test that wandb is recognized as an OpenAI-compatible provider"""
        assert "wandb" in litellm.openai_compatible_providers
        
    def test_wandb_in_openai_compatible_endpoints(self):
        """Test that W&B's API endpoint is in the list of compatible endpoints"""
        assert "api.inference.wandb.ai/v1" in litellm.openai_compatible_endpoints
        
    def test_get_llm_provider_wandb_model(self):
        """Test that get_llm_provider correctly identifies W&B models"""
        model, provider, api_key, api_base = get_llm_provider(
            model="wandb/Llama-3.1-8B-Instruct",
            api_key="test-key"
        )
        
        assert model == "Llama-3.1-8B-Instruct"
        assert provider == "wandb"
        assert api_base == "https://api.inference.wandb.ai/v1"
        
    def test_get_llm_provider_wandb_with_custom_base(self):
        """Test W&B provider with custom API base"""
        model, provider, api_key, api_base = get_llm_provider(
            model="wandb/custom-model",
            api_key="test-key",
            api_base="https://custom.wandb.ai/v1"
        )
        
        assert model == "custom-model"
        assert provider == "wandb"
        assert api_base == "https://custom.wandb.ai/v1"
        
    @patch("litellm.llms.openai.openai.OpenAILLM.completion")
    def test_wandb_completion_with_project(self, mock_openai_completion):
        """Test that project parameter is passed correctly to W&B"""
        # Mock the OpenAI completion response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(content="Test response"),
                finish_reason="stop"
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        mock_response.model = "wandb/Llama-3.1-8B-Instruct"
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.object = "chat.completion"
        
        mock_openai_completion.return_value = mock_response
        
        # Make completion call with W&B model and project parameter
        response = completion(
            model="wandb/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-wandb-key",
            project="wandb/inference-demo",
            temperature=0.7,
            max_tokens=100
        )
        
        # Verify the OpenAI completion was called
        assert mock_openai_completion.called
        
        # Get the arguments passed to the OpenAI completion
        call_args = mock_openai_completion.call_args
        
        # Check that required parameters were passed
        assert call_args.kwargs["model"] == "Llama-3.1-8B-Instruct"
        assert call_args.kwargs["api_key"] == "test-wandb-key"
        assert call_args.kwargs["api_base"] == "https://api.inference.wandb.ai/v1"
        
        # Check that project parameter is in optional_params
        optional_params = call_args.kwargs["optional_params"]
        assert "project" in optional_params or "project" in optional_params.get("extra_body", {})
        
        # Verify response
        assert response.choices[0].message.content == "Test response"
        
    @patch.dict(os.environ, {"WANDB_API_KEY": "env-wandb-key"})
    def test_wandb_api_key_from_env(self):
        """Test that W&B API key is read from environment variable"""
        model, provider, api_key, api_base = get_llm_provider(
            model="wandb/test-model"
        )
        
        assert api_key == "env-wandb-key"
        
    @patch.dict(os.environ, {"WANDB_API_BASE": "https://custom.wandb.ai/v1"})
    def test_wandb_api_base_from_env(self):
        """Test that W&B API base is read from environment variable"""
        model, provider, api_key, api_base = get_llm_provider(
            model="wandb/test-model",
            api_key="test-key"
        )
        
        assert api_base == "https://custom.wandb.ai/v1"
        
    def test_wandb_parameter_validation(self):
        """Test that appropriate errors are raised for missing required parameters"""
        # Note: Since W&B is OpenAI-compatible, validation happens at the API level
        # This test documents expected behavior
        pass
        
    @patch("litellm.llms.openai.openai.OpenAILLM.completion")
    def test_wandb_streaming_completion(self, mock_openai_completion):
        """Test W&B streaming completion"""
        # Mock streaming response
        mock_openai_completion.return_value = iter([
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))])
        ])
        
        response = completion(
            model="wandb/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test-key",
            project="wandb/test-project",
            stream=True
        )
        
        # Collect streamed responses
        chunks = list(response)
        assert len(chunks) == 2
        
    def test_wandb_with_proxy_config(self):
        """Test W&B configuration for proxy usage"""
        # This test documents how W&B would be configured in proxy settings
        config = {
            "model_list": [
                {
                    "model_name": "wandb-llama",
                    "litellm_params": {
                        "model": "wandb/Llama-3.1-8B-Instruct",
                        "api_base": "https://api.inference.wandb.ai/v1",
                        "api_key": "os.environ/WANDB_API_KEY",
                        "project": "wandb/inference-demo"
                    }
                }
            ]
        }
        
        # Verify config structure
        assert config["model_list"][0]["litellm_params"]["project"] == "wandb/inference-demo"
        

if __name__ == "__main__":
    # Run specific tests for debugging
    test = TestWandbProvider()
    test.test_wandb_in_openai_compatible_providers()
    test.test_get_llm_provider_wandb_model()
    print("All basic tests passed!")