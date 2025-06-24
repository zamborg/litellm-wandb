import json
from typing import Callable, List, Optional, Union

from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.completion.handler import OpenAITextCompletion
from litellm.types.llms.openai import AllMessageValues, OpenAITextCompletionUserMessage
from litellm.types.utils import ModelResponse
from litellm.utils import ProviderConfigManager

from ..common_utils import WandbError as OpenAIError


class WandbTextCompletion(OpenAITextCompletion):
    """
    W&B Text Completion handler that extends OpenAI's handler.
    Extracts the 'project' parameter and passes it to the OpenAI client.
    """

    def __init__(self) -> None:
        super().__init__()

    def _extract_project_param(self, optional_params: dict, headers: Optional[dict] = None) -> Optional[str]:
        """
        Extract project parameter from optional_params or headers.
        W&B requires project in format 'team/project'.
        """
        # First check optional_params
        project = optional_params.get("project")
        
        # If not found, check extra_headers
        if not project and headers:
            project = headers.get("project")
        
        # Validate project format
        if project and "/" not in project:
            raise ValueError("W&B project must be in format 'team/project'")
            
        return project

    def completion(
        self,
        model_response: ModelResponse,
        api_key: str,
        model: str,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
        timeout: float,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        print_verbose: Optional[Callable] = None,
        api_base: Optional[str] = None,
        acompletion: bool = False,
        litellm_params=None,
        logger_fn=None,
        client=None,
        organization: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        # Set default W&B API base if not provided
        if api_base is None:
            api_base = "https://api.inference.wandb.ai/v1"

        # Extract project parameter
        project = self._extract_project_param(optional_params, headers)
        
        if not project:
            raise ValueError("W&B Inference requires a 'project' parameter in the format 'team/project'. "
                           "Pass it in optional_params or extra_headers.")

        # Remove project from optional_params to avoid passing it to OpenAI API body
        optional_params_copy = optional_params.copy()
        optional_params_copy.pop("project", None)
        
        # Remove project from headers if present
        headers_copy = None
        if headers:
            headers_copy = headers.copy()
            headers_copy.pop("project", None)

        # Create OpenAI client with project parameter
        if client is None:
            if acompletion:
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=litellm.aclient_session,
                    timeout=timeout,
                    max_retries=optional_params_copy.pop("max_retries", 2),
                    organization=organization,
                    project=project,  # W&B specific parameter
                )
            else:
                client = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=litellm.client_session,
                    timeout=timeout,
                    max_retries=optional_params_copy.pop("max_retries", 2),
                    organization=organization,
                    project=project,  # W&B specific parameter
                )

        # Call the parent OpenAI handler with the project-configured client
        return super().completion(
            model_response=model_response,
            api_key=api_key,
            model=model,
            messages=messages,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            optional_params=optional_params_copy,
            print_verbose=print_verbose,
            api_base=api_base,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            client=client,
            organization=organization,
            headers=headers_copy,
        )



