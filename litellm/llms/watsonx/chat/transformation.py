"""
Translation from OpenAI's `/chat/completions` endpoint to IBM WatsonX's `/text/chat` endpoint.

Docs: https://cloud.ibm.com/apidocs/watsonx-ai#text-chat
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm import verbose_logger
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.watsonx import (
    WatsonXAIEndpoint,
    WatsonXModelPattern,
)
from litellm.types.utils import Choices, ModelResponse

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import IBMWatsonXMixin

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class IBMWatsonXChatConfig(IBMWatsonXMixin, OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> List:
        return [
            "temperature",  # equivalent to temperature
            "max_tokens",  # equivalent to max_new_tokens
            "top_p",  # equivalent to top_p
            "frequency_penalty",  # equivalent to repetition_penalty
            "stop",  # equivalent to stop_sequences
            "seed",  # equivalent to random_seed
            "stream",  # equivalent to stream
            "tools",
            "tool_choice",  # equivalent to tool_choice + tool_choice_option
            "logprobs",
            "top_logprobs",
            "n",
            "presence_penalty",
            "response_format",
            "reasoning_effort",
        ]

    def is_tool_choice_option(self, tool_choice: Optional[Union[str, dict]]) -> bool:
        if tool_choice is None:
            return False
        if isinstance(tool_choice, str):
            return tool_choice in ["auto", "none", "required"]
        return False

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        ## TOOLS ##
        _tools = non_default_params.pop("tools", None)
        if _tools is not None:
            # remove 'additionalProperties' from tools
            _tools = _remove_additional_properties(_tools)
            # remove 'strict' from tools
            _tools = _remove_strict_from_schema(_tools)
        if _tools is not None:
            non_default_params["tools"] = _tools

        ## TOOL CHOICE ##

        _tool_choice = non_default_params.pop("tool_choice", None)
        if self.is_tool_choice_option(_tool_choice):
            optional_params["tool_choice_option"] = _tool_choice
        elif _tool_choice is not None:
            optional_params["tool_choice"] = _tool_choice
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("HOSTED_VLLM_API_KEY") or ""
        )  # vllm does not require an api key
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        url = self._get_base_url(api_base=api_base)
        if model.startswith("deployment/"):
            deployment_id = "/".join(model.split("/")[1:])
            endpoint = (
                WatsonXAIEndpoint.DEPLOYMENT_CHAT_STREAM.value
                if stream
                else WatsonXAIEndpoint.DEPLOYMENT_CHAT.value
            )
            endpoint = endpoint.format(deployment_id=deployment_id)
        else:
            endpoint = (
                WatsonXAIEndpoint.CHAT_STREAM.value
                if stream
                else WatsonXAIEndpoint.CHAT.value
            )
        url = url.rstrip("/") + endpoint

        ## add api version
        url = self._add_api_version_to_url(
            url=url, api_version=optional_params.pop("api_version", None)
        )
        return url

    @staticmethod
    def _apply_prompt_template_core(
        model: str, messages: List[Dict[str, str]], hf_template_fn
    ) -> Optional[str]:
        """Core logic for applying prompt templates"""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            custom_prompt,
            ibm_granite_pt,
            mistral_instruct_pt,
        )

        if WatsonXModelPattern.GRANITE_CHAT.value in model:
            return ibm_granite_pt(messages=messages)
        elif WatsonXModelPattern.IBM_MISTRAL.value in model:
            return mistral_instruct_pt(messages=messages)
        elif WatsonXModelPattern.GPT_OSS.value in model:
            # Extract HuggingFace model name from watsonx/ or watsonx_text/ prefix
            if "watsonx/" in model:
                hf_model = model.split("watsonx/")[-1]
            elif "watsonx_text/" in model:
                hf_model = model.split("watsonx_text/")[-1]
            else:
                hf_model = model
            try:
                result = hf_template_fn(model=hf_model, messages=messages)
                # Return result if it's truthy (not None and not empty string)
                # The caller will handle None/empty by falling back to default
                if result:
                    return result
            except Exception:
                # Silently fall through to return None - caller will handle fallback
                pass
        elif WatsonXModelPattern.LLAMA3_INSTRUCT.value in model:
            return custom_prompt(
                role_dict={
                    "system": {
                        "pre_message": "<|start_header_id|>system<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                    "user": {
                        "pre_message": "<|start_header_id|>user<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                    "assistant": {
                        "pre_message": "<|start_header_id|>assistant<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                },
                messages=messages,
                initial_prompt_value="<|begin_of_text|>",
                final_prompt_value="<|start_header_id|>assistant<|end_header_id|>\n",
            )
        return None

    @staticmethod
    async def aapply_prompt_template(
        model: str, messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """Apply prompt template (async version)"""
        import litellm
        from litellm.litellm_core_utils.prompt_templates.factory import (
            ahf_chat_template,
            custom_prompt,
            hf_chat_template,
            ibm_granite_pt,
            mistral_instruct_pt,
        )

        if WatsonXModelPattern.GRANITE_CHAT.value in model:
            return ibm_granite_pt(messages=messages)
        elif WatsonXModelPattern.IBM_MISTRAL.value in model:
            return mistral_instruct_pt(messages=messages)
        elif WatsonXModelPattern.GPT_OSS.value in model:
            # Extract HuggingFace model name from watsonx/ or watsonx_text/ prefix
            if "watsonx/" in model:
                hf_model = model.split("watsonx/")[-1]
            elif "watsonx_text/" in model:
                hf_model = model.split("watsonx_text/")[-1]
            else:
                hf_model = model
            try:
                # Use sync if cached, async if not
                if hf_model in litellm.known_tokenizer_config:
                    result = hf_chat_template(model=hf_model, messages=messages)
                else:
                    result = await ahf_chat_template(model=hf_model, messages=messages)
                # Return result if it's truthy (not None and not empty string)
                # The caller (_aconvert_watsonx_messages_core) will handle None/empty by falling back to default
                if result:
                    return result
            except Exception as e:
                # Log the exception for debugging but don't raise it
                # The caller will fall back to default prompt factory
                try:
                    verbose_logger.debug(
                        f"Failed to apply HuggingFace template for model {hf_model}: {e}"
                    )
                except Exception:
                    # If logging fails, silently continue - don't break the flow
                    pass
        elif WatsonXModelPattern.LLAMA3_INSTRUCT.value in model:
            return custom_prompt(
                role_dict={
                    "system": {
                        "pre_message": "<|start_header_id|>system<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                    "user": {
                        "pre_message": "<|start_header_id|>user<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                    "assistant": {
                        "pre_message": "<|start_header_id|>assistant<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                },
                messages=messages,
                initial_prompt_value="<|begin_of_text|>",
                final_prompt_value="<|start_header_id|>assistant<|end_header_id|>\n",
            )
        return None

    @staticmethod
    def apply_prompt_template(
        model: str, messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """Apply prompt template (sync version)"""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            hf_chat_template,
        )

        return IBMWatsonXChatConfig._apply_prompt_template_core(
            model=model, messages=messages, hf_template_fn=hf_chat_template
        )

    @staticmethod
    def _fix_finish_reason_for_tool_calls(choice: Choices, print_verbose) -> None:
        """
        Helper to fix finish_reason for tool calls when WatsonX API returns incorrect finish_reason.
        
        WatsonX API may return "stop" for finish_reason even when tool_calls are present,
        so we need to set it to "tool_calls" when tool_calls are present.
        
        This ensures compatibility with clients like Roo Code that expect finish_reason="tool_calls"
        when tool calls are present in the response.
        """
        if (
            choice.message.tool_calls
            and len(choice.message.tool_calls) > 0
            and choice.finish_reason != "tool_calls"
        ):
            verbose_logger.debug(f"[WATSONX DEBUG] Overriding finish_reason from '{choice.finish_reason}' to 'tool_calls'")
            choice.finish_reason = "tool_calls"

    @staticmethod
    def _transform_response(
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: bool,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        api_key: Optional[str],
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
        json_mode: Optional[bool],
        custom_llm_provider: Optional[str],
        base_model: Optional[str],
    ) -> ModelResponse:
        """
        Override parent's static _transform_response to fix finish_reason for tool calls.
        
        This is called by WatsonXChatHandler.acompletion_function().
        WatsonX API may return "stop" for finish_reason even when tool_calls are present,
        so we need to override it to "tool_calls" for compatibility with clients like Roo Code.
        """
        verbose_logger.debug(f"[WATSONX DEBUG] _transform_response called for model={model}, stream={stream}")
        
        # Call parent's static method
        from ...openai_like.chat.transformation import OpenAILikeChatConfig
        model_response = OpenAILikeChatConfig._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
            json_mode=json_mode,
            custom_llm_provider=custom_llm_provider,
            base_model=base_model,
        )
        
        verbose_logger.debug(f"[WATSONX DEBUG] After parent _transform_response, finish_reason={model_response.choices[0].finish_reason if model_response.choices else 'N/A'}")
        
        # Fix finish_reason if tool_calls are present
        if model_response.choices:
            for choice in model_response.choices:
                if isinstance(choice, Choices):
                    IBMWatsonXChatConfig._fix_finish_reason_for_tool_calls(choice, print_verbose)
        
        return model_response
