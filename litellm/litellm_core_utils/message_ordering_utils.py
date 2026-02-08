"""
Message Ordering Utilities for LiteLLM

This module provides utilities to validate and fix message ordering for LLM providers
that have strict message ordering requirements, particularly Mistral models.

Mistral Message Ordering Rules:
1. After a 'tool' message, the next message MUST be 'assistant'
2. After an 'assistant' message with tool_calls, the next message MUST be 'tool'
3. A 'user' message cannot directly follow a 'tool' message

This module fixes the common issue where Anthropic-format messages (with tool_result
blocks inside user messages) are transformed to OpenAI format (separate tool messages),
resulting in invalid sequences like: tool → user (which Mistral rejects with error 3230).

The fix inserts empty assistant messages to create valid sequences: tool → assistant("") → user
"""

from typing import List, Optional

from litellm._logging import verbose_logger
from litellm.types.llms.openai import ChatCompletionAssistantMessage


def requires_strict_message_ordering(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """
    Determine if a model requires strict message ordering (like Mistral).
    
    Args:
        model: The model name (e.g., "mistral-large-2512", "watsonx/mistral-large-2512")
        custom_llm_provider: The provider name (e.g., "watsonx", "mistral")
    
    Returns:
        True if the model requires strict message ordering, False otherwise
    
    Examples:
        >>> requires_strict_message_ordering("mistral-large-2512")
        True
        >>> requires_strict_message_ordering("watsonx/mistral-large-2512", "watsonx")
        True
        >>> requires_strict_message_ordering("gpt-4")
        False
    """
    model_lower = model.lower()
    
    # Check if it's a Mistral model
    if "mistral" in model_lower:
        return True
    
    # Check if it's a WatsonX Mistral deployment
    if custom_llm_provider and custom_llm_provider.lower() == "watsonx":
        if "mistral" in model_lower:
            return True
    
    return False


def validate_message_ordering(messages: List[dict], model: str) -> List[str]:
    """
    Validate message ordering and return a list of validation errors.
    
    Args:
        messages: List of message dictionaries with 'role' keys
        model: The model name for context in error messages
    
    Returns:
        List of validation error strings (empty if valid)
    
    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "", "tool_calls": [...]},
        ...     {"role": "tool", "content": "result"},
        ...     {"role": "user", "content": "Thanks"}
        ... ]
        >>> errors = validate_message_ordering(messages, "mistral-large")
        >>> print(errors)
        ["Invalid message sequence at index 3: 'user' cannot follow 'tool' for Mistral models"]
    """
    errors = []
    
    for i in range(1, len(messages)):
        prev_msg = messages[i - 1]
        curr_msg = messages[i]
        
        prev_role = prev_msg.get("role")
        curr_role = curr_msg.get("role")
        
        # Check for invalid tool → user sequence
        if prev_role == "tool" and curr_role == "user":
            errors.append(
                f"Invalid message sequence at index {i}: '{curr_role}' cannot follow "
                f"'{prev_role}' for Mistral models. An 'assistant' message is required "
                f"between 'tool' and 'user' messages."
            )
        
        # Check for invalid tool → tool sequence without assistant in between
        # (This is actually valid for multiple tool results, so we don't flag it)
        
        # Check for assistant with tool_calls not followed by tool
        if prev_role == "assistant" and prev_msg.get("tool_calls"):
            if curr_role != "tool":
                errors.append(
                    f"Invalid message sequence at index {i}: '{curr_role}' cannot follow "
                    f"'assistant' with tool_calls. A 'tool' message is expected."
                )
    
    return errors


def fix_message_ordering_for_mistral(messages: List[dict]) -> List[dict]:
    """
    Fix message ordering for Mistral models by inserting empty assistant messages
    after tool messages when followed by user messages.
    
    This function ensures that the message sequence complies with Mistral's strict
    ordering requirements:
    - Transforms: tool → user (INVALID)
    - Into: tool → assistant("") → user (VALID)
    
    Args:
        messages: List of message dictionaries to fix
    
    Returns:
        New list of messages with fixed ordering (original list is not modified)
    
    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "Calculate 5 squared"},
        ...     {"role": "assistant", "content": "", "tool_calls": [...]},
        ...     {"role": "tool", "tool_call_id": "1", "content": "25"},
        ...     {"role": "user", "content": "Thanks!"}
        ... ]
        >>> fixed = fix_message_ordering_for_mistral(messages)
        >>> # Result will have an empty assistant message inserted after the tool message
        >>> len(fixed) == len(messages) + 1
        True
        >>> fixed[3]["role"] == "assistant"
        True
        >>> fixed[3]["content"] == ""
        True
    """
    fixed_messages = []
    
    for i, msg in enumerate(messages):
        # Always add the current message
        fixed_messages.append(msg)
        
        # Check if this is a tool message
        if msg.get("role") == "tool":
            # Check if next message exists and is a user message
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                # Insert empty assistant message
                empty_assistant = ChatCompletionAssistantMessage(
                    role="assistant",
                    content=""
                )
                fixed_messages.append(empty_assistant)
                
                verbose_logger.debug(
                    f"[MESSAGE_ORDERING] Inserted empty assistant message after tool "
                    f"message at index {i} (before user message at index {i + 1})"
                )
                print(
                    f"[MESSAGE_ORDERING] Fixed ordering: Inserted empty assistant message "
                    f"after tool message at index {i}",
                    flush=True
                )
    
    return fixed_messages


def log_message_sequence(messages: List[dict], prefix: str = "") -> None:
    """
    Log the message sequence for debugging purposes.
    
    Args:
        messages: List of messages to log
        prefix: Optional prefix for log messages
    """
    if prefix:
        print(f"[MESSAGE_ORDERING] {prefix}", flush=True)
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        has_tool_calls = "tool_calls" in msg and msg["tool_calls"] is not None
        has_content = bool(msg.get("content"))
        
        content_preview = ""
        if has_content:
            content = msg.get("content", "")
            if isinstance(content, str):
                content_preview = content[:50] + "..." if len(content) > 50 else content
            else:
                content_preview = f"<{type(content).__name__}>"
        
        tool_info = " [has_tool_calls]" if has_tool_calls else ""
        content_info = f" content='{content_preview}'" if content_preview else " content=<empty>"
        
        print(
            f"[MESSAGE_ORDERING]   [{i}] role={role}{tool_info}{content_info}",
            flush=True
        )

# Made with Bob
