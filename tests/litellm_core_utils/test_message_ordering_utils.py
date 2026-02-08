"""
Unit tests for message ordering utilities.

Tests the message ordering validation and fixing logic for Mistral models.
"""

import pytest
from litellm.litellm_core_utils.message_ordering_utils import (
    requires_strict_message_ordering,
    validate_message_ordering,
    fix_message_ordering_for_mistral,
)


class TestRequiresStrictMessageOrdering:
    """Test the requires_strict_message_ordering function."""
    
    def test_mistral_model_requires_ordering(self):
        """Test that Mistral models are detected."""
        assert requires_strict_message_ordering("mistral-large-2512") is True
        assert requires_strict_message_ordering("mistral-medium") is True
        assert requires_strict_message_ordering("mistral-small") is True
        assert requires_strict_message_ordering("Mistral-Large-2512") is True  # Case insensitive
    
    def test_watsonx_mistral_requires_ordering(self):
        """Test that WatsonX Mistral deployments are detected."""
        assert requires_strict_message_ordering("watsonx/mistral-large-2512", "watsonx") is True
        assert requires_strict_message_ordering("watsonx/mistralai/mistral-large-2512", "watsonx") is True
        assert requires_strict_message_ordering("WatsonX/Mistral-Large", "watsonx") is True  # Case insensitive
    
    def test_non_mistral_models_dont_require_ordering(self):
        """Test that non-Mistral models are not affected."""
        assert requires_strict_message_ordering("gpt-4") is False
        assert requires_strict_message_ordering("gpt-4-turbo") is False
        assert requires_strict_message_ordering("claude-3-opus") is False
        assert requires_strict_message_ordering("watsonx/granite-13b", "watsonx") is False
        assert requires_strict_message_ordering("watsonx/llama-3-70b", "watsonx") is False


class TestValidateMessageOrdering:
    """Test the validate_message_ordering function."""
    
    def test_valid_sequence_no_errors(self):
        """Test that valid sequences return no errors."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        errors = validate_message_ordering(messages, "mistral-large")
        assert len(errors) == 0
    
    def test_valid_tool_sequence_no_errors(self):
        """Test that valid tool sequences return no errors."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "assistant", "content": "The result is 25"},
        ]
        errors = validate_message_ordering(messages, "mistral-large")
        assert len(errors) == 0
    
    def test_invalid_tool_user_sequence(self):
        """Test that tool → user sequence is detected as invalid."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "user", "content": "Thanks!"},  # Invalid: user after tool
        ]
        errors = validate_message_ordering(messages, "mistral-large")
        assert len(errors) == 1
        assert "'user' cannot follow 'tool'" in errors[0]
        assert "index 3" in errors[0]
    
    def test_invalid_assistant_without_tool(self):
        """Test that assistant with tool_calls not followed by tool is detected."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "user", "content": "Never mind"},  # Invalid: user after assistant with tool_calls
        ]
        errors = validate_message_ordering(messages, "mistral-large")
        assert len(errors) == 1
        assert "'user' cannot follow 'assistant' with tool_calls" in errors[0]
        assert "index 2" in errors[0]


class TestFixMessageOrderingForMistral:
    """Test the fix_message_ordering_for_mistral function."""
    
    def test_fix_tool_user_sequence(self):
        """Test that tool → user is fixed to tool → assistant → user."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "user", "content": "Thanks!"},
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Should have one more message (the inserted assistant)
        assert len(fixed) == len(messages) + 1
        
        # Check the sequence
        assert fixed[0]["role"] == "user"
        assert fixed[1]["role"] == "assistant"
        assert fixed[2]["role"] == "tool"
        assert fixed[3]["role"] == "assistant"  # Inserted empty assistant
        assert fixed[3]["content"] == ""  # Should be empty
        assert fixed[4]["role"] == "user"
    
    def test_fix_multiple_tool_messages(self):
        """Test fixing with multiple tool messages."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared and 3 cubed"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "function": {"name": "calc"}},
                {"id": "2", "function": {"name": "calc"}}
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "tool", "tool_call_id": "2", "content": "27"},
            {"role": "user", "content": "What's the sum?"},
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Should have one more message (inserted after last tool)
        assert len(fixed) == len(messages) + 1
        
        # Check the sequence
        assert fixed[0]["role"] == "user"
        assert fixed[1]["role"] == "assistant"
        assert fixed[2]["role"] == "tool"
        assert fixed[3]["role"] == "tool"
        assert fixed[4]["role"] == "assistant"  # Inserted empty assistant
        assert fixed[4]["content"] == ""
        assert fixed[5]["role"] == "user"
    
    def test_no_change_for_valid_sequence(self):
        """Test that valid sequences are not modified."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "assistant", "content": "The result is 25"},
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Should be unchanged
        assert len(fixed) == len(messages)
        assert fixed == messages
    
    def test_preserves_message_content(self):
        """Test that all original message content is preserved."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared"},
            {"role": "assistant", "content": "Let me calculate that", "tool_calls": [{"id": "1", "function": {"name": "calc", "arguments": '{"x": 5}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "user", "content": "Thanks for the calculation!"},
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Check that original content is preserved
        assert fixed[0]["content"] == "Calculate 5 squared"
        assert fixed[1]["content"] == "Let me calculate that"
        assert fixed[1]["tool_calls"][0]["function"]["arguments"] == '{"x": 5}'
        assert fixed[2]["content"] == "25"
        assert fixed[3]["role"] == "assistant"  # Inserted
        assert fixed[3]["content"] == ""  # Empty
        assert fixed[4]["content"] == "Thanks for the calculation!"
    
    def test_multiple_tool_user_sequences(self):
        """Test fixing multiple tool → user sequences in one conversation."""
        messages = [
            {"role": "user", "content": "First calculation"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "user", "content": "Second calculation"},  # First fix needed here
            {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "2", "content": "27"},
            {"role": "user", "content": "Thanks!"},  # Second fix needed here
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Should have two more messages (two inserted assistants)
        assert len(fixed) == len(messages) + 2
        
        # Check both insertions
        assert fixed[3]["role"] == "assistant"  # First insertion
        assert fixed[3]["content"] == ""
        assert fixed[7]["role"] == "assistant"  # Second insertion
        assert fixed[7]["content"] == ""
    
    def test_empty_messages_list(self):
        """Test that empty message list is handled."""
        messages = []
        fixed = fix_message_ordering_for_mistral(messages)
        assert fixed == []
    
    def test_single_message(self):
        """Test that single message is handled."""
        messages = [{"role": "user", "content": "Hello"}]
        fixed = fix_message_ordering_for_mistral(messages)
        assert fixed == messages
    
    def test_tool_at_end_no_fix_needed(self):
        """Test that tool message at end doesn't trigger fix."""
        messages = [
            {"role": "user", "content": "Calculate"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Should be unchanged (no user after tool)
        assert len(fixed) == len(messages)
        assert fixed == messages


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_roo_code_tool_calling_scenario(self):
        """Test the exact scenario from Roo Code that was failing."""
        # This is the sequence that was causing error 3230
        messages = [
            {"role": "user", "content": "Use the calculator to compute 5 squared"},
            {"role": "assistant", "content": "I'll use the calculator tool", "tool_calls": [
                {"id": "toolu_1", "type": "function", "function": {"name": "calculator", "arguments": '{"operation": "square", "x": 5}'}}
            ]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "25"},
            {"role": "user", "content": "What's the result?"},  # This was causing the error
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Verify the fix
        assert len(fixed) == 5
        assert fixed[3]["role"] == "assistant"
        assert fixed[3]["content"] == ""
        assert fixed[4]["role"] == "user"
        
        # Verify no validation errors after fix
        errors = validate_message_ordering(fixed, "mistral-large")
        assert len(errors) == 0
    
    def test_multi_turn_tool_conversation(self):
        """Test a multi-turn conversation with multiple tool calls."""
        messages = [
            {"role": "user", "content": "Calculate 5 squared"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "25"},
            {"role": "user", "content": "Now calculate 3 cubed"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "calc"}}]},
            {"role": "tool", "tool_call_id": "2", "content": "27"},
            {"role": "user", "content": "What's the sum?"},
        ]
        
        fixed = fix_message_ordering_for_mistral(messages)
        
        # Should have 3 inserted assistants (after each tool before user)
        assert len(fixed) == len(messages) + 3
        
        # Verify no validation errors
        errors = validate_message_ordering(fixed, "mistral-large")
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
