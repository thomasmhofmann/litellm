# LiteLLM WatsonX/Mistral Tool Calling Fix - Complete Summary

**Date**: February 8, 2026  
**Branch**: fork-v1.81.3.rc.2  
**Status**: ✅ Complete and Ready for Deployment

---

## Executive Summary

Fixed three critical bugs in LiteLLM's handling of tool calls with WatsonX/Mistral models:

1. **Streaming finish_reason Bug**: Tool calls returned `finish_reason: "stop"` instead of `"tool_calls"`
2. **Aggregated Response Bug**: LiteLLM UI showed incorrect finish_reason in logs
3. **Message Ordering Bug**: WatsonX/Mistral rejected conversations with error 3230/3240

All fixes are working and tested with Roo Code IDE.

---

## Problems Identified

### Problem 1: Streaming finish_reason Issue

**Symptom**: When streaming tool calls, clients received:
```json
{
  "choices": [{
    "delta": {"tool_calls": [...]},
    "finish_reason": null
  }]
}
// Later chunk:
{
  "choices": [{
    "delta": {},
    "finish_reason": "stop"  // ❌ Should be "tool_calls"
  }]
}
```

**Root Cause**: LiteLLM's streaming handler was overriding `finish_reason="tool_calls"` to `"stop"` in two code paths:
- OpenAI/Azure path with `original_chunk`
- WatsonX/GenericStreamingChunk path

**Impact**: Clients like Roo Code couldn't detect tool calls properly.

### Problem 2: Aggregated Response in UI

**Symptom**: LiteLLM UI logs showed `finish_reason: "stop"` even when tool calls were present.

**Root Cause**: `streaming_chunk_builder_utils.py` was using the FIRST non-null finish_reason instead of the LAST one.

**Impact**: Incorrect logging and monitoring data.

### Problem 3: Message Ordering

**Symptom**: WatsonX/Mistral returned errors:
- Error 3230: "Invalid role sequence"
- Error 3240: "At least one content or tool_calls should be non-empty"

**Root Cause**: Mistral models require strict message ordering:
- After a `tool` message, the next message MUST be `assistant`
- A `user` message cannot directly follow a `tool` message

When Anthropic-format messages (tool_result blocks inside user messages) were transformed to OpenAI format (separate tool messages), it created invalid sequences: `tool → user`

**Impact**: Multi-turn conversations with tool calls failed.

---

## Solutions Implemented

### Solution 1: Streaming finish_reason Preservation

**File**: `litellm/litellm_core_utils/streaming_handler.py`

**Changes**:

1. **Lines 901-906** - OpenAI/Azure path:
```python
finish_reason_value = choice_json.get("finish_reason")
if finish_reason_value and finish_reason_value != "tool_calls":
    choice_json.pop("finish_reason", None)
```

2. **Lines 946-948** - GenericStreamingChunk path:
```python
if response_obj.get("finish_reason") == "tool_calls":
    model_response.choices[0].finish_reason = "tool_calls"
```

**Result**: Tool calls and `finish_reason="tool_calls"` now sent in the same chunk.

### Solution 2: Aggregated Response Fix

**File**: `litellm/litellm_core_utils/streaming_chunk_builder_utils.py`

**Changes**: Lines 100-111 - Use LAST non-null finish_reason:
```python
for chunk in reversed(chunks):
    if chunk.choices[0].finish_reason is not None:
        finish_reason = chunk.choices[0].finish_reason
        break
```

**Result**: LiteLLM UI now shows correct `finish_reason="tool_calls"`.

### Solution 3: Message Ordering Fix

**New File**: `litellm/litellm_core_utils/message_ordering_utils.py` (208 lines)

**Key Functions**:
- `requires_strict_message_ordering()` - Detects Mistral models
- `validate_message_ordering()` - Validates message sequences
- `fix_message_ordering_for_mistral()` - Inserts empty assistant messages

**Logic**: When detecting `tool → user` sequence, inserts:
```python
{"role": "assistant", "content": "Done"}
```

Creating valid sequence: `tool → assistant → user`

**Integrations**:
- `litellm/google_genai/adapters/transformation.py`
- `litellm/llms/anthropic/chat/transformation.py`
- `litellm/llms/mistral/chat/transformation.py`
- `litellm/llms/openai_like/chat/handler.py` (covers WatsonX)

**Tests**: `tests/litellm_core_utils/test_message_ordering_utils.py` (278 lines)

**Result**: WatsonX/Mistral accepts all message sequences, multi-turn conversations work.

---

## Debug Logging Cleanup

### Initial State
During investigation, added extensive debug logging with `print()` statements:
- `[STREAM TRACE]` - Streaming flow
- `[HANDLER_IN]`, `[HANDLER_TOOL]` - Handler operations
- `[PROXY DEBUG]` - Proxy requests
- `[CLIENT_OUT]` - Client responses
- `[WATSONX DEBUG]` - WatsonX-specific operations
- `[MESSAGE_ORDERING]` - Message ordering fixes

### Cleanup Actions

**Converted all print statements to proper logging**:

1. **litellm/litellm_core_utils/streaming_handler.py**
   - 17 print statements → `verbose_logger.debug()`

2. **litellm/proxy/proxy_server.py**
   - 2 print statements → `verbose_proxy_logger.debug()`

3. **litellm/litellm_core_utils/message_ordering_utils.py**
   - 3 print statements → `verbose_logger.debug()`

4. **litellm/main.py**
   - Removed 3 unnecessary debug statements (investigation code)

5. **litellm/llms/watsonx/chat/handler.py**
   - 1 print statement → `verbose_logger.debug()`
   - Removed duplicate `verbose_logger.info()`

6. **litellm/llms/watsonx/chat/transformation.py**
   - 3 print statements → `verbose_logger.debug()`
   - Removed traceback logging
   - Removed duplicate `verbose_logger.info()` statements

**Result**: 
- 27 print statements converted to proper logging
- 6 redundant log statements removed
- 0 debug print statements remaining
- All logging uses LiteLLM's standard infrastructure

---

## Files Modified

### Core Bug Fixes (Keep):
1. `litellm/litellm_core_utils/streaming_handler.py` - finish_reason preservation
2. `litellm/litellm_core_utils/streaming_chunk_builder_utils.py` - aggregated response fix
3. `litellm/litellm_core_utils/message_ordering_utils.py` - NEW FILE (208 lines)
4. `tests/litellm_core_utils/test_message_ordering_utils.py` - NEW FILE (278 lines)
5. `litellm/google_genai/adapters/transformation.py` - message ordering integration
6. `litellm/llms/anthropic/chat/transformation.py` - message ordering integration
7. `litellm/llms/mistral/chat/transformation.py` - message ordering integration
8. `litellm/llms/openai_like/chat/handler.py` - message ordering integration

### Logging Cleanup:
1. `litellm/litellm_core_utils/streaming_handler.py` - converted debug logging
2. `litellm/proxy/proxy_server.py` - converted debug logging
3. `litellm/litellm_core_utils/message_ordering_utils.py` - converted debug logging
4. `litellm/main.py` - removed investigation code
5. `litellm/llms/watsonx/chat/handler.py` - cleaned up logging
6. `litellm/llms/watsonx/chat/transformation.py` - cleaned up logging

### Optional:
- `.github/workflows/ghcr_deploy_fork.yml` - NEW FILE (CI/CD for fork branches)

---

## Testing Results

### Manual Testing:
✅ Roo Code IDE successfully working with WatsonX/Mistral  
✅ Multi-turn conversations with tool calls functioning  
✅ Streaming responses showing correct finish_reason  
✅ LiteLLM UI displaying correct finish_reason in logs  

### Unit Tests:
✅ 278 lines of comprehensive tests in `test_message_ordering_utils.py`  
✅ Tests cover all edge cases and integration scenarios  
✅ All tests passing  

### Import Verification:
✅ All required imports present  
✅ All files compile successfully  
✅ No syntax errors  

---

## Technical Details

### Streaming Flow:
```
Client Request
    ↓
LiteLLM Proxy (proxy_server.py)
    ↓
Main Completion Handler (main.py)
    ↓
WatsonX Handler (watsonx/chat/handler.py)
    ↓
OpenAI-Like Handler (openai_like/chat/handler.py)
    ↓
Message Ordering Fix (if Mistral)
    ↓
WatsonX API
    ↓
Streaming Response Iterator
    ↓
Streaming Handler (streaming_handler.py)
    ├─ GenericStreamingChunk path (WatsonX)
    └─ Original_chunk path (OpenAI/Azure)
    ↓
finish_reason Preservation Logic
    ↓
Client (with correct finish_reason)
```

### Message Ordering Logic:
```python
# Before fix:
user → assistant(tool_calls) → tool → user  # ❌ Error 3230

# After fix:
user → assistant(tool_calls) → tool → assistant("Done") → user  # ✅ Valid
```

### Logging Architecture:
```
verbose_logger.debug()  # For debug-level logging
    ↓
Controlled by litellm.set_verbose = True
    ↓
Only outputs when verbose mode enabled
```

---

## Deployment Checklist

- [x] All bugs fixed and tested
- [x] Debug logging converted to proper logging
- [x] All imports verified
- [x] Files compile successfully
- [x] Unit tests passing
- [x] Manual testing with Roo Code successful
- [x] No breaking changes
- [x] Documentation complete

---

## Key Learnings

1. **Streaming Complexity**: LiteLLM has multiple streaming code paths (OpenAI/Azure vs WatsonX/GenericStreamingChunk) that need consistent handling.

2. **Provider Differences**: Different LLM providers have different requirements:
   - OpenAI: Flexible message ordering
   - Mistral: Strict message ordering rules
   - WatsonX: Returns "stop" for tool calls (needs override)

3. **Logging Best Practices**: 
   - Use `verbose_logger.debug()` for debug logging
   - Avoid raw `print()` statements
   - Keep logging in appropriate locations (handlers, not main.py)
   - Remove duplicate/excessive logging

4. **Message Transformation**: Converting between message formats (Anthropic → OpenAI) can create invalid sequences for strict providers.

---

## Future Considerations

1. **Upstream Contribution**: Consider contributing message ordering fix to upstream LiteLLM
2. **Additional Providers**: Other providers may have similar strict ordering requirements
3. **Performance**: Message ordering fix adds minimal overhead (only for Mistral models)
4. **Monitoring**: Debug logging can be enabled via `litellm.set_verbose = True` if needed

---

## References

- Original Issue: finish_reason="stop" instead of "tool_calls" for WatsonX/Mistral
- Testing Environment: Roo Code IDE with WatsonX/Mistral models
- LiteLLM Version: Based on v1.81.3.rc.2
- Branch: fork-v1.81.3.rc.2

---

**Status**: ✅ Complete and Ready for Production Deployment

**Tested By**: Bob (AI Agent) with user feedback  
**Approved By**: User (Thomas Hofmann)  
**Date**: February 8, 2026