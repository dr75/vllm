# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import anthropic
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--served-model-name",
        "claude-3-7-sonnet-latest",
        "--enable-prompt-tokens-details",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client_anthropic() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_simple_messages(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "how are you!"}],
    )
    assert resp.stop_reason == "end_turn"
    assert resp.role == "assistant"

    print(f"Anthropic response: {resp.model_dump_json()}")


@pytest.mark.asyncio
async def test_system_message(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        system="you are a helpful assistant",
        messages=[{"role": "user", "content": "how are you!"}],
    )
    assert resp.stop_reason == "end_turn"
    assert resp.role == "assistant"

    print(f"Anthropic response: {resp.model_dump_json()}")


@pytest.mark.asyncio
async def test_anthropic_streaming(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "how are you!"}],
        stream=True,
    )

    first_chunk = None
    chunk_count = 0
    async for chunk in resp:
        chunk_count += 1
        if first_chunk is None and chunk.type == "message_start":
            first_chunk = chunk
        print(chunk.model_dump_json())

    assert chunk_count > 0
    assert first_chunk is not None, "message_start chunk was never observed"
    assert first_chunk.message is not None, "first chunk should include message"
    assert first_chunk.message.usage is not None, (
        "first chunk should include usage stats"
    )
    assert first_chunk.message.usage.output_tokens == 0
    assert first_chunk.message.usage.input_tokens > 5


@pytest.mark.asyncio
async def test_anthropic_tool_call(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What's the weather like in New York today?"}
        ],
        tools=[
            {
                "name": "get_current_weather",
                "description": "Useful for querying the weather in a specified city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region, for example: "
                            "New York, London, Tokyo, etc.",
                        }
                    },
                    "required": ["location"],
                },
            }
        ],
        stream=False,
    )
    assert resp.stop_reason == "tool_use"
    assert resp.role == "assistant"

    print(f"Anthropic response: {resp.model_dump_json()}")


@pytest.mark.asyncio
async def test_anthropic_tool_call_streaming(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in New York today?",
            }
        ],
        tools=[
            {
                "name": "get_current_weather",
                "description": "Useful for querying the weather in a specified city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region, for example: "
                            "New York, London, Tokyo, etc.",
                        }
                    },
                    "required": ["location"],
                },
            }
        ],
        stream=True,
    )

    async for chunk in resp:
        print(chunk.model_dump_json())


@pytest.mark.asyncio
async def test_anthropic_usage_cache(client: anthropic.AsyncAnthropic):
    """Test that input_tokens + cache_read_input_tokens == total prompt tokens.

    Sends the same request twice so the second hits prefix caching,
    then verifies the Anthropic usage sum invariant."""
    _LONG_PROMPT = " ".join(["Explain streaming caches."] * 5)
    messages = [{"role": "user", "content": _LONG_PROMPT}]

    # First request — populates the prefix cache
    resp1 = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=10,
        messages=messages,
    )
    assert resp1.usage is not None
    total1 = resp1.usage.input_tokens + (resp1.usage.cache_read_input_tokens or 0)
    # total1 should equal the full prompt token count
    assert total1 > 0

    # Second request — same prompt, should produce a cache hit
    resp2 = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=10,
        messages=messages,
    )
    assert resp2.usage is not None
    cached = resp2.usage.cache_read_input_tokens or 0
    assert cached > 0, "Expected cache_read_input_tokens > 0 on repeated prompt"
    # The Anthropic invariant: input_tokens + cache_read == total prompt tokens
    total2 = resp2.usage.input_tokens + cached
    assert total2 == total1, (
        f"Sum mismatch: input_tokens({resp2.usage.input_tokens}) + "
        f"cache_read({cached}) = {total2} != first request total {total1}"
    )


@pytest.mark.asyncio
async def test_anthropic_usage_cache_streaming(client: anthropic.AsyncAnthropic):
    """Streaming variant: verify cached token reporting in streamed usage.

    Note: cached token details are only available in the final usage chunk
    (message_delta), not in the initial message_start event."""
    _LONG_PROMPT = " ".join(["Explain streaming caches."] * 5)
    messages = [{"role": "user", "content": _LONG_PROMPT}]

    # First request to populate cache
    resp1 = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=10,
        messages=messages,
    )
    assert resp1.usage is not None
    total_prompt_tokens = resp1.usage.input_tokens + (
        resp1.usage.cache_read_input_tokens or 0
    )

    # Second request — streaming, should hit cache
    stream = await client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=10,
        messages=messages,
        stream=True,
    )

    message_start_usage = None
    message_delta_usage = None
    async for event in stream:
        if event.type == "message_start" and event.message:
            message_start_usage = event.message.usage
        elif event.type == "message_delta":
            message_delta_usage = getattr(event, "usage", None)

    # message_start has usage; cached token details may not be available yet
    # but the sum invariant must still hold
    assert message_start_usage is not None, "No usage in message_start"
    cached_start = message_start_usage.cache_read_input_tokens or 0
    total_start = message_start_usage.input_tokens + cached_start
    assert total_start == total_prompt_tokens, (
        f"message_start sum mismatch: {total_start} != {total_prompt_tokens}"
    )

    # Verify message_delta usage (final chunk includes cached token details)
    assert message_delta_usage is not None, "No usage in message_delta"
    cached_delta = message_delta_usage.cache_read_input_tokens or 0
    assert cached_delta > 0, "Expected cache_read_input_tokens > 0 in message_delta"
    total_delta = message_delta_usage.input_tokens + cached_delta
    assert total_delta == total_prompt_tokens, (
        f"message_delta sum mismatch: {total_delta} != {total_prompt_tokens}"
    )


@pytest.mark.asyncio
async def test_anthropic_cache_salt(client: anthropic.AsyncAnthropic):
    """Test that cache_salt is forwarded and isolates prefix caches.

    Two requests with the same prompt but different cache_salt values should
    NOT share their prefix cache."""
    _LONG_PROMPT = " ".join(["Explain streaming caches."] * 5)
    messages = [{"role": "user", "content": _LONG_PROMPT}]
    create_kwargs = dict(
        model="claude-3-7-sonnet-latest",
        max_tokens=10,
        messages=messages,
    )

    # Request 1 with salt "alpha" — populates cache
    await client.messages.create(**create_kwargs, extra_body={"cache_salt": "alpha"})

    # Request 2 with same salt "alpha" — should hit cache
    resp2 = await client.messages.create(
        **create_kwargs, extra_body={"cache_salt": "alpha"}
    )
    cached_same = resp2.usage.cache_read_input_tokens or 0
    assert cached_same > 0, "Expected cache hit with same cache_salt"

    # Request 3 with different salt "beta" — should NOT hit cache
    resp3 = await client.messages.create(
        **create_kwargs, extra_body={"cache_salt": "beta"}
    )
    cached_diff = resp3.usage.cache_read_input_tokens or 0
    assert cached_diff == 0, (
        f"Expected no cache hit with different cache_salt, "
        f"got cache_read_input_tokens={cached_diff}"
    )

    # Request 4 with invalid empty salt — should fail validation
    with pytest.raises(anthropic.BadRequestError, match="cache_salt"):
        await client.messages.create(**create_kwargs, extra_body={"cache_salt": ""})

    # Request 5 — streaming with salt "alpha", should hit existing cache
    stream1 = await client.messages.create(
        **create_kwargs, stream=True, extra_body={"cache_salt": "alpha"}
    )
    message_delta_usage = None
    async for event in stream1:
        if event.type == "message_delta":
            message_delta_usage = getattr(event, "usage", None)
    assert message_delta_usage is not None
    cached_stream = message_delta_usage.cache_read_input_tokens or 0
    assert cached_stream > 0, "Expected cache hit with same cache_salt (streaming)"

    # Request 6 — streaming with different salt "gamma", should NOT hit cache
    stream2 = await client.messages.create(
        **create_kwargs, stream=True, extra_body={"cache_salt": "gamma"}
    )
    message_delta_usage = None
    async for event in stream2:
        if event.type == "message_delta":
            message_delta_usage = getattr(event, "usage", None)
    assert message_delta_usage is not None
    cached_stream_diff = message_delta_usage.cache_read_input_tokens or 0
    assert cached_stream_diff == 0, (
        f"Expected no cache hit with different cache_salt (streaming), "
        f"got cache_read_input_tokens={cached_stream_diff}"
    )
