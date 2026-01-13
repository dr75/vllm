<!-- markdownlint-disable -->
# Transcriptions & translations (OpenAI-compatible)

This document explains how vLLM implements the OpenAI-compatible audio endpoints:

- `POST /v1/audio/transcriptions` (transcribe)
- `POST /v1/audio/translations` (translate)

The implementation is split across:

- `vllm/entrypoints/openai/serving_transcription.py`: endpoint-specific wrappers (`OpenAIServingTranscription` / `OpenAIServingTranslation`).
- `vllm/entrypoints/openai/speech_to_text.py`: shared speech-to-text core (`OpenAISpeechToText`).

## High-level request flow

1. FastAPI endpoints live in `vllm/entrypoints/openai/api_server.py`.
   - The request body is parsed as `Form()` data into a `TranscriptionRequest` or `TranslationRequest`.
   - The uploaded audio bytes come from `await request.file.read()`.

2. The endpoint calls into the handler:
   - Transcription: `OpenAIServingTranscription.create_transcription(audio_data, request, raw_request)`
   - Translation: `OpenAIServingTranslation.create_translation(audio_data, request, raw_request)`

3. Both wrappers delegate to the shared implementation:
   - `_create_speech_to_text(...)` in `OpenAISpeechToText`.

4. The result is returned as either:
   - a JSON response object (`json`, `text`, or `verbose_json` modes), or
   - an SSE stream (`text/event-stream`) when `request.stream=True`.

## Protocol objects and supported formats

Request models are defined in `vllm/entrypoints/openai/protocol.py`:

- `TranscriptionRequest` / `TranslationRequest` contain the `UploadFile` plus OpenAI-style parameters (e.g., `language`, `prompt`, sampling params).
- `response_format` is declared as `"json" | "text" | "srt" | "verbose_json" | "vtt"`.

Important implementation detail:

- `OpenAISpeechToText._create_speech_to_text` currently supports only `response_format in ["text", "json", "verbose_json"]`.
- `verbose_json` is not supported with streaming (`request.stream=True` returns an error).
- `srt`/`vtt` appear in the request schema but are not implemented here.

## `vllm/entrypoints/openai/serving_transcription.py`

This file contains thin wrappers that bind the shared speech-to-text pipeline to concrete OpenAI endpoints:

- `OpenAIServingTranscription(OpenAISpeechToText)`
  - Sets `task_type="transcribe"`.
  - Chooses the output class based on `request.response_format`:
    - `TranscriptionResponseVerbose` for `verbose_json`
    - `TranscriptionResponse` otherwise
  - Provides `transcription_stream_generator(...)`, which calls the shared `_speech_to_text_stream_generator(...)` with transcription-specific stream types.

- `OpenAIServingTranslation(OpenAISpeechToText)`
  - Sets `task_type="translate"`.
  - Uses `TranslationResponse*` types.
  - Provides `translation_stream_generator(...)` analogously.

These wrappers are intentionally minimal: almost all logic lives in `OpenAISpeechToText`.

## `vllm/entrypoints/openai/speech_to_text.py` (core)

### Model capability and configuration

`OpenAISpeechToText` subclasses `OpenAIServing` and uses a model-side interface:

- `SupportsTranscription` + `supports_transcription(...)` (from `vllm.model_executor.models`).
- `self.model_cls = get_model_cls(self.model_config)` (cached property).

Key model hooks expected by this pipeline:

- `get_speech_to_text_config(model_config, task_type)` → ASR config (sample rate, chunking limits, etc.).
- `validate_language(lang)` → validates/normalizes language codes.
- `get_generation_prompt(audio, stt_config, model_config, language, task_type, request_prompt, to_language)` → returns a `PromptType` used by the engine.
- Optional: `supports_segment_timestamp` and timestamp-token behavior for `verbose_json`.
- Optional: `get_num_audio_tokens(audio_duration_s, asr_config, model_config)` used to estimate prompt tokens for streaming usage reporting.

### Warmups

`__init__` performs two warmups to reduce first-request latency:

- `_warmup_audio_preprocessing()`
  - Calls `librosa.get_duration(...)` and optionally a mel-spectrogram calculation.
  - Skips if `librosa` is not installed.

- `_warmup_input_processor()`
  - Builds a dummy generation prompt (1s of silence) and runs `self.input_processor.process_inputs(...)`.

Both warmups are best-effort and non-fatal.

### Audio preprocessing and chunking

`_preprocess_speech_to_text(request, audio_data)`:

1. Validates `language` (and `to_language` if present).
2. Enforces a max upload size via `envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`.
3. Loads audio bytes with `librosa.load(BytesIO(audio_data), sr=self.asr_config.sample_rate)`.
   - Audio is resampled immediately to the model’s expected sample rate.
4. Computes audio duration with `librosa.get_duration(...)`.
5. Optionally splits long audio if:
   - `asr_config.allow_audio_chunking` is true, and
   - `duration > asr_config.max_audio_clip_s`.

Chunk splitting uses `_split_audio(...)` with an overlap window and a “quietest point” heuristic:

- `overlap_chunk_second` determines the overlap size.
- `_find_split_point(...)` scans the overlap region for the minimum RMS energy window (`min_energy_split_window_size`).

### Prompt construction

For each chunk, the model constructs the prompt via `model_cls.get_generation_prompt(...)`.

Special-case for `verbose_json`:

- The prompt must be a `dict` containing a string `decoder_prompt`.
- The code replaces `"<|notimestamps|>"` with `"<|0.00|>"` in the decoder prompt.
  - This forces timestamp tokens to appear so `_get_verbose_segments(...)` can recover segment boundaries.

### Sampling params and generation

`_create_speech_to_text(...)`:

- Checks the request/model via `_check_model(request)`.
- Computes `default_max_tokens` using `model_config.max_model_len` and `request.max_completion_tokens`.
- Builds `SamplingParams` via `request.to_sampling_params(default_max_tokens, self.default_sampling_params)`.
- Creates one engine generator per chunk:
  - `engine_client.generate(prompt, sampling_params, f"{request_id}_{i}", lora_request=...)`

Request IDs are prefixed by task type:

- `request_id = f"{task_type}-{base_request_id(raw_request)}"`

### Non-streaming response assembly

For non-streaming requests, the method iterates each chunk generator and accumulates text.

- For `text`/`json`, it concatenates `op.outputs[0].text`.
- For `verbose_json`, it constructs timestamp segments from token IDs:
  - `_get_verbose_segments(tokens, request, segment_class, start_time=...)`
  - Uses a fixed timestamp base offset: `0.02` seconds per timestamp step.
  - Uses the tokenizer to find the `<|0.00|>` token and decode segment text.
  - Segment times are offset by `start_time` when audio is chunked.

Response differences:

- Transcription (`task_type == "transcribe"`)
  - `TranscriptionResponse(text=..., usage={"type": "duration", "seconds": ceil(duration_s)})` for `json`/`text` formats.
  - `TranscriptionResponseVerbose(text, language, duration, segments)` for `verbose_json`.

- Translation (`task_type == "translate"`)
  - Similar, but does not include the `usage` duration object in the non-verbose response.

### Streaming (SSE) behavior

When `request.stream=True`, `_create_speech_to_text` returns an async generator.

`_speech_to_text_stream_generator(...)`:

- Produces OpenAI-style SSE frames:
  - `yield f"data: {chunk_json}\n\n"`
  - ends with `data: [DONE]\n\n`.
- Tracks usage stats while streaming:
  - `stream_include_usage` and `stream_continuous_usage_stats` are flattened form fields in `TranscriptionRequest`.
  - If enabled, the generator emits a final `usage`-only chunk (and optionally continuous usage per chunk).
- Writes aggregate usage back to `raw_request.state.request_metadata.final_usage_info` for middleware aggregation.

## Practical debugging tips

- If audio endpoints fail early, check whether `librosa` is installed; audio decoding/resampling depends on it.
- For `verbose_json`, verify the model supports timestamp tokens (`model_cls.supports_segment_timestamp`) and that `decoder_prompt` includes timestamps.
- If long audio behaves oddly, inspect `asr_config.max_audio_clip_s`, `overlap_chunk_second`, and `min_energy_split_window_size`.

## Known gaps / limitations

- `response_format` includes `srt` and `vtt` in the request schema, but `OpenAISpeechToText` currently rejects them.
- `timestamp_granularities` (word/segment) is accepted on the request model but is not currently used by this implementation.
