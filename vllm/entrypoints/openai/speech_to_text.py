# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import io
import math
import time
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Literal, TypeAlias, TypeVar, cast

import numpy as np
from fastapi import Request
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    DeltaMessage,
    ErrorResponse,
    RequestResponseMetadata,
    TranscriptionResponse,
    TranscriptionResponseStreamChoice,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionStreamResponse,
    TranslationResponse,
    TranslationResponseStreamChoice,
    TranslationResponseVerbose,
    TranslationSegment,
    TranslationStreamResponse,
    UsageInfo,
    VLLMValidationError,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing, SpeechToTextRequest
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsTranscription, supports_transcription
from vllm.outputs import RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.utils.import_utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

SpeechToTextResponse: TypeAlias = TranscriptionResponse | TranslationResponse
SpeechToTextResponseVerbose: TypeAlias = (
    TranscriptionResponseVerbose | TranslationResponseVerbose
)
SpeechToTextSegment: TypeAlias = TranscriptionSegment | TranslationSegment
T = TypeVar("T", bound=SpeechToTextResponse)
V = TypeVar("V", bound=SpeechToTextResponseVerbose)
S = TypeVar("S", bound=SpeechToTextSegment)

ResponseType: TypeAlias = (
    TranscriptionResponse
    | TranslationResponse
    | TranscriptionResponseVerbose
    | TranslationResponseVerbose
)

logger = init_logger(__name__)

VOXTRAL_TIMESTAMP = "<00:00.0>"


def encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(t) for t in token_ids]


def _parse_verbose_json_mistral(
    token_ids: list[int],
    token_strs: list[str],
    *,
    start_time: float,
) -> list[tuple[float, float, str, list[int]]]:
    """Parse verbose_json segments from timestamp tags emitted as text.

    Some models emit timestamps as text tags like `<00:01.5>` and closing tags
    like `</00:01.5>` (or even variants that omit the leading `<`, e.g.
    `00:01.5>` or `/00:01.5>`). This helper is a best-effort parser that tries
    to reconstruct (start, end, text, token_ids) segments from the token stream.

    Signature is intentionally minimal to allow unit testing without a tokenizer.
    """

    def _parse_timestamp_seconds(ts_text: str) -> float | None:
        ts_text = ts_text.strip()
        if not ts_text:
            return None
        parts = ts_text.split(":")
        try:
            total = 0.0
            for p in parts[:-1]:
                total = total * 60.0 + float(int(p))
            total = total * 60.0 + float(parts[-1])
            return total
        except Exception:
            return None

    def _consume_tag_at(i: int) -> tuple[int, str, bool, float] | None:
        """If a timestamp tag starts at i, return (next_i, prefix_text, is_close, seconds)."""
        if i >= len(token_strs):
            return None

        first = token_strs[i]
        if not first:
            return None

        # Heuristic: tags will end with `>` within a short window and contain
        # both ':' and '.' to represent seconds.
        lookahead = "".join(token_strs[i : min(len(token_strs), i + 24)])
        if ">" not in lookahead or ":" not in lookahead or "." not in lookahead:
            return None

        # Tag start must be plausible: `<...`, `</...`, `/...`, or `00:...>`.
        stripped = first.lstrip()
        if not (
            "<" in first
            or stripped.startswith("/")
            or stripped[:1].isdigit()
            or first.endswith("/")
        ):
            return None

        # Split any prefix text before a tag marker; e.g. token like `.</`.
        marker_idx = -1
        for ch in ("<", "/"):
            idx = first.find(ch)
            if idx != -1 and (marker_idx == -1 or idx < marker_idx):
                marker_idx = idx
        if marker_idx == -1:
            marker_idx = 0

        prefix_text = first[:marker_idx]
        tag_pieces: list[str] = [first[marker_idx:]]
        j = i + 1
        # Keep consuming until we see a '>' in the accumulated tag text.
        while j < len(token_strs) and ">" not in "".join(tag_pieces):
            tag_pieces.append(token_strs[j])
            j += 1

        tag_text = "".join(tag_pieces)
        gt = tag_text.find(">")
        if gt == -1:
            return None
        tag_text = tag_text[: gt + 1]

        # Normalize and determine open vs close.
        normalized = tag_text.strip()
        # Handle both `<...>` and `...>` variants.
        if normalized.startswith("<"):
            normalized = normalized[1:]
        if normalized.endswith(">"):
            normalized = normalized[:-1]
        normalized = normalized.strip()

        is_close = normalized.startswith("/") or "</" in tag_text

        # Strip close marker variants like `/00:01.5`.
        normalized = normalized.lstrip("/")
        seconds = _parse_timestamp_seconds(normalized)
        if seconds is None:
            return None
        return j, prefix_text, is_close, seconds

    segments: list[tuple[float, float, str, list[int]]] = []
    current_start: float | None = None
    current_text_parts: list[str] = []
    current_text_token_ids: list[int] = []

    i = 0
    while i < len(token_strs) and i < len(token_ids):
        consumed = _consume_tag_at(i)
        if consumed is None:
            if current_start is not None:
                current_text_parts.append(token_strs[i])
                current_text_token_ids.append(token_ids[i])
            i += 1
            continue

        next_i, prefix_text, is_close, ts_seconds = consumed

        # If this tag-start token contains non-tag prefix text (e.g. `.</`),
        # keep that prefix as segment text.
        if prefix_text and current_start is not None:
            current_text_parts.append(prefix_text)

        if not is_close:
            # Opening tag.
            # If we were already in a segment, close it at this boundary.
            if current_start is not None:
                boundary = start_time + ts_seconds
                segments.append(
                    (
                        current_start,
                        boundary,
                        "".join(current_text_parts),
                        current_text_token_ids,
                    )
                )
            current_start = start_time + ts_seconds
            current_text_parts = []
            current_text_token_ids = []
        else:
            # Closing tag.
            if current_start is not None:
                end_time = start_time + ts_seconds
                segments.append(
                    (
                        current_start,
                        end_time,
                        "".join(current_text_parts),
                        current_text_token_ids,
                    )
                )
            current_start = None
            current_text_parts = []
            current_text_token_ids = []

        i = max(i + 1, next_i)

    return segments


class OpenAISpeechToText(OpenAIServing):
    """Base class for speech-to-text operations like transcription and
    translation."""

    def _append_initial_timestamp_token_ids_for_verbose_json(
        self,
        prompt_dict: dict,
    ) -> None:
        token_ids_obj = prompt_dict.get("prompt_token_ids")
        if isinstance(token_ids_obj, tuple):
            token_ids_list = list(token_ids_obj)
        elif isinstance(token_ids_obj, list):
            token_ids_list = token_ids_obj
        else:
            raise VLLMValidationError(
                "speech_to_text verbose_json: prompt_token_ids is not list/tuple "
                f"(type={type(token_ids_obj).__name__})",
                value=type(token_ids_obj).__name__,
            )

        token_ids_list.extend(encode(self.tokenizer, VOXTRAL_TIMESTAMP))
        prompt_dict["prompt_token_ids"] = token_ids_list

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        task_type: Literal["transcribe", "translate"] = "transcribe",
        log_error_stack: bool = False,
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )

        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.task_type = task_type

        self.asr_config = self.model_cls.get_speech_to_text_config(
            self.model_config, task_type
        )

        self.enable_force_include_usage = enable_force_include_usage

        self.max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
        if self.model_cls.supports_segment_timestamp:
            self.tokenizer = cast(
                PreTrainedTokenizerBase,
                get_tokenizer(
                    tokenizer_name=self.model_config.tokenizer,
                    tokenizer_mode=self.model_config.tokenizer_mode,
                ),
            )

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params,
            )

        # Warm up audio preprocessing to avoid first-request latency
        self._warmup_audio_preprocessing()
        # Warm up input processor with dummy audio
        self._warmup_input_processor()

    def _warmup_audio_preprocessing(self) -> None:
        """Warm up audio processing libraries to avoid first-request latency.

        The first call to librosa functions (load, get_duration, mel-spectrogram)
        triggers JIT compilation and library initialization which can take ~7s.
        This method warms up these operations during server initialization.
        """
        # Skip warmup if librosa is not installed (optional dependency)
        if isinstance(librosa, PlaceholderModule):
            return

        # Skip warmup if model doesn't support transcription
        if not supports_transcription(self.model_cls):
            return

        try:
            warmup_start = time.perf_counter()
            logger.info("Warming up audio preprocessing libraries...")

            # Create a minimal dummy audio (1 second of silence at target sample rate)
            dummy_audio = np.zeros(int(self.asr_config.sample_rate), dtype=np.float32)

            # Warm up librosa.load by using librosa functions on the dummy data
            # This initializes FFTW, numba JIT, and other audio processing libraries
            _ = librosa.get_duration(y=dummy_audio, sr=self.asr_config.sample_rate)

            # Warm up mel-spectrogram computation with model-specific parameters
            from vllm.transformers_utils.processor import (
                cached_processor_from_config,
            )

            processor = cached_processor_from_config(self.model_config)
            feature_extractor = None
            if hasattr(processor, "feature_extractor"):
                feature_extractor = processor.feature_extractor
            elif hasattr(processor, "audio_processor"):
                # For models like GraniteSpeech that use audio_processor
                audio_proc = processor.audio_processor
                if hasattr(audio_proc, "feature_extractor"):
                    feature_extractor = audio_proc.feature_extractor
                # If audio_processor doesn't have feature_extractor,
                # skip mel-spectrogram warmup for these models

            if feature_extractor is not None:
                _ = librosa.feature.melspectrogram(
                    y=dummy_audio,
                    sr=self.asr_config.sample_rate,
                    n_mels=getattr(feature_extractor, "n_mels", 128),
                    n_fft=getattr(feature_extractor, "n_fft", 400),
                    hop_length=getattr(feature_extractor, "hop_length", 160),
                )

            warmup_elapsed = time.perf_counter() - warmup_start
            logger.info("Audio preprocessing warmup completed in %.2fs", warmup_elapsed)
        except Exception:
            # Don't fail initialization if warmup fails - log exception and continue
            logger.exception(
                "Audio preprocessing warmup failed (non-fatal): %s. "
                "First request may experience higher latency.",
            )

    def _warmup_input_processor(self) -> None:
        """Warm up input processor with dummy audio to avoid first-request latency.

        The first call to input_processor.process_inputs() with multimodal audio
        triggers multimodal processing initialization which can take ~2.5s.
        This method processes a dummy audio request to warm up the pipeline.
        """
        # Skip warmup if model doesn't support transcription
        if not supports_transcription(self.model_cls):
            return

        # Only warm up if model supports transcription methods
        if not hasattr(self.model_cls, "get_generation_prompt"):
            return

        try:
            from vllm.sampling_params import SamplingParams

            warmup_start = time.perf_counter()
            logger.info("Warming up multimodal input processor...")

            # Create minimal dummy audio (1 second of silence)
            dummy_audio = np.zeros(int(self.asr_config.sample_rate), dtype=np.float32)

            # Use the same method that _preprocess_speech_to_text uses
            # to create the prompt
            dummy_prompt = self.model_cls.get_generation_prompt(
                audio=dummy_audio,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language="en",
                task_type=self.task_type,
                request_prompt="",
                to_language=None,
            )

            # Create minimal sampling params
            dummy_params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                skip_clone=True,  # Internal warmup, safe to skip clone
            )

            # Process the dummy input through the input processor
            # This will trigger all the multimodal processing initialization
            _ = self.input_processor.process_inputs(
                request_id="warmup",
                prompt=dummy_prompt,
                params=dummy_params,
            )

            warmup_elapsed = time.perf_counter() - warmup_start
            logger.info("Input processor warmup completed in %.2fs", warmup_elapsed)
        except Exception:
            # Don't fail initialization if warmup fails - log warning and continue
            logger.exception(
                "Input processor warmup failed (non-fatal): %s. "
                "First request may experience higher latency."
            )

    @cached_property
    def model_cls(self) -> type[SupportsTranscription]:
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsTranscription], model_cls)

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
    ) -> tuple[list[PromptType], float]:
        # Validate request
        language = self.model_cls.validate_language(request.language)
        # Skip to_language validation to avoid extra logging for Whisper.
        to_language = (
            self.model_cls.validate_language(request.to_language)
            if request.to_language
            else None
        )

        if len(audio_data) / 1024**2 > self.max_audio_filesize_mb:
            raise VLLMValidationError(
                "Maximum file size exceeded",
                parameter="audio_filesize_mb",
                value=len(audio_data) / 1024**2,
            )

        with io.BytesIO(audio_data) as bytes_:
            # NOTE resample to model SR here for efficiency. This is also a
            # pre-requisite for chunking, as it assumes Whisper SR.
            y, sr = librosa.load(bytes_, sr=self.asr_config.sample_rate)

        duration = librosa.get_duration(y=y, sr=sr)
        do_split_audio = (
            self.asr_config.allow_audio_chunking
            and duration > self.asr_config.max_audio_clip_s
        )
        chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
        prompts = []
        for chunk in chunks:
            # The model has control over the construction, as long as it
            # returns a valid PromptType.
            prompt = self.model_cls.get_generation_prompt(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language=language,
                task_type=self.task_type,
                request_prompt=request.prompt,
                to_language=to_language,
            )
            if request.response_format == "verbose_json":
                if not isinstance(prompt, dict):
                    raise VLLMValidationError(
                        "Expected prompt to be a dict",
                        parameter="prompt",
                        value=type(prompt).__name__,
                    )
                prompt_dict = cast(dict, prompt)
                decoder_prompt = prompt.get("decoder_prompt")
                if not isinstance(decoder_prompt, str):
                    self._append_initial_timestamp_token_ids_for_verbose_json(
                        prompt_dict
                    )
                else:
                    prompt_dict["decoder_prompt"] = decoder_prompt.replace(
                        "<|notimestamps|>", "<|0.00|>"
                    )
            prompts.append(prompt)
        return prompts, duration

    def _get_verbose_segments(
        self,
        tokens: tuple,
        request: SpeechToTextRequest,
        segment_class: type[SpeechToTextSegment],
        start_time: float = 0,
    ) -> list[SpeechToTextSegment]:
        """
        Convert tokens to verbose segments.

        This method expects the model to produce
        timestamps as tokens (similar to Whisper).
        If the tokens do not include timestamp information,
        the segments may not be generated correctly.

        Note: Fields like avg_logprob, compression_ratio,
        and no_speech_prob are not supported
        in this implementation and will be None. See docs for details.
        """
        BASE_OFFSET = 0.02
        if not tokens:
            return []

        token_ids_list = [int(t) for t in tokens if isinstance(t, (int, np.integer))]
        token_strs: list[str] | None = None
        token_strs_raw = self.tokenizer.convert_ids_to_tokens(token_ids_list)
        if isinstance(token_strs_raw, str):
            token_strs = [token_strs_raw]
        elif isinstance(token_strs_raw, list):
            token_strs = [str(t) for t in token_strs_raw]
        else:
            token_strs = [str(token_strs_raw)]

        # Remove EOS if present.
        if tokens[-1] == self.tokenizer.eos_token_id:
            tokens = tokens[:-1]

        init_token_ids = encode(self.tokenizer, VOXTRAL_TIMESTAMP)
        if not init_token_ids:
            logger.warning(
                "verbose_json segment reconstruction: tokenizer cannot encode "
                "<|0.00|>; returning empty segments (model=%s)",
                request.model,
            )
            return []

        # Some tokenizers (e.g. sentencepiece variants) treat timestamp markers
        # as plain text and split them across multiple token IDs. In that case,
        # parse timestamps from text tags like `<00:01.5>` and `</00:01.5>`.
        if len(init_token_ids) != 1:
            if token_strs is None:
                convert_ids_to_tokens = getattr(
                    self.tokenizer, "convert_ids_to_tokens", None
                )
                if not callable(convert_ids_to_tokens):
                    logger.warning(
                        "verbose_json segment reconstruction: tokenizer has no convert_ids_to_tokens; cannot parse timestamp tags (model=%s)",
                        request.model,
                    )
                    return []

                token_strs_raw = convert_ids_to_tokens(token_ids_list)
                if isinstance(token_strs_raw, str):
                    token_strs = [token_strs_raw]
                elif isinstance(token_strs_raw, list):
                    token_strs = [str(t) for t in token_strs_raw]
                else:
                    token_strs = [str(token_strs_raw)]

                if len(token_strs) != len(token_ids_list):
                    token_strs = [
                        str(convert_ids_to_tokens(tid)) for tid in token_ids_list
                    ]

            spans = _parse_verbose_json_mistral(
                init_token_ids + token_ids_list,
                token_strs,
                start_time=start_time,
            )

            segments: list[SpeechToTextSegment] = []
            for seg_start, seg_end, seg_text, seg_token_ids in spans:
                segments.append(
                    cast(
                        SpeechToTextSegment,
                        segment_class(
                            id=len(segments),
                            seek=int(start_time),
                            start=seg_start,
                            end=seg_end,
                            temperature=request.temperature,
                            text=seg_text,
                            tokens=seg_token_ids,
                        ),
                    )
                )
            return segments

        if tokens[-1] == self.tokenizer.eos_token_id:
            tokens = tokens[:-1]

        tokens_with_start = (init_token,) + tokens
        segments: list[SpeechToTextSegment] = []
        last_timestamp_start = 0

        if tokens_with_start[-2] < init_token and tokens_with_start[-1] >= init_token:
            tokens_with_start = tokens_with_start + (tokens_with_start[-1],)
        for idx, token in enumerate(tokens_with_start):
            # Timestamp tokens (e.g., <|0.00|>) are assumed to be sorted.
            # If the ordering is violated, this slicing may produce incorrect results.
            if (
                token >= init_token
                and idx != 0
                and tokens_with_start[idx - 1] >= init_token
            ):
                sliced_timestamp_tokens = tokens_with_start[last_timestamp_start:idx]
                start_timestamp = sliced_timestamp_tokens[0] - init_token
                end_timestamp = sliced_timestamp_tokens[-1] - init_token

                casting_segment = cast(
                    SpeechToTextSegment,
                    segment_class(
                        id=len(segments),
                        seek=start_time,
                        start=start_time + BASE_OFFSET * start_timestamp,
                        end=start_time + BASE_OFFSET * end_timestamp,
                        temperature=request.temperature,
                        text=self.tokenizer.decode(sliced_timestamp_tokens[1:-1]),
                        tokens=sliced_timestamp_tokens[1:-1],
                    ),
                )
                segments.append(casting_segment)
                last_timestamp_start = idx
        return segments

    async def _create_speech_to_text(
        self,
        audio_data: bytes,
        request: SpeechToTextRequest,
        raw_request: Request,
        response_class: type[T | V],
        stream_generator_method: Callable[..., AsyncGenerator[str, None]],
    ) -> T | V | AsyncGenerator[str, None] | ErrorResponse:
        """Base method for speech-to-text operations like transcription and
        translation."""
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.response_format not in ["text", "json", "verbose_json"]:
            return self.create_error_response(
                ("Currently only support response_format")
                + ("`text`, `json` or `verbose_json`")
            )

        if (
            request.response_format == "verbose_json"
            and not self.model_cls.supports_segment_timestamp
        ):
            return self.create_error_response(
                f"Currently do not support verbose_json for {request.model}"
            )

        if request.response_format == "verbose_json" and request.stream:
            return self.create_error_response(
                "verbose_json format doesn't support streaming case"
            )
        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(request)

            prompts, duration_s = await self._preprocess_speech_to_text(
                request=request,
                audio_data=audio_data,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(e)

        list_result_generator: list[AsyncGenerator[RequestOutput, None]] | None = None
        try:
            # Unlike most decoder-only models, whisper generation length is not
            # constrained by the size of the input audio, which is mapped to a
            # fixed-size log-mel-spectogram. Still, allow for fewer tokens to be
            # generated by respecting the extra completion tokens arg.
            if request.max_completion_tokens is None:
                default_max_tokens = self.model_config.max_model_len
            else:
                default_max_tokens = min(
                    self.model_config.max_model_len, request.max_completion_tokens
                )
            sampling_params = request.to_sampling_params(
                default_max_tokens, self.default_sampling_params
            )

            self._log_inputs(
                request_id,
                # It will not display special tokens like <|startoftranscript|>
                request.prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            list_result_generator = [
                self.engine_client.generate(
                    prompt,
                    sampling_params,
                    f"{request_id}_{i}",
                    lora_request=lora_request,
                )
                for i, prompt in enumerate(prompts)
            ]
        except ValueError as e:
            return self.create_error_response(e)

        if request.stream:
            return stream_generator_method(
                request, list_result_generator, request_id, request_metadata, duration_s
            )
        # Non-streaming response.
        total_segments = []
        text_parts = []
        try:
            assert list_result_generator is not None
            segments_types: dict[str, type[SpeechToTextSegment]] = {
                "transcribe": TranscriptionSegment,
                "translate": TranslationSegment,
            }
            segment_class: type[SpeechToTextSegment] = segments_types[self.task_type]
            text = ""
            chunk_size_in_s = self.asr_config.max_audio_clip_s
            if chunk_size_in_s is None:
                assert len(list_result_generator) == 1, (
                    "`max_audio_clip_s` is set to None, audio cannot be chunked"
                )
            for idx, result_generator in enumerate(list_result_generator):
                start_time = (
                    float(idx * chunk_size_in_s) if chunk_size_in_s is not None else 0.0
                )
                async for op in result_generator:
                    if request.response_format == "verbose_json":
                        segments: list[SpeechToTextSegment] = (
                            self._get_verbose_segments(
                                tokens=tuple(op.outputs[0].token_ids),
                                segment_class=segment_class,
                                request=request,
                                start_time=start_time,
                            )
                        )

                        total_segments.extend(segments)
                        text_parts.extend([seg.text for seg in segments])
                    else:
                        text_parts.append(op.outputs[0].text)
            text = "".join(text_parts)
            if self.task_type == "transcribe":
                final_response: ResponseType
                # add usage in TranscriptionResponse.
                usage = {
                    "type": "duration",
                    # rounded up as per openAI specs
                    "seconds": int(math.ceil(duration_s)),
                }
                if request.response_format != "verbose_json":
                    final_response = cast(
                        T, TranscriptionResponse(text=text, usage=usage)
                    )
                else:
                    final_response = cast(
                        V,
                        TranscriptionResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            else:
                # no usage in response for translation task
                if request.response_format != "verbose_json":
                    final_response = cast(T, TranslationResponse(text=text))
                else:
                    final_response = cast(
                        V,
                        TranslationResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            return final_response
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)

    async def _speech_to_text_stream_generator(
        self,
        request: SpeechToTextRequest,
        list_result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        chunk_object_type: Literal["translation.chunk", "transcription.chunk"],
        response_stream_choice_class: type[TranscriptionResponseStreamChoice]
        | type[TranslationResponseStreamChoice],
        stream_response_class: type[TranscriptionStreamResponse]
        | type[TranslationStreamResponse],
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = self.enable_force_include_usage or request.stream_include_usage
        include_continuous_usage = (
            request.stream_continuous_usage_stats
            if include_usage and request.stream_continuous_usage_stats
            else False
        )

        try:
            for result_generator in list_result_generator:
                async for res in result_generator:
                    # On first result.
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                        if audio_tokens := self.model_cls.get_num_audio_tokens(
                            audio_duration_s, self.asr_config, self.model_config
                        ):
                            num_prompt_tokens += audio_tokens

                    # We need to do it here, because if there are exceptions in
                    # the result_generator, it needs to be sent as the FIRST
                    # response (by the try...catch).

                    # Just one output (n=1) supported.
                    assert len(res.outputs) == 1
                    output = res.outputs[0]

                    delta_message = DeltaMessage(content=output.text)
                    completion_tokens += len(output.token_ids)

                    if output.finish_reason is None:
                        # Still generating, send delta update.
                        choice_data = response_stream_choice_class(delta=delta_message)
                    else:
                        # Model is finished generating.
                        choice_data = response_stream_choice_class(
                            delta=delta_message,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason,
                        )

                    chunk = stream_response_class(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )

                final_usage_chunk = stream_response_class(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
            )

        except Exception as e:
            logger.exception("Error in %s stream generator.", self.task_type)
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    def _split_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> list[np.ndarray]:
        assert self.asr_config.max_audio_clip_s is not None, (
            f"{self.asr_config.max_audio_clip_s=} cannot be None to"
            " split audio into chunks."
        )
        chunk_size = sample_rate * self.asr_config.max_audio_clip_s
        overlap_size = sample_rate * self.asr_config.overlap_chunk_second
        chunks = []
        i = 0
        while i < audio_data.shape[-1]:
            if i + chunk_size >= audio_data.shape[-1]:
                # handle last chunk
                chunks.append(audio_data[..., i:])
                break

            # Find the best split point in the overlap region
            search_start = i + chunk_size - overlap_size
            search_end = min(i + chunk_size, audio_data.shape[-1])
            split_point = self._find_split_point(audio_data, search_start, search_end)

            # Extract chunk up to the split point
            chunks.append(audio_data[..., i:split_point])
            i = split_point
        return chunks

    def _find_split_point(self, wav: np.ndarray, start_idx: int, end_idx: int) -> int:
        """Find the best point to split audio by
        looking for silence or low amplitude.
        Args:
            wav: Audio tensor [1, T]
            start_idx: Start index of search region
            end_idx: End index of search region
        Returns:
            Index of best splitting point
        """
        segment = wav[start_idx:end_idx]

        # Calculate RMS energy in small windows
        min_energy = math.inf
        quietest_idx = 0
        min_energy_window = self.asr_config.min_energy_split_window_size
        assert min_energy_window is not None
        for i in range(0, len(segment) - min_energy_window, min_energy_window):
            window = segment[i : i + min_energy_window]
            energy = (window**2).mean() ** 0.5
            if energy < min_energy:
                quietest_idx = i + start_idx
                min_energy = energy
        return quietest_idx
