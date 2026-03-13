#!/bin/bash

# Disable FP8 DeepGEMM warmup: the precompiled vLLM wheel is built for cu130
# but local dev boxes may run cu128 (or have a missing vendored deep_gemm
# extension), which causes the warmup to abort even on BF16 models like
# voxtral-mini. The Docker image the patches actually ship in has a working
# deep_gemm, so this is purely a local-test workaround.
export VLLM_USE_DEEP_GEMM=0

# voxtral segmentation
pytest -v tests/entrypoints/test_speech_to_text_segments.py

# anthropic cache salting and cached token reporting
pytest -v tests/entrypoints/anthropic/test_messages.py
