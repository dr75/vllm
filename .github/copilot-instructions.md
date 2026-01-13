<!-- markdownlint-disable -->

# Copilot instructions (vLLM)

## Scope
- This repo’s production code lives under `vllm/` and `csrc/`.
- Treat `vllm-debug/` as **out-of-tree local debugging tooling** (separate repo); don’t assume its conventions apply unless the change explicitly targets `vllm-debug/`.

## Big picture (where to look)
- **User entrypoints**
  - CLI: `vllm` → `vllm.entrypoints.cli.main:main` (modules are lazily imported).
  - OpenAI-compatible server: `vllm/entrypoints/openai/api_server.py`.
  - OpenAI API implementation details live in `vllm/entrypoints/openai/` (start at `serving_chat.py` and `serving_transcription.py`).
  - Python API: `vllm/entrypoints/llm.py` (`LLM` class).
- **Engine core**
  - `vllm/engine/*` is mostly a facade; `vllm/engine/llm_engine.py` aliases to `vllm/v1/engine/llm_engine.py`.
  - When changing scheduling/execution behavior, expect the real implementation under `vllm/v1/engine/` and `vllm/v1/executor/`.
- **Kernels / custom ops**
  - CUDA/C++ sources: `csrc/`; Python shims: `vllm/_custom_ops.py`; built artifacts land as `vllm/_C.abi3.so` etc.
- **Config + env**
  - Environment variables are centralized in `vllm/envs.py` (import via `import vllm.envs as envs`).
  - Repo config objects live under `vllm/config/` and are validated by CI hooks.

## OpenAI serving modules
- `vllm/entrypoints/openai/serving_chat.py`: chat-completions implementation used by `api_server.py` (request/response shaping, streaming, tool-calling parsing via `vllm/tool_parsers/`, and chat-template/message normalization).
- `vllm/entrypoints/openai/serving_transcription.py`: audio transcription/translation implementation used by `api_server.py` (OpenAI-compatible `/v1/audio/*` endpoints; see `TranscriptionRequest` and response variants in `vllm/entrypoints/openai/protocol.py`).
- Transcription/translation deep dive: `vllm/entrypoints/openai/TRANSCRIPTIONS.md`.

## Developer workflows that matter
- Recommended dev Python version matches CI: Python 3.12 (see `docs/contributing/README.md`).
- Editable install (Python-only changes; uses precompiled wheel for compiled libs):
  - `VLLM_USE_PRECOMPILED=1 uv pip install -e .`
- Editable install (kernel/C++/CUDA changes; compiles locally):
  - `uv pip install -e .`
- Fast kernel iteration (after initial editable install):
  - `python tools/generate_cmake_presets.py`
  - `cmake --preset release`
  - `cmake --build --preset release --target install` (re-run after edits)
  - See `docs/contributing/incremental_build.md`.

## Lint/format conventions (don’t fight CI)
- vLLM relies on `pre-commit` (install + run):
  - `uv pip install pre-commit && pre-commit install`
  - `pre-commit run -a`
- Python formatting/lint is `ruff` (configured in `pyproject.toml`).
- C++/CUDA formatting is `clang-format` (hooked via pre-commit).
- CI has extra manual-stage hooks (e.g., `markdownlint`, `mypy-*`).

## Repo-specific guardrails
- Keep imports **lazy** at module roots where required:
  - `vllm/__init__.py` uses `__getattr__` dispatch; don’t add eager heavyweight imports there.
  - `vllm/entrypoints/cli/main.py` notes “future modules must be lazily loaded within main”.
- Prefer `import regex as re` (a pre-commit hook enforces this).
- Don’t introduce new `pickle`/`cloudpickle` imports (pre-commit blocks them).
- Don’t directly `import triton`; use existing utilities/wrappers (see `vllm/triton_utils/` and related helpers).
- Avoid editing vendored code under `vllm/third_party/` unless absolutely required.
- New Python files should include SPDX header lines (checked by pre-commit).

## Tests (what’s actually used)
- Main runner: `pytest tests/` (see `docs/contributing/README.md`).
- Quick single-test invocation: `pytest -s -v tests/test_logger.py`.
- Compile-specific tests are organized under `tests/compile/` (see `tests/compile/README.md`).
- Useful markers are defined in `pyproject.toml` (e.g., `cpu_test`, `distributed`, `optional`).
