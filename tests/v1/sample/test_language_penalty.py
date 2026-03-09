# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import torch

import vllm.v1.sample.language_penalty as language_penalty
from vllm.v1.sample.language_penalty import (
    LANGUAGE_LOGIT_BIAS,
    LanguagePenaltyLogitsProcessor,
    compute_bias_from_tokenizer,
)


class _FakeTokenizer:
    def __init__(self, decoded_tokens: list[str]):
        self.decoded_tokens = decoded_tokens
        self.max_token_id = len(decoded_tokens) - 1

    def decode(self, token_ids, **kwargs):
        assert len(token_ids) == 1
        return self.decoded_tokens[token_ids[0]]


def test_compute_language_penalty_from_tokenizer():
    tokenizer = _FakeTokenizer(
        [
            "hello",
            "world",
            "犹豫",
            "案",
            "!",
        ]
    )

    logit_bias = compute_bias_from_tokenizer(tokenizer, vocab_size=5)

    expected = torch.tensor(
        [
            0.0,
            0.0,
            LANGUAGE_LOGIT_BIAS,
            LANGUAGE_LOGIT_BIAS,
            0.0,
        ]
    )
    assert torch.equal(logit_bias, expected)


def test_compute_language_penalty_penalizes_ambiguous_han_tokens():
    tokenizer = _FakeTokenizer(
        [
            "東京",
            "日本",
            "hello",
        ]
    )

    logit_bias = compute_bias_from_tokenizer(tokenizer, vocab_size=3)

    expected = torch.tensor(
        [
            LANGUAGE_LOGIT_BIAS,
            LANGUAGE_LOGIT_BIAS,
            0.0,
        ]
    )
    assert torch.equal(logit_bias, expected)


def test_compute_language_penalty_skips_han_tokens_with_kana():
    tokenizer = _FakeTokenizer(
        [
            "東京で",
            "カタカナ漢字",
            "ひらがな漢字",
            "案",
        ]
    )

    logit_bias = compute_bias_from_tokenizer(tokenizer, vocab_size=4)

    expected = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            LANGUAGE_LOGIT_BIAS,
        ]
    )
    assert torch.equal(logit_bias, expected)


def test_language_penalty_logits_processor(monkeypatch):
    calls = {"count": 0}
    logit_bias = torch.tensor([0.0, LANGUAGE_LOGIT_BIAS, 0.0], dtype=torch.float32)

    def fake_get_logit_bias(**kwargs):
        calls["count"] += 1
        assert kwargs["vocab_size"] == 3
        return logit_bias.to(device=kwargs["device"])

    monkeypatch.setattr(
        language_penalty,
        "get_logit_bias",
        fake_get_logit_bias,
    )

    processor = LanguagePenaltyLogitsProcessor(
        cast(
            Any,
            SimpleNamespace(
                model_config=SimpleNamespace(),
                load_config=SimpleNamespace(),
            ),
        ),
        torch.device("cpu"),
        False,
    )

    updated = processor.apply(torch.zeros((2, 3), dtype=torch.float32))
    expected = logit_bias.unsqueeze(0).expand(2, -1)

    assert torch.equal(updated, expected)

    updated_again = processor.apply(torch.zeros((1, 3), dtype=torch.float32))
    assert torch.equal(updated_again, logit_bias.unsqueeze(0))
    assert calls["count"] == 1
