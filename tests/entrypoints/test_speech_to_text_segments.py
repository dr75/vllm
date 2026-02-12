# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import pytest

from vllm.entrypoints.openai.speech_to_text import speech_to_text


def _split_tag(
    ts_text: str,
    *,
    closing: bool = False,
    terminator: str = ">",
) -> list[str]:
    return ["</" if closing else "<", *list(ts_text), terminator]


def _segment(
    start: float,
    end: float,
    text: str,
    kept_ids: list[int],
) -> dict[str, float | str | list[int]]:
    return {
        "start": start,
        "end": end,
        "text": text,
        "token_ids": kept_ids,
        "indices": kept_ids,
    }


TEST_CASES = [
    {
        "name": "basic_split_tags",
        "token_strs": [
            *_split_tag("00:00.0"),
            " hi",
            ".</",
            *list("00:01.0"),
            ">\n",
            *_split_tag("00:01.5"),
            " bye",
            ".</",
            *list("00:02.0"),
            ">",
        ],
        "start_time": 0.0,
        "expected": [
            _segment(0.0, 1.0, " hi.", [9]),
            _segment(1.5, 2.0, " bye.", [28]),
        ],
    },
    {
        "name": "implicit_close_with_prefix_on_open",
        "token_strs": [
            *_split_tag("00:00.0"),
            " one",
            " tail<",
            *list("00:01.0"),
            ">",
            " two",
            ".</",
            *list("00:02.0"),
            ">",
        ],
        "start_time": 0.0,
        "expected": [
            _segment(0.0, 1.0, " one tail", [9]),
            _segment(1.0, 2.0, " two.", [19]),
        ],
    },
    {
        "name": "invalid_tag_inside_segment_is_text",
        "token_strs": [
            *_split_tag("00:00.0"),
            " keep",
            " <bad>",
            " text",
            ".</",
            *list("00:01.0"),
            ">",
        ],
        "start_time": 0.0,
        "expected": [
            _segment(0.0, 1.0, " keep <bad> text.", [9, 10, 11]),
        ],
    },
    {
        "name": "ignore_outside_text_and_apply_offset",
        "token_strs": [
            "noise",
            *_split_tag("00:00.5", closing=True),
            *_split_tag("00:01.0"),
            " hi",
            ".</",
            *list("00:02.5"),
            ">",
            "trailing",
        ],
        "start_time": 10.0,
        "expected": [
            _segment(11.0, 12.5, " hi.", [19]),
        ],
    },
    {
        "name": "unfinished_segment_at_eof_is_dropped",
        "token_strs": [
            *_split_tag("00:00.0"),
            " dangling",
        ],
        "start_time": 0.0,
        "expected": [],
    },
    {
        "name": "truncated_tag_inside_segment_is_text",
        "token_strs": [
            *_split_tag("00:00.0"),
            "<",
            "0",
            "0",
            ":",
            "0",
            "1",
            ".",
            "0",
            " broken",
            ".</",
            *list("00:02.0"),
            ">",
        ],
        "start_time": 0.0,
        "expected": [
            _segment(0.0, 2.0, "<00:01.0 broken.", [9, 10, 11, 12, 13, 14, 15, 16, 17]),
        ],
    },
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_parse_verbose_json_voxtral(case):
    token_strs = case["token_strs"]
    token_ids = list(range(len(token_strs)))
    start_time = float(case.get("start_time", 0.0))

    spans = speech_to_text._parse_verbose_json_voxtral(
        token_ids,
        token_strs=token_strs,
        start_time=start_time,
    )

    expected = case["expected"]
    assert len(spans) == len(expected)
    for (seg_start, seg_end, seg_text, seg_token_ids, seg_indices), exp in zip(
        spans, expected
    ):
        assert seg_start == pytest.approx(exp["start"])
        assert seg_end == pytest.approx(exp["end"])
        assert seg_text == exp["text"]
        assert seg_token_ids == exp["token_ids"]
        assert seg_indices == exp["indices"]
