#!/bin/bash

# voxtral segmentation
pytest -v tests/entrypoints/test_speech_to_text_segments.py

# anthropic cache salting and cached token reporting
pytest -v tests/entrypoints/openai/test_messages.py
