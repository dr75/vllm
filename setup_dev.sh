#!/bin/bash

pip install -r requirements/cuda.txt
pip install -r requirements/dev.txt

pre-commit install --hook-type pre-commit --hook-type commit-msg

echo "https://docs.vllm.ai/en/latest/contributing/overview.html"
