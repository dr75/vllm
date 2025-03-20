#!/bin/bash -e

pytest tests/v1/core/test_prefix_caching.py
pytest tests/v1/core/test_kv_cache_utils.py
pytest tests/core/block/test_prefix_caching_block.py


# pytest tests/v1/core/test_prefix_caching.py -s
# pytest tests/v1/core/test_kv_cache_utils.py -s
# pytest tests/core/block/test_prefix_caching_block.py -s
