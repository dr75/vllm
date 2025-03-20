import hashlib
import math
import pickle
import random
import time
from typing import List, Optional
import blake3

def _block_hash_sha256(input) -> int:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)         #~2.5ms
    h = int.from_bytes(hashlib.sha256(input_bytes).digest(), byteorder="big")   #~4.8ms
    return h, h


def _block_hash_blake3(input: tuple[int, ...]) -> tuple[int, int]:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)         #~2.5ms
    h = int.from_bytes(blake3.blake3(input_bytes).digest(), byteorder="big")    #~5.8ms

    # hash_digest = blake3.blake3(input_bytes).digest()
    # h = int.from_bytes(memoryview(hash_digest), byteorder="big", signed=False)

    return h, h

def _block_hash_hash(input) -> int:
    h = hash(input)
    return h, h

def _block_hash_raw(input) -> int:
    key = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    h = hash(key)
    return h, key

_block_hash = None
_none_hash = None

def init_hash(hash_name: str):
    global _block_hash
    if hash_name == "sha256":
        _block_hash = _block_hash_sha256
    elif hash_name == "blake3":
        _block_hash = _block_hash_blake3
    elif hash_name == "hash":
        _block_hash = _block_hash_hash
    elif hash_name == "raw":
        _block_hash = _block_hash_raw
    else:
        raise ValueError(f"Unknown hash function: {hash_name}")

    global _none_hash
    none_key = tuple(int(ord(c)) for c in 'None')
    _none_hash = _block_hash(none_key)

def compute_hash(is_first_block: bool,
         prev_block_hash: Optional[int],
         cur_block_token_ids: List[int],
         extra_hash: Optional[int] = None) -> int:
    if is_first_block and prev_block_hash is None:
        prev_block_hash = _none_hash
    return _block_hash((int(is_first_block), prev_block_hash, *cur_block_token_ids,
                    int(extra_hash)))

def split_tokens(tokens: List[int], block_size: int) -> List[List[int]]:
    return [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]

def test(hash_name: str):
    init_hash(hash_name)
    random.seed(0)
    num_tokens = 50_000

    t = []
    for _ in range(50):
        h = 4242
        token_ids = [random.randint(0, 100_000) for _ in range(num_tokens)]
        chain = split_tokens(token_ids, 16)

        t0 = time.time()
        for chain_block in chain:
            h, key = compute_hash(False, h, chain_block, 123)
            assert key is not None

        t.append((time.time() - t0) * 1000*1000)

    t_mean = sum(t) / len(t)
    t_std = math.sqrt(sum((t - t_mean)**2 for t in t) / len(t))
    t_mean_error = t_std / math.sqrt(len(t))

    print(f"{hash_name}, {len(chain)} blocks")
    print(f"\tmean: {t_mean / 1000:.2f}ms")
    print(f"\tstd: {t_std / 1000:.2f}ms")
    print(f"\tmean error: {t_mean_error:.2f}us")
    print(f"\tper block: {t_mean / len(chain):.3f}us")
    return t_mean

test("raw")
t_sha = test("sha256")
test("blake3")
t_hash = test("hash")

print(f"hash is {t_sha / t_hash:.2f} times faster than sha256")
