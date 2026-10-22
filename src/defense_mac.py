"""Phase 2c defense: a keyed integrity check (truncated MAC) per weight row.

For every row we store tag = first t bits of MAC_key(address || row bytes), t = 8 or 16.
  - MAC = AES-CMAC with a 128-bit key (default), or HMAC-SHA256 (option), both from the
    `cryptography` library.
  - address = (layer, k, nt), 3 bytes, so a valid row + tag copied to another place fails.
  - The key lives inside the accelerator; the attacker does not know it. Without the key
    the attacker cannot predict which flips keep the tag the same: any change to a row
    passes only by luck, with chance about 2^-t per changed row.
  - The key here comes from a fixed seed so results are reproducible. A real chip would
    use a secret random key (and, against replaying OLD weights, a version counter in
    the message; not modelled).

Cycle cost (ASSUMPTIONS, not measured; one MAC engine next to the weight loader, rows of
a tile arrive one per cycle, compute waits until the last row of the tile is checked):
  AES-CMAC on one 11-byte message = 1 AES-128 block (10 rounds, 1 round per cycle)
  + 1 cycle to truncate/compare = 11 cycles per row.
  HMAC-SHA256 with the key's inner/outer states precomputed = 2 SHA-256 compressions
  (64 rounds each, 1 round per cycle) + 2 cycles = 130 cycles per row.
  - "pipelined" engine (fully unrolled, one row in per cycle): extra = latency per tile.
  - "iterative" engine (one row at a time): extra = rows_per_tile * latency - rows_per_tile
    (the 8 load cycles overlap with checking).
"""
import numpy as np
from cryptography.hazmat.primitives import cmac, hashes, hmac
from cryptography.hazmat.primitives.ciphers import algorithms

from src.defense_checksum import RowCodeChecker

KEY_SEED = 2026
ROW_LATENCY = {"cmac": 11, "hmac": 130}


def default_key():
    return np.random.default_rng(KEY_SEED).bytes(16)


def tile_extra_cycles(algo="cmac", engine="pipelined", rows_per_tile=8):
    lat = ROW_LATENCY[algo]
    if engine == "pipelined":
        return lat
    if engine == "iterative":
        return rows_per_tile * lat - rows_per_tile
    raise ValueError(engine)


def mac_bytes(key, algo, message):
    if algo == "cmac":
        c = cmac.CMAC(algorithms.AES(key))
    elif algo == "hmac":
        c = hmac.HMAC(key, hashes.SHA256())
    else:
        raise ValueError(algo)
    c.update(message)
    return c.finalize()


def row_message(layer, k, nt, row):
    return bytes([layer, k, nt]) + np.asarray(row, dtype=np.int8).tobytes()


class MacChecker(RowCodeChecker):
    def __init__(self, clean_weights, tag_bits=8, algo="cmac", engine="pipelined", key=None):
        assert 1 <= tag_bits <= 32
        self.kind = f"mac{tag_bits}"
        self.tag_bits, self.algo, self.engine = tag_bits, algo, engine
        self._key = default_key() if key is None else key
        self.check_latency = tile_extra_cycles(algo, engine)
        self._store(clean_weights)

    def bits_for_row(self, n_bytes):
        return self.tag_bits

    def tag32(self, layer, k, nt, row):
        """First 32 bits of the MAC (used to get several truncations from one MAC call)."""
        return int.from_bytes(mac_bytes(self._key, self.algo, row_message(layer, k, nt, row))[:4], "big")

    def encode_at(self, layer, k, nt, row):
        return self.tag32(layer, k, nt, row) >> (32 - self.tag_bits)

    def encode(self, row):
        raise TypeError("a MAC tag depends on the row's address; use encode_at")

    def bit_syndromes(self, n_bytes):
        raise TypeError("a keyed MAC is not linear: invisible patterns cannot be computed without the key")
