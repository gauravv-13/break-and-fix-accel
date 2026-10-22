"""Phase 2 defense: one checksum per weight row, checked as the row loads into the array.

A "row" is what the array loads in one cycle: up to 8 INT8 weights W[k, 8*nt : 8*nt+8]
(row k of the tile in tile-column nt). Layer 1 (64x32) has 64*4 = 256 rows of 8 weights.
Layer 2 (32x10) has 32 rows of 8 weights and 32 rows of 2 weights (its last tile-column
only has 2 real outputs; padding is not stored or checked).

Two checksums:
- parity : 1 bit per row  = XOR of all the row's bits. Catches any ODD number of flips
           in a row, misses any EVEN number.
- crc8   : 8 bits per row, CRC-8 with polynomial x^8 + x^2 + x + 1 (0x07), init 0,
           bytes fed in memory order, most significant bit first. Catches every 1-, 2- and
           3-bit error in a row this short (tested in tests/test_phase2.py) and every
           burst up to 8 bits long.
(A third defense, SECDED error correction, is in defense_secded.py and uses the same
RowCodeChecker base class.)

Checksums are computed once from the clean weights ("when the weights are written").
By default they are assumed to be stored in a protected place; Phase 2b also lets random
flips hit them (with_flipped_stored_bits). Cost model: the checker is a pipelined XOR tree
working on each row while it loads, so the only extra time is check_latency (default 1)
cycle per tile, waiting for the last row's result before compute starts.

Bit numbering inside a row (used by bit_syndromes and the adaptive attack): row bit j is
bit j % 8 (0 = LSB) of the row's byte j // 8, which matches the flat weight memory.
"""
import copy

import numpy as np

from src import faults

CRC8_POLY = 0x07


def _crc8_table():
    table = np.zeros(256, dtype=np.uint8)
    for b in range(256):
        c = b
        for _ in range(8):
            c = ((c << 1) ^ CRC8_POLY) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        table[b] = c
    return table


CRC8_TABLE = _crc8_table()


def parity(row_bytes):
    return int(np.unpackbits(np.asarray(row_bytes, dtype=np.int8).view(np.uint8)).sum() & 1)


def crc8(row_bytes):
    c = 0
    for b in np.asarray(row_bytes, dtype=np.int8).view(np.uint8):
        c = int(CRC8_TABLE[c ^ int(b)])
    return c


CHECKSUMS = {"parity": (parity, 1), "crc8": (crc8, 8)}


def rows(weights):
    """Yield (layer, k, nt, row_values) for every stored weight row, in load order."""
    for layer, key in enumerate(faults.WEIGHT_KEYS):
        W = weights[key]
        K, N = W.shape
        for nt in range(-(-N // 8)):
            for k in range(K):
                yield layer, k, nt, W[k, nt * 8: min(nt * 8 + 8, N)]


def single_bit_pattern(n_bytes, j):
    """A row of n_bytes zero bytes with only row bit j set (as int8)."""
    e = np.zeros(n_bytes, dtype=np.uint8)
    e[j // 8] = np.uint8(1 << (j % 8))
    return e.view(np.int8)


class RowCodeChecker:
    """Base class: one stored code word (an int) per weight row.

    Subclasses define encode(row) -> int, bits_for_row(n_bytes) and bit_syndromes(n_bytes).
    """
    check_latency = 1

    def _store(self, clean_weights):
        self.keys = [(l, k, nt) for l, k, nt, _ in rows(clean_weights)]
        self.widths = [self.bits_for_row(len(r)) for _, _, _, r in rows(clean_weights)]
        self.stored = {(l, k, nt): self.encode(r) for l, k, nt, r in rows(clean_weights)}

    def decode_row(self, layer, k, nt, row):
        """Return (status, row_to_load). Detection-only codes never change the row."""
        ok = self.encode(row) == self.stored[(layer, k, nt)]
        return ("ok" if ok else "detected"), row

    def check_row(self, layer, k, nt, row):
        """Called by the accelerator as each weight row loads. True = row is OK."""
        return self.encode(row) == self.stored[(layer, k, nt)]

    def count_bad_rows(self, weights):
        """Check every row directly (same check as the accelerator, without running it)."""
        return sum(not self.check_row(l, k, nt, r) for l, k, nt, r in rows(weights))

    def row_statuses(self, weights):
        """List of (key, status, decoded_row) for every row."""
        out = []
        for l, k, nt, r in rows(weights):
            status, fixed = self.decode_row(l, k, nt, r)
            out.append(((l, k, nt), status, fixed))
        return out

    def memory_bits(self):
        return sum(self.widths)

    def with_flipped_stored_bits(self, positions):
        """Copy of this checker with bits of the STORED code words flipped.

        Stored-code memory is all rows' code words back to back, in load order.
        """
        c = copy.copy(self)
        c.stored = dict(self.stored)
        offsets = np.cumsum([0] + self.widths)
        for p in np.asarray(positions, dtype=np.int64).ravel():
            i = int(np.searchsorted(offsets, p, side="right") - 1)
            key = self.keys[i]
            c.stored[key] ^= 1 << int(p - offsets[i])
        return c

    def bit_syndromes(self, n_bytes):
        """Syndrome of a single flip of each row bit (ints). The code is linear, so a flip
        pattern is invisible to the check exactly when the XOR of its syndromes is 0."""
        raise NotImplementedError


class ChecksumChecker(RowCodeChecker):
    def __init__(self, clean_weights, kind="parity", check_latency=1):
        self.kind = kind
        self.fn, self.bits_per_row = CHECKSUMS[kind]
        self.check_latency = check_latency
        self._store(clean_weights)

    def encode(self, row):
        return self.fn(row)

    def bits_for_row(self, n_bytes):
        return self.bits_per_row

    def bit_syndromes(self, n_bytes):
        # parity and CRC-8 (init 0, no final XOR) are linear: code(a ^ b) = code(a) ^ code(b)
        return np.array([self.fn(single_bit_pattern(n_bytes, j)) for j in range(n_bytes * 8)])
