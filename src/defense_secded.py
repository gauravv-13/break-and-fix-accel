"""Phase 2b defense: SECDED error-correcting code per weight row.

SECDED = Single Error Correction, Double Error Detection (an "extended Hamming code").
For each row we store:
  - r Hamming check bits, h = XOR of the "positions" of all data bits that are 1.
    Data bits get the positions 3, 5, 6, 7, 9, ... (every number that is not a power of
    two); the check bits themselves sit at positions 1, 2, 4, 8, ...
  - 1 overall parity bit P over all data bits and check bits.
A row of 8 weights (64 data bits) needs r = 7, so 8 stored bits (the classic (72, 64)
code). A row of 2 weights (16 data bits) needs r = 5, so 6 stored bits.

When a row loads, the decoder recomputes h and compares:
  s = h_now XOR h_stored  (which position looks wrong), p = overall parity mismatch.
  s = 0, p = 0  -> no error
  p = 1         -> exactly one bit is wrong (if the code assumption holds):
                   s = a data position -> flip that data bit back ("corrected")
                   s = 0 or a check position -> the error was in a stored check bit,
                   the data is fine ("corrected")
                   s = a position that does not exist -> "detected"
  s != 0, p = 0 -> two bits wrong: cannot correct, "detected"
Three or more flips in one row can be "corrected" to the WRONG value (mis-correction)
or pass unseen; the experiments measure how often.

Cost model assumption: syndrome XOR tree + correction step = check_latency 2 cycles per
tile (one more than parity/CRC, because of the correction step).
"""
import numpy as np

from src import faults
from src.defense_checksum import RowCodeChecker


def n_check_bits(k_data):
    r = 1
    while (1 << r) < k_data + r + 1:
        r += 1
    return r


def data_positions(k_data):
    pos, p = [], 3
    while len(pos) < k_data:
        if p & (p - 1):          # not a power of two
            pos.append(p)
        p += 1
    return np.array(pos, dtype=np.int64)


class SecdedChecker(RowCodeChecker):
    kind = "secded"

    def __init__(self, clean_weights, check_latency=2):
        self.check_latency = check_latency
        self._pos = {}
        self._store(clean_weights)

    def _positions(self, n_bytes):
        if n_bytes not in self._pos:
            self._pos[n_bytes] = data_positions(n_bytes * 8)
        return self._pos[n_bytes]

    def bits_for_row(self, n_bytes):
        return n_check_bits(n_bytes * 8) + 1

    def _hamming(self, row):
        bits = np.unpackbits(np.asarray(row, dtype=np.int8).view(np.uint8), bitorder="little")
        pos = self._positions(len(row))[bits.astype(bool)]
        h = int(np.bitwise_xor.reduce(pos)) if pos.size else 0
        return h, int(bits.sum() & 1)

    def encode(self, row):
        """Stored word: Hamming bits in the low r bits, overall parity P in bit r."""
        r = n_check_bits(len(row) * 8)
        h, data_par = self._hamming(row)
        P = data_par ^ (bin(h).count("1") & 1)
        return h | (P << r)

    def decode_row(self, layer, k, nt, row):
        n_bytes = len(row)
        r = n_check_bits(n_bytes * 8)
        stored = self.stored[(layer, k, nt)]
        h_st, P_st = stored & ((1 << r) - 1), (stored >> r) & 1
        h_now, data_par = self._hamming(row)
        s = h_now ^ h_st
        p = data_par ^ (bin(h_st).count("1") & 1) ^ P_st
        if s == 0 and p == 0:
            return "ok", row
        if p == 0:
            return "detected", row
        if s == 0 or (s & (s - 1)) == 0:
            return "corrected", row                       # a stored check bit was hit
        where = np.nonzero(self._positions(n_bytes) == s)[0]
        if where.size == 0:
            return "detected", row                        # position does not exist
        j = int(where[0])
        fixed = row.copy().view(np.uint8)
        fixed[j // 8] ^= np.uint8(1 << (j % 8))
        return "corrected", fixed.view(np.int8)

    def check_row(self, layer, k, nt, row):
        """True if the row is clean or could be corrected."""
        return self.decode_row(layer, k, nt, row)[0] != "detected"

    def correct_weights(self, weights):
        """Return (weights after correction, counts of ok/corrected/detected rows)."""
        out = dict(weights)
        for key in faults.WEIGHT_KEYS:
            out[key] = weights[key].copy()
        counts = {"ok": 0, "corrected": 0, "detected": 0}
        for (l, k, nt), status, fixed in self.row_statuses(weights):
            counts[status] += 1
            W = out[faults.WEIGHT_KEYS[l]]
            W[k, nt * 8: nt * 8 + len(fixed)] = fixed
        return out, counts

    def bit_syndromes(self, n_bytes):
        # flipping data bit j changes s by its position and always flips p
        r = n_check_bits(n_bytes * 8)
        return self._positions(n_bytes) | (1 << r)
