"""Phase 5: test vectors for the Verilog row checkers, with expected outputs computed by the
Python models (src/defense_checksum.py and src/defense_secded.py).

Each vector is one wide hex word, fields packed MSB first:
  parity: {row, stored(1), exp_code(1), exp_ok(1)}
  crc8  : {row, stored(8), exp_code(8), exp_ok(1)}
  secded: {row, stored(R+1), exp_fixed(row), exp_status(2), exp_code(R+1)}
Row packing: byte i (memory order) is bits [8i+7 : 8i], so row bit j = bit j.
"""
from pathlib import Path

import numpy as np

from src import quantize
from src.defense_checksum import ChecksumChecker, crc8, parity, rows
from src.defense_secded import SecdedChecker, n_check_bits

VEC_DIR = Path(__file__).resolve().parent.parent / "hw" / "vectors"
STATUS = {"ok": 0, "corrected": 1, "detected": 2}


def row_int(row):
    return int.from_bytes(np.asarray(row, dtype=np.int8).tobytes(), "little")


def int_row(v, n_bytes):
    return np.frombuffer(int(v).to_bytes(n_bytes, "little"), dtype=np.int8).copy()


def flip(row, bits):
    v = row_int(row)
    for b in bits:
        v ^= 1 << int(b)
    return int_row(v, len(row))


def cases(q, n_bytes, rng, n_random=300):
    """(clean_row, read_row, code_bit_flips) cases for rows of n_bytes."""
    clean = [r.copy() for _, _, _, r in rows(q) if len(r) == n_bytes]
    L = n_bytes * 8
    out = []
    for r in clean:
        out.append((r, r, []))
        for m in (1, 2, 3, 4):
            out.append((r, flip(r, rng.choice(L, m, replace=False)), []))
        out.append((r, r, ["code1"]))          # one stored check bit flipped
        out.append((r, flip(r, [rng.integers(L)]), ["code1"]))
    for j in range(L):                          # exhaustive single flips on one row
        out.append((clean[0], flip(clean[0], [j]), []))
    for _ in range(n_random):
        r = rng.integers(-128, 128, n_bytes, dtype=np.int64).astype(np.int8)
        m = int(rng.integers(0, 5))
        out.append((r, flip(r, rng.choice(L, m, replace=False)), ["code1"] if rng.random() < 0.2 else []))
    return out


def stored_with_flips(code, width, flips, rng):
    for f in flips:
        code ^= 1 << int(rng.integers(width))
    return code


def write_vectors(seed=0):
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    q, _, _ = quantize.load_npz()
    sec = SecdedChecker(q)
    key_for = {8: (0, 0, 0), 2: (1, 0, 1)}       # an 8-byte row and a 2-byte row
    counts = {}
    for n_bytes in (8, 2):
        rng = np.random.default_rng(seed + n_bytes)
        cs = cases(q, n_bytes, rng)
        R = n_check_bits(n_bytes * 8)
        lines = {"parity": [], "crc8": [], "secded": []}
        for clean, read, fl in cs:
            rv = row_int(read)
            # parity
            st = stored_with_flips(parity(clean), 1, fl, rng)
            code = parity(read)
            lines["parity"].append((((rv << 1 | st) << 1 | code) << 1) | int(code == st))
            # crc8
            st = stored_with_flips(crc8(clean), 8, fl, rng)
            code = crc8(read)
            lines["crc8"].append((((rv << 8 | st) << 8 | code) << 1) | int(code == st))
            # secded: decode with the stored word replaced by ours
            st = stored_with_flips(sec.encode(clean), R + 1, fl, rng)
            saved = sec.stored[key_for[n_bytes]]
            sec.stored[key_for[n_bytes]] = st
            status, fixed = sec.decode_row(*key_for[n_bytes], read)
            sec.stored[key_for[n_bytes]] = saved
            code = sec.encode(read)
            v = rv
            v = (v << (R + 1)) | st
            v = (v << (8 * n_bytes)) | row_int(fixed)
            v = (v << 2) | STATUS[status]
            v = (v << (R + 1)) | code
            lines["secded"].append(v)
        widths = {"parity": 8 * n_bytes + 3, "crc8": 8 * n_bytes + 17, "secded": 16 * n_bytes + 2 * R + 4}
        for name, vs in lines.items():
            hexw = -(-widths[name] // 4)
            path = VEC_DIR / f"{name}_{n_bytes}B.hex"
            path.write_text("\n".join(f"{v:0{hexw}x}" for v in vs) + "\n")
            counts[f"{name}_{n_bytes}B"] = len(vs)
    return counts


if __name__ == "__main__":
    print(write_vectors())
