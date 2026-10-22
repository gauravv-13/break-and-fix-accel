"""Readable, textbook AES-128 encryption from scratch (FIPS-197).

State representation:
- A 16-byte block is represented as a 4x4 matrix of bytes `state[row][col]`
  in column-major order, matching FIPS-197 Section 3.4:
      state[row][col] = block[4 * col + row]
- Key expansion produces 11 round keys (each 16 bytes, or 4 words of 4 bytes).
"""

from __future__ import annotations
from typing import List, Dict

# FIPS-197 Figure 7: AES S-box (256 bytes)
SBOX: tuple[int, ...] = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

# Round constants Rcon[1..10] (first byte of word; remaining 3 bytes are 0x00)
RCON: tuple[int, ...] = (
    0x00,  # unused 0-index placeholder so RCON[1..10] matches 1-based round index
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36,
)

State = List[List[int]]


def bytes_to_state(block: bytes) -> State:
    """Convert a 16-byte buffer into a 4x4 column-major state matrix."""
    if len(block) != 16:
        raise ValueError(f"Expected 16 bytes, got {len(block)}")
    state = [[0] * 4 for _ in range(4)]
    for col in range(4):
        for row in range(4):
            state[row][col] = block[4 * col + row]
    return state


def state_to_bytes(state: State) -> bytes:
    """Convert a 4x4 column-major state matrix back into 16 bytes."""
    out = bytearray(16)
    for col in range(4):
        for row in range(4):
            out[4 * col + row] = state[row][col] & 0xFF
    return bytes(out)


def sub_bytes(state: State) -> State:
    """Apply the AES S-box to every byte of the 4x4 state."""
    return [[SBOX[state[r][c]] for c in range(4)] for r in range(4)]


def shift_rows(state: State) -> State:
    """Cyclically shift row `r` of the state left by `r` positions."""
    return [
        [state[r][(c + r) % 4] for c in range(4)]
        for r in range(4)
    ]


def xtime(b: int) -> int:
    """Multiply byte `b` by 2 in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1 (0x11B)."""
    b &= 0xFF
    return ((b << 1) ^ 0x1B) & 0xFF if (b & 0x80) else (b << 1)


def gf_mul(a: int, b: int) -> int:
    """Multiply two bytes in GF(2^8) (used for readability in MixColumns)."""
    res = 0
    a &= 0xFF
    b &= 0xFF
    for _ in range(8):
        if b & 1:
            res ^= a
        a = xtime(a)
        b >>= 1
    return res


def mix_columns(state: State) -> State:
    """Multiply each column of the state by the AES MixColumns matrix over GF(2^8)."""
    new_state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        s0, s1, s2, s3 = state[0][c], state[1][c], state[2][c], state[3][c]
        new_state[0][c] = gf_mul(2, s0) ^ gf_mul(3, s1) ^ s2 ^ s3
        new_state[1][c] = s0 ^ gf_mul(2, s1) ^ gf_mul(3, s2) ^ s3
        new_state[2][c] = s0 ^ s1 ^ gf_mul(2, s2) ^ gf_mul(3, s3)
        new_state[3][c] = gf_mul(3, s0) ^ s1 ^ s2 ^ gf_mul(2, s3)
    return new_state


def add_round_key(state: State, round_key: bytes) -> State:
    """XOR the 4x4 state with a 16-byte round key."""
    rk_state = bytes_to_state(round_key)
    return [
        [state[r][c] ^ rk_state[r][c] for c in range(4)]
        for r in range(4)
    ]


def key_expansion_words(key: bytes) -> List[bytes]:
    """Expand a 16-byte AES-128 key into 44 4-byte words w[0..43] (FIPS-197 Section 5.2)."""
    if len(key) != 16:
        raise ValueError(f"AES-128 key must be 16 bytes, got {len(key)}")

    words: List[bytes] = [key[4 * i : 4 * (i + 1)] for i in range(4)]

    for i in range(4, 44):
        temp = list(words[i - 1])
        if i % 4 == 0:
            # RotWord: [a0, a1, a2, a3] -> [a1, a2, a3, a0]
            temp = temp[1:] + temp[:1]
            # SubWord: apply S-box to each byte
            temp = [SBOX[b] for b in temp]
            # XOR with Rcon[i // 4]
            temp[0] ^= RCON[i // 4]
        prev = words[i - 4]
        words.append(bytes(prev[j] ^ temp[j] for j in range(4)))

    return words


def key_expansion(key: bytes) -> List[bytes]:
    """Expand a 16-byte AES-128 key into 11 16-byte round keys."""
    words = key_expansion_words(key)
    return [b"".join(words[4 * r : 4 * (r + 1)]) for r in range(11)]


def aes128_encrypt_rounds(plaintext: bytes, key: bytes) -> Dict[str, List[bytes] | bytes]:
    """Encrypt one 16-byte block and record intermediate states per round for testing."""
    round_keys = key_expansion(key)
    state = bytes_to_state(plaintext)

    after_add_round_key: List[bytes] = []
    after_sub_bytes: List[bytes] = []
    after_shift_rows: List[bytes] = []
    after_mix_columns: List[bytes] = []

    # Initial round (Round 0)
    state = add_round_key(state, round_keys[0])
    after_add_round_key.append(state_to_bytes(state))

    # Rounds 1..9
    for r in range(1, 10):
        state = sub_bytes(state)
        after_sub_bytes.append(state_to_bytes(state))
        state = shift_rows(state)
        after_shift_rows.append(state_to_bytes(state))
        state = mix_columns(state)
        after_mix_columns.append(state_to_bytes(state))
        state = add_round_key(state, round_keys[r])
        after_add_round_key.append(state_to_bytes(state))

    # Final round (Round 10 - no MixColumns)
    state = sub_bytes(state)
    after_sub_bytes.append(state_to_bytes(state))
    state = shift_rows(state)
    after_shift_rows.append(state_to_bytes(state))
    state = add_round_key(state, round_keys[10])
    after_add_round_key.append(state_to_bytes(state))

    ciphertext = state_to_bytes(state)
    return {
        "round_keys": round_keys,
        "after_sub_bytes": after_sub_bytes,
        "after_shift_rows": after_shift_rows,
        "after_mix_columns": after_mix_columns,
        "after_add_round_key": after_add_round_key,
        "ciphertext": ciphertext,
    }


def aes128_encrypt_block(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt a single 16-byte plaintext block with a 16-byte AES-128 key."""
    round_keys = key_expansion(key)
    state = bytes_to_state(plaintext)

    state = add_round_key(state, round_keys[0])
    for r in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[r])

    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[10])
    return state_to_bytes(state)
