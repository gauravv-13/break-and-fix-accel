"""Phase 2: simulated bit flips in weight memory (a software model of RowHammer-style faults).

Memory layout: all INT8 weights sit in one flat memory, layer 1 first, then layer 2,
each stored row-major (W[k, 0], W[k, 1], ... for k = 0, 1, ...).
  byte index b  = position // 8
  bit index     = position % 8    (bit 0 = least significant, bit 7 = sign bit)
Total bits = (64*32 + 32*10) * 8 = 18,944.

Biases (int32) and scales are not part of this memory and are never flipped.
"""
import numpy as np

WEIGHT_KEYS = ("W1q", "W2q")


def weight_memory(weights):
    """Flatten the INT8 weights into one byte array (a copy)."""
    return np.concatenate([weights[k].ravel() for k in WEIGHT_KEYS]).view(np.uint8).copy()


def n_bits(weights):
    return sum(weights[k].size for k in WEIGHT_KEYS) * 8


def from_memory(weights, mem):
    """Return a copy of `weights` whose INT8 matrices are read back from byte memory `mem`."""
    out = dict(weights)
    signed = mem.view(np.int8)
    offset = 0
    for k in WEIGHT_KEYS:
        size = weights[k].size
        out[k] = signed[offset:offset + size].reshape(weights[k].shape).copy()
        offset += size
    return out


def flip_bits(weights, positions):
    """Flip the given bit positions. Returns a NEW weights dict; the input is not changed.

    A position listed twice is flipped twice (and so ends up unchanged).
    """
    mem = weight_memory(weights)
    pos = np.asarray(positions, dtype=np.int64).ravel()
    if pos.size:
        assert pos.min() >= 0 and pos.max() < mem.size * 8, "bit position out of range"
        masks = (np.uint8(1) << (pos % 8).astype(np.uint8)).astype(np.uint8)
        np.bitwise_xor.at(mem, pos // 8, masks)
    return from_memory(weights, mem)


def position_of(layer, k, n, bit, weights):
    """Bit position of bit `bit` of weight W[layer][k, n] in the flat memory."""
    offset = sum(weights[key].size for key in WEIGHT_KEYS[:layer])
    N = weights[WEIGHT_KEYS[layer]].shape[1]
    return (offset + k * N + n) * 8 + bit
