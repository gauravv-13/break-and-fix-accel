"""Phase 3: two power side-channel defenses for the accelerator model (COST ONLY).

No attack code here. The side-channel evidence for this project is the AES CPA work in
results/aes/ (AES track). This module only adds the defenses and counts what they cost.

1) Shuffling: every inference (or every batch) uses a fresh random order of the input
   index k and the output index n: W is loaded as W[pk][:, pn] and the inputs as X[:, pk],
   and the outputs are put back in order. Integer addition does not care about order
   (also modulo 2^32), so the logits are bit-identical. A given weight's multiply now
   lands in a random tile/row/column, i.e. at a random time.

2) First-order Boolean masking of the multiply step:
   - Each weight is stored in memory as two 8-bit Boolean shares (w XOR m, m).
   - When a tile row loads, the shares are re-randomised (8 fresh bits), sign-extended to
     32 bits (sign extension is linear over XOR) and the upper 24 bits are masked with 24
     fresh random bits, so both 32-bit shares are uniform.
   - Goubin's Boolean-to-arithmetic conversion (CHES 2001) turns (w XOR r, r) into
     (A, r) with w = A + r mod 2^32 without ever recombining w; it needs 32 fresh bits.
   - Each PE multiplies the public activation x by BOTH shares and keeps two partial-sum
     chains: x*A + x*r = x*w (mod 2^32). The two column outputs are added in the
     post-processing unit, so only the finished dot product is ever unmasked.
   Why 32-bit shares: with 8-bit arithmetic shares the product would wrap modulo 256
   (wrong result), and narrow shares of a wider value are not uniform (they would leak).
   Only the multiply step is masked (as planned); bias/ReLU/requantize run unmasked.

Cycle/hardware ASSUMPTIONS (stated, not measured):
   - Shuffling: the permutation is generated (Fisher-Yates, one swap per cycle) while the
     previous layer runs, so 0 extra cycles; if not overlapped: K + N cycles per layer.
   - Masking: a pipelined Goubin converter at the weight-load port (1 row per cycle,
     2 cycles latency: it has two 32-bit subtractions in series) adds 2 cycles per tile.
     "parallel" datapath = 2 multipliers (8x32) and 2 partial-sum registers per PE, so no
     extra compute cycles; "serial" datapath = one widened PE, each tile run twice.
"""
import math

import numpy as np

from src import quantize, systolic

R = C = 8
MASK32 = np.uint64(0xFFFFFFFF)
CONVERT_LATENCY = 2          # cycles per tile, assumed (see module docstring)


# ---------------------------------------------------------------- shuffling
def fisher_yates_bits(n):
    """Random bits for one Fisher-Yates permutation of n items, one exact draw per step
    (ceil(log2(i+1)) bits for step i). Rejection sampling would need a bit more."""
    return sum(math.ceil(math.log2(i + 1)) for i in range(1, n))


def log2_factorial(n):
    """Information-theoretic minimum bits for a uniform permutation of n items."""
    return math.lgamma(n + 1) / math.log(2)


def shuffled_matmul(X, W, rng):
    K, N = W.shape
    pk, pn = rng.permutation(K), rng.permutation(N)
    Yp, st = systolic.matmul(X[:, pk], W[pk][:, pn])
    Y = np.empty_like(Yp)
    Y[:, pn] = Yp
    return Y, st


def shuffle_costs(weights):
    """Random bits and permutation-register bits for one shuffled inference."""
    bits = lo = regs = extra_serial = 0
    for key in ("W1q", "W2q"):
        K, N = weights[key].shape
        bits += fisher_yates_bits(K) + fisher_yates_bits(N)
        lo += log2_factorial(K) + log2_factorial(N)
        regs = max(regs, K * math.ceil(math.log2(K)) + N * math.ceil(math.log2(N)))
        extra_serial += K + N
    return {"random_bits": bits, "random_bits_min_log2_factorial": lo,
            "perm_register_bits": regs, "extra_cycles_if_not_overlapped": extra_serial}


# ---------------------------------------------------------------- masking
def goubin_b2a(xm, r, gamma):
    """Goubin's Boolean-to-arithmetic conversion on uint32 arrays.

    Input: Boolean shares xm = x XOR r, and r; gamma is fresh randomness.
    Output: A with x = A + r (mod 2^32). Uses the fact that (xm XOR r) - r is affine in r.
    """
    xm, r, g = (np.asarray(v, dtype=np.uint64) for v in (xm, r, gamma))
    T = xm ^ g
    T = (T - g) & MASK32
    T = T ^ xm
    G = g ^ r
    A = xm ^ G
    A = (A - G) & MASK32
    A = A ^ T
    return A.astype(np.uint32)


def sext8_to_u32(v8):
    """Sign-extend 8-bit values (given as uint8) to 32-bit, returned as uint32."""
    return v8.astype(np.uint8).view(np.int8).astype(np.int32).view(np.uint32)


def store_shares(Wq, rng):
    """Shares as written to memory: (w XOR m, m), both uint8."""
    m = rng.integers(0, 256, Wq.shape, dtype=np.uint8)
    return Wq.view(np.uint8) ^ m, m


def load_masked(s1, s2, rng):
    """What happens at weight load: refresh, extend to 32 bits, convert to arithmetic.

    Returns (A, r) as int32 arrays with w = A + r mod 2^32, and the random bits used.
    """
    rho = rng.integers(0, 256, s1.shape, dtype=np.uint8)               # 8 bits / weight
    s1, s2 = s1 ^ rho, s2 ^ rho
    upper = (rng.integers(0, 1 << 24, s1.shape, dtype=np.uint64) << np.uint64(8)).astype(np.uint32)  # 24 bits
    xm32 = sext8_to_u32(s1) ^ upper
    r32 = sext8_to_u32(s2) ^ upper
    gamma = rng.integers(0, 1 << 32, s1.shape, dtype=np.uint64).astype(np.uint32)            # 32 bits
    A = goubin_b2a(xm32, r32, gamma)
    return A.view(np.int32), r32.view(np.int32), s1.size * (8 + 24 + 32)


def masked_matmul(X, s1, s2, rng, datapath="parallel"):
    A, r, bits = load_masked(s1, s2, rng)
    YA, stA = systolic.matmul(X, A)
    Yr, str_ = systolic.matmul(X, r)
    Y = (YA.astype(np.int64) + Yr.astype(np.int64)).astype(np.int32)   # wraps mod 2^32
    conv = CONVERT_LATENCY * stA["tiles"]
    if datapath == "parallel":
        cycles = stA["total"] + conv
    elif datapath == "serial":
        cycles = stA["total"] + str_["total"] + conv
    else:
        raise ValueError(datapath)
    return Y, {"cycles": cycles, "random_bits": bits, "tiles": stA["tiles"]}


def masking_register_costs():
    """Extra storage per PE / array for the parallel datapath (from the design above)."""
    per_pe_weight = 2 * 32 - 8          # two 32-bit shares instead of one 8-bit weight
    per_pe_psum = 32                    # a second 32-bit partial-sum register
    return {"extra_bits_per_pe": per_pe_weight + per_pe_psum,
            "extra_bits_array": (per_pe_weight + per_pe_psum) * R * C,
            "multipliers_per_pe": "2 x (8x32) instead of 1 x (8x8)"}


# ---------------------------------------------------------------- inference wrappers
def _layers(h, weights, layer_fn):
    cycles = bits = 0
    for i, key in enumerate(("W1q", "W2q")):
        acc, c, b = layer_fn(h, key)
        cycles, bits = cycles + c, bits + b
        h = quantize.postprocess_hidden(acc, weights) if i == 0 else quantize.postprocess_output(acc, weights)
    return h, cycles, bits


def run_inference_shuffled(xq, weights, rng, per_image=False):
    """per_image=False: one random order for the whole batch (weights load once).
    per_image=True: a fresh order for every image, so every image reloads the weights."""
    groups = [xq[i:i + 1] for i in range(len(xq))] if per_image else [xq]
    sc = shuffle_costs(weights)
    outs, cycles, bits = [], 0, 0

    def layer(h, key):
        Y, st = shuffled_matmul(h, weights[key], rng)
        return Y, st["total"], 0

    for g in groups:
        out, c, _ = _layers(g, weights, layer)
        outs.append(out)
        cycles += c
        bits += sc["random_bits"]
    return np.concatenate(outs), {"total_cycles": cycles, "random_bits": bits}


def run_inference_masked(xq, shares, weights, rng, per_image=False, datapath="parallel"):
    """shares: {"W1q": (s1, s2), "W2q": (s1, s2)} from store_shares.
    per_image=True: fresh masks for every image (weights reload per image)."""
    groups = [xq[i:i + 1] for i in range(len(xq))] if per_image else [xq]
    outs, cycles, bits = [], 0, 0

    def layer(h, key):
        Y, st = masked_matmul(h, *shares[key], rng, datapath)
        return Y, st["cycles"], st["random_bits"]

    for g in groups:
        out, c, b = _layers(g, weights, layer)
        outs.append(out)
        cycles += c
        bits += b
    return np.concatenate(outs), {"total_cycles": cycles, "random_bits": bits}
