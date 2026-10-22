"""Phase 1, step 3: cycle-by-cycle model of an 8x8 weight-stationary systolic array.

Picture of the array (R = C = 8):

            col 0   col 1  ...  col 7
  x[:,0] -> PE00 -> PE01 -> ... PE07        activations move RIGHT one PE per cycle
  x[:,1] -> PE10 -> PE11 -> ... PE17        partial sums move DOWN one PE per cycle
   ...       |       |           |
  x[:,7] -> PE70 -> PE71 -> ... PE77
             v       v           v
           out 0   out 1  ...  out 7        finished dot products leave at the bottom

PE(r, c) holds the weight W[r, c] (it never moves: "weight stationary").
Every cycle it takes the activation from its left neighbour and the partial sum from
the PE above, computes psum + activation * weight, and passes both on.

Skew: row r gets its input stream r cycles late, so that the activation x[m, r] and the
partial sum of vector m meet in PE(r, c) at the same cycle (cycle m + r + c).

Timing of one 8x8 tile with M input vectors (no overlap between tiles):
  load    : R cycles   (one weight row shifted into the array per cycle)
  fill    : R - 1      (input skew: until the bottom row receives its first activation)
  compute : M          (each row injects M activations, one per cycle)
  drain   : C - 1      (the last results ripple out through the remaining columns)
  total   : R + (R - 1) + M + (C - 1) = M + 22 for an 8x8 array.

Bigger matrices are split into 8x8 tiles (zero-padded at the edges). Partial results
from tiles along the K (input) dimension are added up in 32-bit output accumulators.
"""
import numpy as np

R = C = 8


def _wrap32(x):
    """Emulate a 32-bit two's complement register."""
    return x.astype(np.int64).astype(np.int32)


def run_tile(W_tile, X_stream, w_mask=None):
    """Simulate one weight tile cycle by cycle.

    W_tile:   (R, C) int8 weights (already zero-padded).
    X_stream: (M, R) int8 activations (already zero-padded).
    w_mask:   (R, C) bool, True where the weight is real (not padding). Used to count
              useful ("busy") PE-cycles.

    Returns (out, info): out is (M, C) int32 dot products, info has cycle counts.
    """
    W = W_tile.astype(np.int32)
    M = X_stream.shape[0]
    if w_mask is None:
        w_mask = np.ones((R, C), dtype=bool)

    act = np.zeros((R, C), dtype=np.int32)      # activation register in each PE
    valid = np.zeros((R, C), dtype=bool)        # does that register hold real data?
    psum = np.zeros((R, C), dtype=np.int32)     # partial-sum register in each PE
    out = np.zeros((M, C), dtype=np.int32)
    busy = 0

    n_stream = (R - 1) + M + (C - 1)            # fill + compute + drain
    for t in range(n_stream):
        # 1) activations shift right; new (skewed) inputs enter at column 0
        new_act = np.empty_like(act)
        new_valid = np.empty_like(valid)
        new_act[:, 1:] = act[:, :-1]
        new_valid[:, 1:] = valid[:, :-1]
        for r in range(R):
            m = t - r                           # row r is delayed by r cycles
            if 0 <= m < M:
                new_act[r, 0] = X_stream[m, r]
                new_valid[r, 0] = True
            else:
                new_act[r, 0] = 0
                new_valid[r, 0] = False
        # 2) every PE does one multiply-accumulate; psums move down one row
        prod = new_act * W
        new_psum = np.empty_like(psum)
        new_psum[0, :] = prod[0, :]
        new_psum[1:, :] = _wrap32(psum[:-1, :].astype(np.int64) + prod[1:, :])
        busy += int(np.sum(new_valid & w_mask))
        act, valid, psum = new_act, new_valid, new_psum
        # 3) collect results leaving the bottom row: column c finishes vector m = t-(R-1)-c
        for c in range(C):
            m = t - (R - 1) - c
            if 0 <= m < M:
                out[m, c] = psum[R - 1, c]

    info = {"load": R, "fill": R - 1, "compute": M, "drain": C - 1, "busy_pe_cycles": busy}
    info["total"] = info["load"] + n_stream
    return out, info


def matmul(X, W, check_row=None, check_latency=0):
    """Tiled matrix multiply X (M, K) int8 @ W (K, N) int8 -> (M, N) int32 on the array.

    W may also be int32 (used for 32-bit masked weight shares in Phase 3); products and
    partial sums then wrap modulo 2^32, exactly like 32-bit hardware registers.

    check_row(k, nt, row_values): optional checker called for every weight row as it loads
    (row k of the tile in tile-column nt). It returns either a bool (True = row is OK), or
    a tuple (status, row) for error-correcting checkers, where status is "ok", "corrected"
    or "detected" and row is the (possibly corrected) row that gets loaded into the array.
    check_latency: extra cycles added per tile while the checker finishes.

    Returns (Y, stats).
    """
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    kt_n = -(-K // R)
    nt_n = -(-N // C)
    Y = np.zeros((M, N), dtype=np.int32)
    stats = {"tiles": 0, "load": 0, "fill": 0, "compute": 0, "drain": 0, "check": 0,
             "busy_pe_cycles": 0, "total": 0, "check_failures": 0, "corrections": 0}

    for nt in range(nt_n):
        n0, n1 = nt * C, min((nt + 1) * C, N)
        for kt in range(kt_n):
            k0, k1 = kt * R, min((kt + 1) * R, K)
            W_tile = np.zeros((R, C), dtype=W.dtype)   # int8 normally; int32 for masked shares
            W_tile[: k1 - k0, : n1 - n0] = W[k0:k1, n0:n1]
            w_mask = np.zeros((R, C), dtype=bool)
            w_mask[: k1 - k0, : n1 - n0] = True
            if check_row is not None:
                for k in range(k0, k1):
                    res = check_row(k, nt, W[k, n0:n1])
                    if isinstance(res, tuple):
                        status, fixed = res
                        W_tile[k - k0, : n1 - n0] = fixed
                        stats["corrections"] += status == "corrected"
                        stats["check_failures"] += status == "detected"
                    elif not res:
                        stats["check_failures"] += 1
            X_tile = np.zeros((M, R), dtype=np.int8)
            X_tile[:, : k1 - k0] = X[:, k0:k1]
            part, info = run_tile(W_tile, X_tile, w_mask)
            Y[:, n0:n1] = _wrap32(Y[:, n0:n1].astype(np.int64) + part[:, : n1 - n0])
            stats["tiles"] += 1
            for key in ["load", "fill", "compute", "drain", "busy_pe_cycles", "total"]:
                stats[key] += info[key]
            stats["check"] += check_latency
            stats["total"] += check_latency

    stats["pe_cycles"] = stats["total"] * R * C
    stats["utilization"] = stats["busy_pe_cycles"] / stats["pe_cycles"]
    return Y, stats
