"""Phase 2b: adaptive attacker who knows which row code protects the weights.

Idea: every defense here (parity, CRC-8, SECDED) is a LINEAR code. Each single bit flip
in a row has a "syndrome" (how it changes the check), and a flip pattern is invisible
exactly when the XOR of its bits' syndromes is 0. So we can SEARCH for invisible patterns:
  - parity : any 2 bits in the same row (smallest invisible pattern: 2 bits)
  - CRC-8  : no invisible 1-, 2- or 3-bit patterns in our rows; we enumerate all
             invisible 4-bit patterns (pairs of pairs with equal syndrome)
  - SECDED : same, smallest invisible pattern has 4 bits
The attacker only uses the smallest invisible patterns (a limitation: bigger patterns
are not searched).

Attack (same spirit as the Phase 2 bit search, but over patterns instead of single bits):
each round, compute the gradient, score EVERY allowed pattern with a first-order estimate
(sum over its bits of gradient x weight change), take the top_p patterns, apply each one
for real and measure the INT8 loss on the attacker batch, keep the best one. Repeat until
test accuracy <= stop_acc or max_bits bits have been flipped.

The undefended control uses the same search with single-bit patterns, so differences
between attacks come from the constraint, not from the search method.
"""
import itertools

import numpy as np

from src import bfa, faults, quantize
from src.defense_checksum import rows

BIT_VALUE = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.float64)   # two's complement


def invisible_patterns(syndromes, weight):
    """All row-local bit patterns of the given weight (1, 2 or 4) with XOR syndrome 0."""
    L = len(syndromes)
    if weight == 1:
        return np.array([[j] for j in range(L) if syndromes[j] == 0], dtype=np.int64).reshape(-1, 1)
    pairs = np.array(list(itertools.combinations(range(L), 2)), dtype=np.int64)
    ps = syndromes[pairs[:, 0]] ^ syndromes[pairs[:, 1]]
    if weight == 2:
        return pairs[ps == 0]
    assert weight == 4
    found = set()
    order = np.argsort(ps, kind="stable")
    ps_sorted, pairs_sorted = ps[order], pairs[order]
    bounds = np.flatnonzero(np.diff(ps_sorted)) + 1
    for group in np.split(pairs_sorted, bounds):
        for a, b in itertools.combinations(range(len(group)), 2):
            quad = set(group[a]) | set(group[b])
            if len(quad) == 4:
                found.add(tuple(sorted(quad)))
    return np.array(sorted(found), dtype=np.int64).reshape(-1, 4)


def smallest_invisible(checker, n_bytes):
    """(weight, local patterns) for the smallest invisible patterns in a row of n_bytes."""
    syn = checker.bit_syndromes(n_bytes)
    for w in (1, 2, 4):
        p = invisible_patterns(syn, w)
        if len(p):
            return w, p
    raise RuntimeError("no invisible pattern of weight <= 4")


def global_patterns(weights, checker=None):
    """All allowed flip patterns as global bit positions, shape (P, pattern_weight).

    checker=None means no defense: every single bit is a pattern. Rows are contiguous in
    the flat memory, so row bit j sits at (first bit of the row) + j.
    """
    if checker is None:
        return np.arange(faults.n_bits(weights), dtype=np.int64).reshape(-1, 1)
    cache, out = {}, []
    for l, k, nt, row in rows(weights):
        n_bytes = len(row)
        if n_bytes not in cache:
            cache[n_bytes] = smallest_invisible(checker, n_bytes)
        _, local = cache[n_bytes]
        out.append(faults.position_of(l, k, nt * 8, 0, weights) + local)
    widths = {p.shape[1] for p in out}
    assert len(widths) == 1, "all rows must use the same pattern weight"
    return np.concatenate(out)


def run_pattern_attack(w, patterns, X_attack, y_attack, X_test, y_test,
                       top_p=200, max_bits=400, stop_acc=0.11):
    def test_acc(ww):
        return float(np.mean(np.argmax(quantize.int8_forward(X_test, ww)[0], axis=1) == y_test))

    cur = w
    chosen, accs, losses, bits_used = [], [test_acc(w)], [bfa.int8_loss(X_attack, y_attack, w)], [0]
    pw = patterns.shape[1]
    while bits_used[-1] + pw <= max_bits and accs[-1] > stop_acc:
        g = bfa.weight_gradients(X_attack, y_attack, cur)
        g_byte = np.concatenate([g[key].ravel() for key in faults.WEIGHT_KEYS])
        bit_now = np.unpackbits(faults.weight_memory(cur), bitorder="little").astype(np.float64)
        n = bit_now.size
        delta = (1.0 - 2.0 * bit_now) * np.tile(BIT_VALUE, n // 8)   # weight change per bit flip
        bit_score = g_byte[np.arange(n) // 8] * delta
        score = bit_score[patterns].sum(axis=1)
        cand = np.argpartition(-score, min(top_p, len(score) - 1))[:top_p]
        best = (-np.inf, None)
        for c in cand:
            loss = bfa.int8_loss(X_attack, y_attack, faults.flip_bits(cur, patterns[c]))
            if loss > best[0]:
                best = (loss, c)
        cur = faults.flip_bits(cur, patterns[best[1]])
        chosen.append(patterns[best[1]].tolist())
        losses.append(best[0])
        bits_used.append(bits_used[-1] + pw)
        accs.append(test_acc(cur))
    return {"patterns": chosen, "test_acc": accs, "attack_loss": losses, "bits": bits_used,
            "final_weights": cur}
