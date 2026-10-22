import itertools

import numpy as np
import pytest

from src import accel, adaptive, faults, model, quantize
from src.defense_checksum import ChecksumChecker, rows
from src.defense_secded import SecdedChecker, data_positions, n_check_bits


@pytest.fixture(scope="module")
def qm():
    return quantize.load_npz()


@pytest.fixture(scope="module")
def checkers(qm):
    q, _, _ = qm
    return {"parity": ChecksumChecker(q, "parity"), "crc8": ChecksumChecker(q, "crc8"),
            "secded": SecdedChecker(q)}


def zero_model():
    # one 8-byte row (layer 1) and one 2-byte row (layer 2)
    return {"W1q": np.zeros((1, 8), dtype=np.int8), "W2q": np.zeros((1, 2), dtype=np.int8)}


def flip_local(row, bits):
    e = row.copy().view(np.uint8)
    for j in bits:
        e[j // 8] ^= np.uint8(1 << (j % 8))
    return e.view(np.int8)


def test_secded_sizes():
    assert n_check_bits(64) == 7 and n_check_bits(16) == 5
    assert list(data_positions(4)) == [3, 5, 6, 7]


def test_secded_memory_bits(checkers):
    # 288 rows of 8 weights x (7+1) bits + 32 rows of 2 weights x (5+1) bits
    assert checkers["secded"].memory_bits() == 288 * 8 + 32 * 6


def test_secded_corrects_every_single_data_bit_flip(qm, checkers):
    q, _, _ = qm
    c = checkers["secded"]
    for l, k, nt, row in rows(q):
        assert c.decode_row(l, k, nt, row)[0] == "ok"
        for j in range(len(row) * 8):
            status, fixed = c.decode_row(l, k, nt, flip_local(row, [j]))
            assert status == "corrected"
            np.testing.assert_array_equal(fixed, row)


def test_secded_corrects_every_single_check_bit_flip(qm, checkers):
    q, _, _ = qm
    c = checkers["secded"]
    offsets = np.cumsum([0] + c.widths)
    all_rows = list(rows(q))
    for i, (l, k, nt, row) in enumerate(all_rows):
        for b in range(c.widths[i]):
            fc = c.with_flipped_stored_bits([offsets[i] + b])
            status, fixed = fc.decode_row(l, k, nt, row)
            assert status == "corrected"
            np.testing.assert_array_equal(fixed, row)


def test_secded_detects_every_double_flip():
    # SECDED is linear, so checking every 2-bit pattern on zero rows covers every row.
    z = zero_model()
    c = SecdedChecker(z)
    for l, k, nt, row in rows(z):
        n = len(row) * 8
        for a, b in itertools.combinations(range(n), 2):
            assert c.decode_row(l, k, nt, flip_local(row, [a, b]))[0] == "detected"
        # one data bit + one stored check bit
        offsets = np.cumsum([0] + c.widths)
        i = c.keys.index((l, k, nt))
        for a in range(n):
            for b in range(c.widths[i]):
                fc = c.with_flipped_stored_bits([offsets[i] + b])
                assert fc.decode_row(l, k, nt, flip_local(row, [a]))[0] == "detected"


def test_secded_accelerator_repairs_single_flips(qm, checkers):
    q, Xq, _ = qm
    rng = np.random.default_rng(7)
    all_rows = list(rows(q))
    picks = rng.choice(len(all_rows), size=30, replace=False)       # 30 different rows
    pos = []
    for i in picks:
        l, k, nt, r = all_rows[i]
        pos.append(faults.position_of(l, k, nt * 8, 0, q) + int(rng.integers(len(r) * 8)))
    faulty = faults.flip_bits(q, pos)
    logits, st = accel.run_inference(Xq, faulty, checker=checkers["secded"])
    np.testing.assert_array_equal(logits, quantize.int8_forward(Xq, q)[0])
    assert st["corrections"] == 30 and st["check_failures"] == 0


def test_secded_clean_accuracy_and_cycles(qm, checkers):
    q, Xq, y = qm
    plain, st0 = accel.run_inference(Xq, q)
    lg, st = accel.run_inference(Xq, q, checker=checkers["secded"])
    np.testing.assert_array_equal(lg, plain)
    assert model.accuracy(lg, y) == model.accuracy(plain, y)
    assert st["total_cycles"] - st0["total_cycles"] == 40 * checkers["secded"].check_latency


@pytest.mark.parametrize("kind", ["crc8", "secded"])
def test_no_invisible_patterns_below_4_bits(checkers, kind):
    c = checkers[kind]
    for n_bytes in (8, 2):
        syn = c.bit_syndromes(n_bytes)
        assert np.all(syn != 0)
        for a, b in itertools.combinations(range(len(syn)), 2):
            assert syn[a] ^ syn[b] != 0
        trip = np.array(list(itertools.combinations(range(len(syn)), 3)))
        assert np.all((syn[trip[:, 0]] ^ syn[trip[:, 1]] ^ syn[trip[:, 2]]) != 0)
        w, pats = adaptive.smallest_invisible(c, n_bytes)
        assert w == 4 and len(pats) > 0


def test_parity_smallest_invisible_is_any_pair(checkers):
    w, pats = adaptive.smallest_invisible(checkers["parity"], 8)
    assert w == 2 and len(pats) == 64 * 63 // 2


@pytest.mark.parametrize("kind", ["parity", "crc8", "secded"])
def test_enumerated_patterns_are_really_invisible(qm, checkers, kind):
    q, _, _ = qm
    c = checkers[kind]
    P = adaptive.global_patterns(q, c)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(P), size=300, replace=False):
        faulty = faults.flip_bits(q, P[i])
        assert not np.array_equal(faults.weight_memory(faulty), faults.weight_memory(q))
        assert all(st == "ok" for _, st, _ in c.row_statuses(faulty))


def test_adaptive_attack_stays_invisible_and_hurts(qm, checkers):
    q, Xq, y = qm
    X_train, _, y_train, _ = model.load_data()
    idx = np.random.default_rng(0).choice(len(y_train), 128, replace=False)
    Xa = quantize.quantize_input(X_train[idx], q["s_in"])
    c = checkers["crc8"]
    r = adaptive.run_pattern_attack(q, adaptive.global_patterns(q, c), Xa, y_train[idx], Xq, y, max_bits=16)
    assert r["bits"][-1] == 16
    assert c.count_bad_rows(r["final_weights"]) == 0
    assert r["test_acc"][-1] < r["test_acc"][0] - 0.1


def test_flipping_a_stored_checksum_bit_gives_false_alarm(qm, checkers):
    q, _, _ = qm
    for kind in ["parity", "crc8"]:
        c = checkers[kind]
        fc = c.with_flipped_stored_bits([0])
        assert fc.count_bad_rows(q) == 1          # weights untouched, one row flagged
        assert c.count_bad_rows(q) == 0           # original checker not changed
