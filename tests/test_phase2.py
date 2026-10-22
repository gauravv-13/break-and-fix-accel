import itertools

import numpy as np
import pytest

from src import accel, bfa, defense_checksum as dc, faults, model, quantize


@pytest.fixture(scope="module")
def qm():
    return quantize.load_npz()


def test_flip_twice_restores_original(qm):
    q, _, _ = qm
    rng = np.random.default_rng(0)
    pos = rng.choice(faults.n_bits(q), size=50, replace=False)
    once = faults.flip_bits(q, pos)
    twice = faults.flip_bits(once, pos)
    assert not np.array_equal(faults.weight_memory(once), faults.weight_memory(q))
    for k in faults.WEIGHT_KEYS:
        np.testing.assert_array_equal(twice[k], q[k])
    # same position listed twice in one call -> no change
    same = faults.flip_bits(q, [123, 123])
    np.testing.assert_array_equal(same["W1q"], q["W1q"])


def test_flip_does_not_modify_input(qm):
    q, _, _ = qm
    before = q["W1q"].copy()
    faults.flip_bits(q, [0, 7, 15])
    np.testing.assert_array_equal(q["W1q"], before)


def test_flip_changes_exactly_the_right_bit(qm):
    q, _, _ = qm
    for layer, key in enumerate(faults.WEIGHT_KEYS):
        K, N = q[key].shape
        k, n = K - 1, N - 1
        for bit in range(8):
            f = faults.flip_bits(q, [faults.position_of(layer, k, n, bit, q)])
            orig = np.int8(q[key][k, n]).view(np.uint8)
            assert np.int8(f[key][k, n]).view(np.uint8) == orig ^ np.uint8(1 << bit)
            changed = np.argwhere(f[key] != q[key])
            assert changed.tolist() == [[k, n]]


def test_sign_bit_flip_example():
    w = {"W1q": np.array([[114]], dtype=np.int8), "W2q": np.array([[0]], dtype=np.int8)}
    assert faults.flip_bits(w, [7])["W1q"][0, 0] == -14     # 0111_0010 -> 1111_0010
    assert faults.flip_bits(w, [8 + 7])["W2q"][0, 0] == -128


def test_crc8_standard_check_value():
    # CRC-8 (poly 0x07, init 0) of ASCII "123456789" is 0xF4
    assert dc.crc8(np.frombuffer(b"123456789", dtype=np.int8)) == 0xF4


@pytest.mark.parametrize("kind", ["parity", "crc8"])
def test_checksum_catches_every_single_bit_flip(qm, kind):
    q, _, _ = qm
    c = dc.ChecksumChecker(q, kind)
    assert c.count_bad_rows(q) == 0
    # exhaustive: every bit of every stored row
    for layer, k, nt, row in dc.rows(q):
        mem = row.copy()
        for i in range(len(row)):
            for bit in range(8):
                bad = mem.copy().view(np.uint8)
                bad[i] ^= np.uint8(1 << bit)
                assert not c.check_row(layer, k, nt, bad.view(np.int8))


def test_crc8_catches_every_double_flip_in_a_row():
    # CRC is linear: whether an error is caught depends only on the error pattern, not the
    # data. So checking every 2-bit pattern on a zero row covers every row of that length.
    for n_bytes in (8, 2):
        zero = np.zeros(n_bytes, dtype=np.int8)
        ref = dc.crc8(zero)
        for a, b in itertools.combinations(range(n_bytes * 8), 2):
            e = zero.copy().view(np.uint8)
            e[a // 8] ^= np.uint8(1 << (a % 8))
            e[b // 8] ^= np.uint8(1 << (b % 8))
            assert dc.crc8(e.view(np.int8)) != ref


def test_parity_misses_double_flip_in_a_row(qm):
    q, _, _ = qm
    c = dc.ChecksumChecker(q, "parity")
    f = faults.flip_bits(q, [faults.position_of(0, 0, 0, 0, q), faults.position_of(0, 0, 1, 0, q)])
    assert c.count_bad_rows(f) == 0          # known weakness, shown on purpose


@pytest.mark.parametrize("kind", ["parity", "crc8"])
def test_defense_does_not_change_clean_accuracy(qm, kind):
    q, Xq, y = qm
    c = dc.ChecksumChecker(q, kind)
    plain, st0 = accel.run_inference(Xq, q)
    checked, st1 = accel.run_inference(Xq, q, checker=c)
    np.testing.assert_array_equal(plain, checked)
    assert model.accuracy(checked, y) == model.accuracy(plain, y)
    assert st1["check_failures"] == 0
    # cost: check_latency extra cycles per tile (32 + 8 tiles)
    assert st1["total_cycles"] - st0["total_cycles"] == 40 * c.check_latency


def test_accelerator_flags_same_rows_as_direct_check(qm):
    q, Xq, _ = qm
    rng = np.random.default_rng(3)
    f = faults.flip_bits(q, rng.choice(faults.n_bits(q), size=20, replace=False))
    for kind in ["parity", "crc8"]:
        c = dc.ChecksumChecker(q, kind)
        _, st = accel.run_inference(Xq[:1], f, checker=c)
        assert st["check_failures"] == c.count_bad_rows(f) > 0


def test_memory_overhead(qm):
    q, _, _ = qm
    assert dc.ChecksumChecker(q, "parity").memory_bits() == 320
    assert dc.ChecksumChecker(q, "crc8").memory_bits() == 320 * 8


def test_bfa_gradient_matches_finite_difference(qm):
    # The straight-through gradient should point the same way as the real loss change
    # for the layer-2 weights (no rounding after layer 2, so it should be close).
    q, Xq, y = qm
    X, Y = Xq[:64], y[:64]
    g = bfa.weight_gradients(X, Y, q)["W2q"]
    base = bfa.int8_loss(X, Y, q)
    agree = 0
    idx = np.argsort(-np.abs(g).ravel())[:10]
    for flat in idx:
        k, n = np.unravel_index(flat, g.shape)
        w = dict(q)
        w["W2q"] = q["W2q"].copy()
        w["W2q"][k, n] = np.clip(int(w["W2q"][k, n]) + 1, -128, 127)
        agree += np.sign(bfa.int8_loss(X, Y, w) - base) == np.sign(g[k, n])
    assert agree == 10


def test_bfa_beats_random_quickly(qm):
    q, Xq, y = qm
    X_train, _, y_train, _ = model.load_data()
    idx = np.random.default_rng(0).choice(len(y_train), 128, replace=False)
    r = bfa.run_bfa(q, quantize.quantize_input(X_train[idx], q["s_in"]), y_train[idx], Xq, y,
                    max_flips=10)
    assert len(r["positions"]) == 10
    assert r["test_acc"][-1] < r["test_acc"][0] - 0.05
    # each round increases the attacker's loss
    assert all(b > a for a, b in zip(r["attack_loss"], r["attack_loss"][1:]))
