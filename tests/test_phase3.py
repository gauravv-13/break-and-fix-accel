import numpy as np
import pytest

from src import accel, defense_sca as sca, model, quantize, systolic


@pytest.fixture(scope="module")
def qm():
    return quantize.load_npz()


def test_systolic_with_int32_weights_wraps_like_32bit_hardware():
    rng = np.random.default_rng(1)
    X = rng.integers(-128, 128, (6, 19), dtype=np.int8)
    W = rng.integers(-2**31, 2**31, (19, 13), dtype=np.int64).astype(np.int32)
    Y, _ = systolic.matmul(X, W)
    ref = ((X.astype(np.int64) @ W.astype(np.int64)) % 2**32).astype(np.uint32).view(np.int32)
    np.testing.assert_array_equal(Y, ref)


def test_goubin_conversion_is_correct():
    rng = np.random.default_rng(2)
    n = 5000
    x = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    r = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    g = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    x[:2], r[:2] = [0, 0xFFFFFFFF], [0xFFFFFFFF, 0]
    A = sca.goubin_b2a(x ^ r, r, g)
    np.testing.assert_array_equal((A.astype(np.uint64) + r) & 0xFFFFFFFF, x)


def test_masked_shares_recombine_and_are_fresh(qm):
    q, _, _ = qm
    rng = np.random.default_rng(3)
    s1, s2 = sca.store_shares(q["W1q"], rng)
    np.testing.assert_array_equal((s1 ^ s2).view(np.int8), q["W1q"])
    A1, r1, bits = sca.load_masked(s1, s2, rng)
    A2, r2, _ = sca.load_masked(s1, s2, rng)
    for A, r in [(A1, r1), (A2, r2)]:
        w = (A.view(np.uint32).astype(np.uint64) + r.view(np.uint32)) & 0xFFFFFFFF
        np.testing.assert_array_equal(w.astype(np.uint32).view(np.int32), q["W1q"].astype(np.int32))
    assert np.mean(A1 != A2) > 0.99          # every load uses new masks
    assert bits == q["W1q"].size * 64


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shuffled_inference_is_bit_identical(qm, seed):
    q, Xq, _ = qm
    ref, st = accel.run_inference(Xq, q)
    lg, s = sca.run_inference_shuffled(Xq, q, np.random.default_rng(seed))
    np.testing.assert_array_equal(lg, ref)
    assert s["total_cycles"] == st["total_cycles"]
    lg1, _ = sca.run_inference_shuffled(Xq[:20], q, np.random.default_rng(seed), per_image=True)
    np.testing.assert_array_equal(lg1, ref[:20])


@pytest.mark.parametrize("datapath", ["parallel", "serial"])
def test_masked_inference_is_bit_identical_and_costs(qm, datapath):
    q, Xq, y = qm
    ref, st = accel.run_inference(Xq, q)
    rng = np.random.default_rng(4)
    shares = {k: sca.store_shares(q[k], rng) for k in ("W1q", "W2q")}
    lg, s = sca.run_inference_masked(Xq, shares, q, rng, datapath=datapath)
    np.testing.assert_array_equal(lg, ref)
    assert model.accuracy(lg, y) == model.accuracy(ref, y)
    conv = sca.CONVERT_LATENCY * 40                       # 40 tiles per inference
    expect = st["total_cycles"] + conv if datapath == "parallel" else 2 * st["total_cycles"] + conv
    assert s["total_cycles"] == expect
    assert s["random_bits"] == (64 * 32 + 32 * 10) * 64


def test_shuffle_random_bit_counts(qm):
    q, _, _ = qm
    assert sca.fisher_yates_bits(8) == 3 + 3 + 3 + 3 + 2 + 2 + 1   # steps i = 7..1
    c = sca.shuffle_costs(q)
    assert c["random_bits"] == sum(sca.fisher_yates_bits(n) for n in (64, 32, 32, 10))
    assert c["random_bits"] >= c["random_bits_min_log2_factorial"]
    assert c["perm_register_bits"] == 64 * 6 + 32 * 5
