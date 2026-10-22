import numpy as np
import pytest

from src import accel, model, quantize, systolic


def rand_int8(rng, shape):
    return rng.integers(-128, 128, shape, dtype=np.int8)


@pytest.mark.parametrize("M,K,N", [(1, 1, 1), (1, 8, 8), (4, 8, 8), (8, 16, 24),
                                   (3, 64, 32), (360, 32, 10), (2, 64, 10)])
def test_systolic_matches_numpy_exactly(M, K, N):
    rng = np.random.default_rng(M * 1000 + K * 10 + N)
    X, W = rand_int8(rng, (M, K)), rand_int8(rng, (K, N))
    Y, _ = systolic.matmul(X, W)
    assert Y.dtype == np.int32
    np.testing.assert_array_equal(Y, quantize.int_matmul(X, W))


@pytest.mark.parametrize("M,K,N", [(5, 13, 11), (2, 9, 7), (1, 7, 9), (3, 17, 1), (6, 1, 17)])
def test_tiling_edge_cases_not_divisible_by_8(M, K, N):
    rng = np.random.default_rng(K * 100 + N)
    X, W = rand_int8(rng, (M, K)), rand_int8(rng, (K, N))
    Y, st = systolic.matmul(X, W)
    np.testing.assert_array_equal(Y, quantize.int_matmul(X, W))
    assert st["tiles"] == -(-K // 8) * -(-N // 8)
    # busy PE-cycles must equal the number of real multiplies
    assert st["busy_pe_cycles"] == M * K * N


def test_extreme_values():
    X = np.full((4, 64), -128, dtype=np.int8)
    W = np.full((64, 16), -128, dtype=np.int8)
    Y, _ = systolic.matmul(X, W)
    np.testing.assert_array_equal(Y, quantize.int_matmul(X, W))


def test_tile_cycle_counts():
    rng = np.random.default_rng(1)
    _, info = systolic.run_tile(rand_int8(rng, (8, 8)), rand_int8(rng, (10, 8)))
    assert (info["load"], info["fill"], info["compute"], info["drain"]) == (8, 7, 10, 7)
    assert info["total"] == 10 + 22
    assert info["busy_pe_cycles"] == 10 * 64


def test_quantized_weights_in_range():
    q, _, _ = quantize.load_npz()
    for k in ["W1q", "W2q"]:
        assert q[k].dtype == np.int8
        assert q[k].min() >= -127 and q[k].max() <= 127


def test_float_and_int8_accuracy_are_reasonable():
    X_train, X_test, y_train, y_test = model.load_data()
    p = model.train_float(X_train, y_train)
    q = quantize.quantize_model(p, X_train)
    fa = model.accuracy(model.float_forward(X_test, p)[0], y_test)
    ia = model.accuracy(quantize.int8_forward(quantize.quantize_input(X_test, q["s_in"]), q)[0], y_test)
    assert fa > 0.9 and ia > 0.9
    assert fa - ia < 0.02


def test_accelerator_accuracy_equals_int8_numpy():
    q, Xq, y = quantize.load_npz()
    logits_acc, stats = accel.run_inference(Xq, q)
    logits_ref, _ = quantize.int8_forward(Xq, q)
    np.testing.assert_array_equal(logits_acc, logits_ref)
    assert model.accuracy(logits_acc, y) == model.accuracy(logits_ref, y)
    assert stats["total_cycles"] == sum(s["total"] for s in stats["layers"])
