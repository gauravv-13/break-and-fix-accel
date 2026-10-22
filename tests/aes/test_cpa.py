"""Tests for simulated Hamming-weight power traces and CPA key recovery."""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.aes.traces import hamming_weight, simulate_traces
from src.aes.cpa import correlate_byte_time_series, run_cpa_on_dataset


def test_hamming_weight() -> None:
    """Verify Hamming weight values on edge and typical bytes."""
    assert hamming_weight(0x00) == 0
    assert hamming_weight(0xFF) == 8
    assert hamming_weight(0x55) == 4
    assert hamming_weight(0x80) == 1
    arr = np.array([0x00, 0x0F, 0xAA, 0xFF], dtype=np.uint8)
    np.testing.assert_array_equal(hamming_weight(arr), np.array([0.0, 4.0, 4.0, 8.0]))


def test_trace_shape_and_non_leaking_samples() -> None:
    """Check that simulated traces contain pre/post non-leaking samples around the 16 leaking samples."""
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    ds = simulate_traces(key=key, n_traces=50, sigma=0.5, seed=123, n_pre_samples=4, n_post_samples=6)
    assert ds.plaintexts.shape == (50, 16)
    assert ds.traces.shape == (50, 26)
    np.testing.assert_array_equal(ds.leakage_indices, np.arange(4, 20))


def test_cpa_noiseless_and_noisy_recovery() -> None:
    """Verify full 16-byte key recovery at sigma=0.0 and sigma=1.0, both with PoI and full-trace search."""
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")

    # Noiseless: 30 traces is more than enough for r = 1.0 on every true key byte
    ds_clean = simulate_traces(key=key, n_traces=30, sigma=0.0, seed=42)
    res_clean = run_cpa_on_dataset(ds_clean, use_poi=True)
    assert res_clean.recovered_key == key
    assert res_clean.num_correct_bytes == 16
    assert res_clean.correct_key_ranks is not None
    np.testing.assert_array_equal(res_clean.correct_key_ranks, np.ones(16, dtype=np.int64))
    for b in range(16):
        assert np.isclose(res_clean.correlations[b, key[b]], 1.0, atol=1e-6)

    # Noisy (sigma = 1.0) with 120 traces, both with PoI and full time-series search
    ds_noisy = simulate_traces(key=key, n_traces=120, sigma=1.0, seed=101)
    res_poi = run_cpa_on_dataset(ds_noisy, use_poi=True)
    assert res_poi.recovered_key == key
    assert res_poi.num_correct_bytes == 16

    res_blind = run_cpa_on_dataset(ds_noisy, use_poi=False)
    assert res_blind.recovered_key == key
    assert res_blind.num_correct_bytes == 16

    # Check that time-series correlation for byte 0 peaks at leakage_indices[0] = 4
    r_bt = correlate_byte_time_series(ds_noisy.plaintexts, ds_noisy.traces, byte_index=0)
    peak_guess, peak_time = np.unravel_index(np.argmax(np.abs(r_bt)), r_bt.shape)
    assert peak_guess == key[0]
    assert peak_time == ds_noisy.leakage_indices[0]
