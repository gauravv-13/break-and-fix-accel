"""Simulated power trace generator using a Hamming-weight leakage model.

For each trace:
- Pick a random 16-byte plaintext `p`.
- For each byte index `b in 0..15`, the sensitive intermediate value in Round 1
  is `v_b = SBOX[p[b] ^ k[b]]`.
- Its instantaneous leakage is `HW(v_b) + Normal(0, sigma)`, where `HW(x)` is
  the number of 1-bits in byte `x` (ranging from 0 to 8, mean 4.0).
- Non-leaking time samples (random internal activity with baseline mean 4.0 +
  Gaussian noise `Normal(0, sigma)`) are padded before and after the 16 S-box
  operations so that each trace is a short time series.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .aes import SBOX

# Precomputed lookup tables as NumPy arrays
SBOX_LUT: np.ndarray = np.array(SBOX, dtype=np.uint8)
HW_LUT: np.ndarray = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)


def hamming_weight(values: np.ndarray | int) -> np.ndarray | int:
    """Return the Hamming weight (0..8) of byte value(s)."""
    if isinstance(values, (int, np.integer)):
        return int(HW_LUT[int(values) & 0xFF])
    return HW_LUT[np.asarray(values, dtype=np.uint8)]


@dataclass
class TraceDataset:
    """Container for simulated power traces and associated metadata."""

    plaintexts: np.ndarray       # shape (n_traces, 16), dtype uint8
    traces: np.ndarray           # shape (n_traces, n_samples), dtype float64
    key: bytes                   # 16-byte true secret key
    sigma: float                 # Gaussian noise standard deviation
    leakage_indices: np.ndarray  # shape (16,), time indices where bytes 0..15 leak


def simulate_traces(
    key: bytes,
    n_traces: int,
    sigma: float,
    seed: int | np.random.Generator = 42,
    n_pre_samples: int = 4,
    n_post_samples: int = 4,
) -> TraceDataset:
    """Simulate `n_traces` power traces for AES-128 first-round SubBytes.

    Parameters
    ----------
    key : bytes
        16-byte secret AES-128 key.
    n_traces : int
        Number of random plaintexts / power traces to capture.
    sigma : float
        Standard deviation of additive zero-mean Gaussian noise.
    seed : int or np.random.Generator
        Random seed or generator for reproducibility.
    n_pre_samples : int
        Number of non-leaking time samples before the 16 S-box operations.
    n_post_samples : int
        Number of non-leaking time samples after the 16 S-box operations.

    Returns
    -------
    TraceDataset
        Simulated plaintexts, time-series power traces, and leakage sample indices.
    """
    if len(key) != 16:
        raise ValueError(f"Expected 16-byte AES-128 key, got {len(key)} bytes")
    if n_traces <= 0:
        raise ValueError(f"n_traces must be positive, got {n_traces}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    key_arr = np.frombuffer(key, dtype=np.uint8)  # shape (16,)
    plaintexts = rng.integers(0, 256, size=(n_traces, 16), dtype=np.uint8)

    # Sensitive intermediate state: Sbox(p XOR k), shape (n_traces, 16)
    sbox_out = SBOX_LUT[plaintexts ^ key_arr[None, :]]
    leaking_signal = HW_LUT[sbox_out]  # shape (n_traces, 16)

    n_samples = n_pre_samples + 16 + n_post_samples
    leakage_indices = np.arange(n_pre_samples, n_pre_samples + 16, dtype=np.int64)

    # Non-leaking samples have baseline mean 4.0 (average HW of a random byte)
    # plus independent random byte HW variation or ambient Gaussian noise.
    traces = np.full((n_traces, n_samples), 4.0, dtype=np.float64)
    traces[:, leakage_indices] = leaking_signal

    if sigma > 0.0:
        noise = rng.normal(loc=0.0, scale=sigma, size=(n_traces, n_samples))
        traces = traces + noise

    return TraceDataset(
        plaintexts=plaintexts,
        traces=traces,
        key=key,
        sigma=float(sigma),
        leakage_indices=leakage_indices,
    )
