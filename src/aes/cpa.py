"""Correlation Power Analysis (CPA) on AES-128 first-round S-box leakage."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .traces import SBOX_LUT, HW_LUT, TraceDataset


@dataclass
class CPAResult:
    """Results of a Correlation Power Analysis attack across all 16 key bytes."""

    recovered_key: bytes                  # 16-byte recovered key
    correlations: np.ndarray              # shape (16, 256), signed Pearson r at PoI
    abs_correlations: np.ndarray          # shape (16, 256), |r| used for ranking
    num_correct_bytes: int | None         # number of correctly recovered bytes (0..16)
    correct_key_ranks: np.ndarray | None  # shape (16,), 1-based rank of true key byte (1 = best)


def compute_hypothetical_hw(plaintexts: np.ndarray) -> np.ndarray:
    """Compute predicted HW(Sbox(p XOR g)) for all traces, 16 bytes, and 256 guesses.

    Parameters
    ----------
    plaintexts : np.ndarray
        Array of shape `(n_traces, 16)` with dtype `uint8`.

    Returns
    -------
    np.ndarray
        Array of shape `(n_traces, 16, 256)` with dtype `float64`.
    """
    guesses = np.arange(256, dtype=np.uint8).reshape(1, 1, 256)
    sbox_inputs = plaintexts[:, :, None] ^ guesses
    return HW_LUT[SBOX_LUT[sbox_inputs]]


def correlate_byte_time_series(
    plaintexts: np.ndarray,
    traces: np.ndarray,
    byte_index: int,
) -> np.ndarray:
    """Compute Pearson correlation across all 256 guesses and all time samples for one byte.

    Returns an array of shape `(256, n_samples)`.
    """
    p_byte = plaintexts[:, byte_index : byte_index + 1]  # (N, 1)
    guesses = np.arange(256, dtype=np.uint8).reshape(1, 256)
    h = HW_LUT[SBOX_LUT[p_byte ^ guesses]]  # (N, 256)

    h_centered = h - h.mean(axis=0, keepdims=True)       # (N, 256)
    t_centered = traces - traces.mean(axis=0, keepdims=True)  # (N, T)

    num = h_centered.T @ t_centered  # (256, T)
    den_h = np.sqrt(np.sum(h_centered ** 2, axis=0))[:, None]  # (256, 1)
    den_t = np.sqrt(np.sum(t_centered ** 2, axis=0))[None, :]  # (1, T)
    denom = den_h * den_t
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-12)


def run_cpa(
    plaintexts: np.ndarray,
    traces: np.ndarray,
    true_key: bytes | None = None,
    leakage_indices: np.ndarray | None = None,
) -> CPAResult:
    """Run CPA on `plaintexts` and `traces` to recover all 16 AES-128 key bytes.

    Parameters
    ----------
    plaintexts : np.ndarray
        Array of shape `(n_traces, 16)` of `uint8`.
    traces : np.ndarray
        Array of shape `(n_traces, n_samples)` of `float64`.
    true_key : bytes, optional
        If provided, computes `num_correct_bytes` and `correct_key_ranks`.
    leakage_indices : np.ndarray, optional
        Time sample indices of shape `(16,)` corresponding to bytes 0..15.
        If `None` and `traces.shape[1] == 16`, uses `0..15`. If `None` and
        `traces.shape[1] != 16`, correlates against all time samples and selects
        the maximum absolute correlation across time for each guess.

    Returns
    -------
    CPAResult
    """
    n_traces = plaintexts.shape[0]
    if n_traces < 2:
        raise ValueError("CPA requires at least 2 traces to compute Pearson correlation")

    if leakage_indices is not None or traces.shape[1] == 16:
        idx = np.arange(16) if leakage_indices is None else np.asarray(leakage_indices)
        y = traces[:, idx]  # (N, 16)
        h = compute_hypothetical_hw(plaintexts)  # (N, 16, 256)

        h_c = h - h.mean(axis=0, keepdims=True)  # (N, 16, 256)
        y_c = y - y.mean(axis=0, keepdims=True)  # (N, 16)

        num = np.einsum("nbg,nb->bg", h_c, y_c)  # (16, 256)
        den_h = np.sqrt(np.sum(h_c ** 2, axis=0))  # (16, 256)
        den_y = np.sqrt(np.sum(y_c ** 2, axis=0))[:, None]  # (16, 1)
        denom = den_h * den_y
        correlations = np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-12)
        abs_correlations = np.abs(correlations)
    else:
        # Unknown PoI: correlate against all time samples and pick best time sample per guess
        correlations = np.zeros((16, 256), dtype=np.float64)
        abs_correlations = np.zeros((16, 256), dtype=np.float64)
        for b in range(16):
            r_bt = correlate_byte_time_series(plaintexts, traces, b)  # (256, T)
            best_t = np.argmax(np.abs(r_bt), axis=1)
            correlations[b] = r_bt[np.arange(256), best_t]
            abs_correlations[b] = np.abs(correlations[b])

    best_guesses = np.argmax(abs_correlations, axis=1).astype(np.uint8)
    recovered_key = bytes(best_guesses.tolist())

    num_correct_bytes: int | None = None
    correct_key_ranks: np.ndarray | None = None

    if true_key is not None:
        true_arr = np.frombuffer(true_key, dtype=np.uint8)
        num_correct_bytes = int(np.sum(best_guesses == true_arr))
        ranks = np.zeros(16, dtype=np.int64)
        for b in range(16):
            # Sort guesses by |r| descending; 1-based rank of true_arr[b]
            order = np.argsort(-abs_correlations[b])
            ranks[b] = int(np.where(order == true_arr[b])[0][0]) + 1
        correct_key_ranks = ranks

    return CPAResult(
        recovered_key=recovered_key,
        correlations=correlations,
        abs_correlations=abs_correlations,
        num_correct_bytes=num_correct_bytes,
        correct_key_ranks=correct_key_ranks,
    )


def run_cpa_on_dataset(dataset: TraceDataset, use_poi: bool = True) -> CPAResult:
    """Convenience wrapper to run CPA directly on a `TraceDataset`."""
    return run_cpa(
        plaintexts=dataset.plaintexts,
        traces=dataset.traces,
        true_key=dataset.key,
        leakage_indices=dataset.leakage_indices if use_poi else None,
    )
