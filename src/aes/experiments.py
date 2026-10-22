"""Run reproducible CPA experiments and generate plots in `results/aes/`."""

from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .traces import simulate_traces
from .cpa import correlate_byte_time_series, run_cpa


DEFAULT_KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
SIGMA_VALUES = [0.5, 1.0, 2.0, 4.0]
N_RUNS = 20
TRACE_GRID = list(range(10, 460, 5))


def run_all_aes_experiments(output_dir: str | Path = "results/aes", seed: int = 20261022) -> dict:
    """Run CPA experiments and save plots + JSON/CSV data to `output_dir`."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Correlation vs. key guess for one byte (Byte 0, true key = 0x2b = 43)
    ds_single = simulate_traces(
        key=DEFAULT_KEY,
        n_traces=100,
        sigma=1.0,
        seed=seed,
        n_pre_samples=4,
        n_post_samples=4,
    )
    cpa_single = run_cpa(
        plaintexts=ds_single.plaintexts,
        traces=ds_single.traces,
        true_key=DEFAULT_KEY,
        leakage_indices=ds_single.leakage_indices,
    )
    byte0_corr = cpa_single.abs_correlations[0]  # shape (256,)
    true_b0 = DEFAULT_KEY[0]
    true_b0_corr = float(byte0_corr[true_b0])
    # Second best wrong guess correlation for Byte 0
    wrong_mask = np.ones(256, dtype=bool)
    wrong_mask[true_b0] = False
    max_wrong_corr = float(np.max(byte0_corr[wrong_mask]))

    # Also compute time-series correlation for Byte 0 across all 24 time samples
    r_bt = np.abs(correlate_byte_time_series(ds_single.plaintexts, ds_single.traces, byte_index=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
    guesses = np.arange(256)
    axes[0].plot(guesses, byte0_corr, color="#94a3b8", linewidth=1.1, label="Incorrect key guesses (255)")
    axes[0].scatter(
        [true_b0],
        [true_b0_corr],
        color="#dc2626",
        s=65,
        zorder=5,
        label=f"True key byte 0x{true_b0:02x} (|r| = {true_b0_corr:.3f})",
    )
    axes[0].axhline(max_wrong_corr, color="#475569", linestyle="--", linewidth=0.9, label=f"Best wrong guess (|r| = {max_wrong_corr:.3f})")
    axes[0].set_title("CPA on Key Byte 0 (N = 100 traces, σ = 1.0)")
    axes[0].set_xlabel("Key Byte Guess (0..255)")
    axes[0].set_ylabel("Absolute Pearson Correlation |r|")
    axes[0].set_xlim(0, 255)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8.5)

    time_axis = np.arange(ds_single.traces.shape[1])
    for g in range(256):
        if g != true_b0:
            axes[1].plot(time_axis, r_bt[g], color="#cbd5e1", linewidth=0.6, alpha=0.5)
    axes[1].plot(
        time_axis,
        r_bt[true_b0],
        color="#dc2626",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label=f"True key 0x{true_b0:02x} (peaks at t={ds_single.leakage_indices[0]})",
    )
    axes[1].axvspan(3.5, 19.5, color="#fef3c7", alpha=0.35, label="Leaking window (t=4..19)")
    axes[1].set_title("Correlation Across Time Samples (Byte 0)")
    axes[1].set_xlabel("Time Sample Index t")
    axes[1].set_ylabel("Absolute Pearson Correlation |r|")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path / "correlation_vs_guess.png")
    plt.close(fig)

    # 2. Success rate (all 16 bytes correct) vs. number of traces for sigma in [0.5, 1, 2, 4]
    max_traces = max(TRACE_GRID)
    rng = np.random.default_rng(seed + 1000)

    success_curves: dict[str, list[float]] = {}
    byte_acc_curves: dict[str, list[float]] = {}
    traces_for_90pct: dict[str, int | None] = {}

    for sigma in SIGMA_VALUES:
        sigma_key = str(sigma)
        # Pre-generate `N_RUNS` full trace datasets of size `max_traces`
        run_datasets = [
            simulate_traces(
                key=DEFAULT_KEY,
                n_traces=max_traces,
                sigma=sigma,
                seed=rng,
                n_pre_samples=4,
                n_post_samples=4,
            )
            for _ in range(N_RUNS)
        ]

        full_key_sr_list: list[float] = []
        byte_sr_list: list[float] = []
        n_90: int | None = None

        for n_tr in TRACE_GRID:
            full_successes = 0
            total_correct_bytes = 0
            for ds in run_datasets:
                res = run_cpa(
                    plaintexts=ds.plaintexts[:n_tr],
                    traces=ds.traces[:n_tr],
                    true_key=DEFAULT_KEY,
                    leakage_indices=ds.leakage_indices,
                )
                assert res.num_correct_bytes is not None
                if res.num_correct_bytes == 16:
                    full_successes += 1
                total_correct_bytes += res.num_correct_bytes

            sr = full_successes / N_RUNS
            byte_sr = total_correct_bytes / (N_RUNS * 16)
            full_key_sr_list.append(sr)
            byte_sr_list.append(byte_sr)
            if n_90 is None and sr >= 0.90:
                n_90 = n_tr

        success_curves[sigma_key] = full_key_sr_list
        byte_acc_curves[sigma_key] = byte_sr_list
        traces_for_90pct[sigma_key] = n_90

    # Plot 2: Success rate vs. number of traces
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    colors = {"0.5": "#16a34a", "1.0": "#2563eb", "2.0": "#d97706", "4.0": "#dc2626"}
    markers = {"0.5": "o", "1.0": "s", "2.0": "^", "4.0": "D"}

    for sigma in SIGMA_VALUES:
        sk = str(sigma)
        n90 = traces_for_90pct[sk]
        lbl = f"σ = {sigma} (90% at N = {n90})"
        ax.plot(
            TRACE_GRID,
            success_curves[sk],
            color=colors[sk],
            marker=markers[sk],
            markevery=4,
            linewidth=2.0,
            markersize=5,
            label=lbl,
        )

    ax.axhline(0.90, color="#334155", linestyle="--", linewidth=1.1, label="90% Success Threshold")
    ax.set_title("CPA Full-Key Recovery Success Rate (All 16 Bytes) vs. Number of Traces\n(Averaged over 20 runs)")
    ax.set_xlabel("Number of Power Traces (N)")
    ax.set_ylabel("Full-Key Success Rate (All 16 Bytes Correct)")
    ax.set_xlim(min(TRACE_GRID), max(TRACE_GRID))
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path / "success_rate_vs_traces.png")
    plt.close(fig)

    # Plot 3: Traces needed for 90% success vs. sigma
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=160)
    sigmas_arr = np.array(SIGMA_VALUES, dtype=np.float64)
    n90_arr = np.array([traces_for_90pct[str(s)] for s in SIGMA_VALUES], dtype=np.float64)

    ax.plot(
        sigmas_arr,
        n90_arr,
        color="#1d4ed8",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Measured traces for ≥90% full-key recovery",
    )
    for s_val, n_val in zip(sigmas_arr, n90_arr):
        ax.annotate(
            f"N = {int(n_val)}",
            (s_val, n_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Traces Required for 90% Full-Key CPA Success vs. Noise σ")
    ax.set_xlabel("Gaussian Noise Standard Deviation (σ)")
    ax.set_ylabel("Traces Needed for ≥90% Success (N)")
    ax.set_xlim(0.2, 4.3)
    ax.set_ylim(0, float(np.max(n90_arr)) * 1.18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path / "traces_for_90pct_vs_sigma.png")
    plt.close(fig)

    # Save CSV summary
    with open(out_path / "cpa_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "snr_theoretical", "traces_for_90pct_success"])
        for s in SIGMA_VALUES:
            snr = 2.0 / (s ** 2)  # Var(HW of uniform byte) = 8 * 0.25 = 2.0
            writer.writerow([s, f"{snr:.4f}", traces_for_90pct[str(s)]])

    summary = {
        "seed": seed,
        "n_runs": N_RUNS,
        "sigma_values": SIGMA_VALUES,
        "trace_grid": TRACE_GRID,
        "single_byte_demo": {
            "n_traces": 100,
            "sigma": 1.0,
            "byte_index": 0,
            "true_key_byte_hex": f"0x{true_b0:02x}",
            "true_key_abs_corr": round(true_b0_corr, 4),
            "max_wrong_guess_abs_corr": round(max_wrong_corr, 4),
            "num_correct_bytes": cpa_single.num_correct_bytes,
            "correct_key_ranks": cpa_single.correct_key_ranks.tolist() if cpa_single.correct_key_ranks is not None else [],
        },
        "traces_for_90pct": traces_for_90pct,
        "success_curves": success_curves,
        "byte_acc_curves": byte_acc_curves,
    }
    with open(out_path / "cpa_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_all_aes_experiments()
    print("Single-byte demo (N=100, sigma=1.0):", res["single_byte_demo"])
    print("Traces needed for >=90% full-key recovery:", res["traces_for_90pct"])
