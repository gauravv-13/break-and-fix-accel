"""Phase 1 experiments: train + quantize, run the accelerator, make tables and plots.

Run:  .venv/bin/python -m src.experiments_phase1
"""
import json

import numpy as np

from src import accel, model, plotstyle, quantize, systolic

OUT = quantize.RESULTS


def main():
    r = quantize.build_and_save()
    q, Xq, y = quantize.load_npz()
    ref_logits, _ = quantize.int8_forward(Xq, q)

    # predictions that changed between float and INT8
    X_train, X_test, y_train, y_test = model.load_data()
    p = model.train_float(X_train, y_train)
    float_pred = np.argmax(model.float_forward(X_test, p)[0], axis=1)
    int8_pred = np.argmax(ref_logits, axis=1)

    # accelerator: whole test set as one batch, and a single image (batch = 1)
    logits, st_batch = accel.run_inference(Xq, q)
    _, st_one = accel.run_inference(Xq[:1], q)
    assert np.array_equal(logits, ref_logits)

    summary = {
        "float_acc": r["float_acc"], "int8_acc": r["int8_acc"],
        "accel_acc": model.accuracy(logits, y),
        "predictions_changed_by_int8": int(np.sum(float_pred != int8_pred)),
        "n_test": int(len(y)), "scales": {k: q[k] for k in ["s_in", "s_w1", "s_h", "s_w2", "m1"]},
        "layers": {},
    }
    rows = ["| layer | shape (K x N) | batch | tiles | load | fill | compute | drain | total cycles | utilization |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, st in [("batch=360", st_batch), ("batch=1", st_one)]:
        for i, (ls, wk) in enumerate(zip(st["layers"], ["W1q", "W2q"])):
            K, N = q[wk].shape
            summary["layers"][f"layer{i+1}_{name}"] = {k: ls[k] for k in
                ["tiles", "load", "fill", "compute", "drain", "total", "busy_pe_cycles", "utilization"]}
            rows.append(f"| {i+1} | {K} x {N} | {name[6:]} | {ls['tiles']} | {ls['load']} | {ls['fill']} | "
                        f"{ls['compute']} | {ls['drain']} | {ls['total']} | {100*ls['utilization']:.1f}% |")
        summary[f"total_cycles_{name}"] = st["total_cycles"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "cycles_table.md").write_text("\n".join(rows) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))
    print("\n".join(rows))

    plt = plotstyle.setup()

    # Plot 1: utilization per layer, for batch 360 and batch 1
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = ["Layer 1 (64x32)", "Layer 2 (32x10)"]
    xs = np.arange(2)
    for j, (name, st, col) in enumerate([("whole test set (batch 360)", st_batch, plotstyle.SERIES[0]),
                                         ("one image (batch 1)", st_one, plotstyle.SERIES[1])]):
        vals = [100 * s["utilization"] for s in st["layers"]]
        bars = ax.bar(xs + (j - 0.5) * 0.36, vals, width=0.34, color=col, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, color=plotstyle.TEXT)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("PE utilization (%)")
    ax.set_ylim(0, 105)
    ax.set_title("8x8 systolic array: share of PE-cycles doing useful work")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "utilization_per_layer.png")
    plt.close(fig)

    # Plot 2: cycles vs matrix size (square M = K = N), simulated, against the formula
    sizes = [8, 16, 24, 32, 48, 64, 96, 128]
    rng = np.random.default_rng(0)
    sim, util = [], []
    for n in sizes:
        X = rng.integers(-128, 128, (n, n), dtype=np.int8)
        W = rng.integers(-128, 128, (n, n), dtype=np.int8)
        Y, st = systolic.matmul(X, W)
        assert np.array_equal(Y, quantize.int_matmul(X, W))
        sim.append(st["total"])
        util.append(st["utilization"])
    formula = [(-(-n // 8)) ** 2 * (n + 22) for n in sizes]
    assert sim == formula
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(sizes, sim, marker="o", color=plotstyle.SERIES[0])
    for n, c, u in zip(sizes, sim, util):
        if n in (8, 32, 64, 128):
            ax.annotate(f"{c:,} cycles\n{100*u:.0f}% util.", (n, c), textcoords="offset points",
                        xytext=(8, -18), ha="left", fontsize=8, color=plotstyle.MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlim(6.5, 200)
    ax.set_xlabel("matrix size n  (n x n inputs times n x n weights)")
    ax.set_ylabel("total cycles (log scale)")
    ax.set_title("Cycles grow about as n^3 / 64 for an 8x8 array")
    fig.tight_layout()
    fig.savefig(OUT / "cycles_vs_size.png")
    plt.close(fig)
    with open(OUT / "cycles_vs_size.csv", "w") as f:
        f.write("n,total_cycles,utilization\n")
        for n, c, u in zip(sizes, sim, util):
            f.write(f"{n},{c},{u:.4f}\n")


if __name__ == "__main__":
    main()
