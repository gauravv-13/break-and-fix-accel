"""Phase 3 experiments: cost of shuffling and masking (no attacks).

Run:  .venv/bin/python -m src.experiments_phase3

Every simulated option is checked to give logits bit-identical to the undefended
accelerator (so clean accuracy is unchanged).
"""
import json
from pathlib import Path

import numpy as np

from src import accel, defense_sca as sca, model, plotstyle, quantize

OUT = Path(__file__).resolve().parent.parent / "results" / "phase3"
N_WEIGHTS_TOTAL = 64 * 32 + 32 * 10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q, Xq, y = quantize.load_npz()
    M = len(Xq)
    ref, st_b = accel.run_inference(Xq, q)
    ref1, st_1 = accel.run_inference(Xq[:1], q)
    base = {"batch1": st_1["total_cycles"], "batch360": st_b["total_cycles"]}
    rng = np.random.default_rng(0)
    shares = {k: sca.store_shares(q[k], rng) for k in ("W1q", "W2q")}
    sc = sca.shuffle_costs(q)
    mreg = sca.masking_register_costs()

    rows = []

    def add(name, fresh_per_image, runner_1, runner_360, extra_storage, note=""):
        lg1, s1 = runner_1()
        lg, s = runner_360()
        assert np.array_equal(lg1, ref1) and np.array_equal(lg, ref), name
        rows.append({"option": name, "fresh_randomness_per_image": fresh_per_image,
                     "cycles_batch1": s1["total_cycles"], "cycles_batch360": s["total_cycles"],
                     "random_bits_batch1": s1["random_bits"], "random_bits_batch360": s["random_bits"],
                     "extra_storage": extra_storage, "clean_acc": model.accuracy(lg, y), "note": note})
        print(name, rows[-1]["cycles_batch1"], rows[-1]["cycles_batch360"], rows[-1]["random_bits_batch360"])

    add("no defense", "-", lambda: (ref1, {"total_cycles": base["batch1"], "random_bits": 0}),
        lambda: (ref, {"total_cycles": base["batch360"], "random_bits": 0}), "-")
    add("shuffle, one order per batch", "no (same order for all images in a batch)",
        lambda: sca.run_inference_shuffled(Xq[:1], q, rng),
        lambda: sca.run_inference_shuffled(Xq, q, rng),
        f"{sc['perm_register_bits']} bits permutation table")
    add("shuffle, new order per image", "yes (weights reload per image)",
        lambda: sca.run_inference_shuffled(Xq[:1], q, rng, per_image=True),
        lambda: sca.run_inference_shuffled(Xq, q, rng, per_image=True),
        f"{sc['perm_register_bits']} bits permutation table")
    for dp, desc in [("parallel", "2 multipliers per PE"), ("serial", "1 widened multiplier, 2 passes")]:
        storage = (f"+{mreg['extra_bits_array']} bits in the array, {mreg['multipliers_per_pe']}"
                   if dp == "parallel" else "+24 bits weight register per PE (8 -> 32), 8x32 multiplier")
        storage += f"; weight memory x2 (+{N_WEIGHTS_TOTAL * 8} bits of shares)"
        add(f"mask ({desc}), masks per batch", "no (same masks for all images in a batch)",
            lambda dp=dp: sca.run_inference_masked(Xq[:1], shares, q, rng, datapath=dp),
            lambda dp=dp: sca.run_inference_masked(Xq, shares, q, rng, datapath=dp), storage)
        add(f"mask ({desc}), new masks per image", "yes (weights reload per image)",
            lambda dp=dp: sca.run_inference_masked(Xq[:1], shares, q, rng, per_image=True, datapath=dp),
            lambda dp=dp: sca.run_inference_masked(Xq, shares, q, rng, per_image=True, datapath=dp), storage)

    # computed (formula, not simulated): refresh both shares inside each PE for every new
    # image instead of reloading: A += rho, r -= rho with 32 fresh bits per weight per image
    par = next(r for r in rows if r["option"].startswith("mask (2 multipliers") and "per batch" in r["option"])
    refresh_bits = (M - 1) * N_WEIGHTS_TOTAL * 32
    rows.append({"option": "mask (2 multipliers per PE), in-PE refresh per image [formula]",
                 "fresh_randomness_per_image": "yes (no reload; needs a refresh adder per share per PE)",
                 "cycles_batch1": par["cycles_batch1"], "cycles_batch360": par["cycles_batch360"],
                 "random_bits_batch1": par["random_bits_batch1"],
                 "random_bits_batch360": par["random_bits_batch360"] + refresh_bits,
                 "extra_storage": par["extra_storage"] + "; + refresh adders",
                 "clean_acc": par["clean_acc"],
                 "note": "assumes the refresh fits in the PE cycle (no stall); needs up to 64 PEs x 32 = 2048 random bits per cycle while streaming"})

    res = {"baseline_cycles": base, "shuffle_costs": sc, "masking_register_costs": mreg,
           "convert_latency_per_tile_assumed": sca.CONVERT_LATENCY, "options": rows}
    (OUT / "phase3_results.json").write_text(json.dumps(res, indent=2))
    write_table(res)
    make_plots(res)


def pct(v, b):
    return f"{v:,} ({100 * (v - b) / b:+.1f}%)"


def write_table(res):
    b = res["baseline_cycles"]
    L = ["| option | fresh randomness per image? | cycles, 1 image | cycles, 360 images | random bits, 1 image | random bits, 360 images | extra hardware / storage | clean acc |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["options"]:
        L.append(f"| {r['option']} | {r['fresh_randomness_per_image']} | {pct(r['cycles_batch1'], b['batch1'])} | "
                 f"{pct(r['cycles_batch360'], b['batch360'])} | {r['random_bits_batch1']:,} | {r['random_bits_batch360']:,} | "
                 f"{r['extra_storage']} | {100 * r['clean_acc']:.2f}% |")
    sc = res["shuffle_costs"]
    L += ["", f"Shuffling random bits per inference: {sc['random_bits']} (Fisher-Yates, one exact draw per step; "
          f"minimum possible log2(n!) = {sc['random_bits_min_log2_factorial']:.1f}). "
          f"If the permutation is NOT generated in the background: +{sc['extra_cycles_if_not_overlapped']} cycles per inference.",
          f"Masking: 64 random bits per weight per load (8 refresh + 24 upper-bit mask + 32 for Goubin), "
          f"{res['convert_latency_per_tile_assumed']} cycles converter latency per tile (assumed)."]
    (OUT / "phase3_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def make_plots(res):
    plt = plotstyle.setup()
    opts = res["options"]
    names = [o["option"].replace(", ", ",\n", 1) for o in opts]
    b = res["baseline_cycles"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, key, title in [(axes[0], "cycles_batch1", "1 image"), (axes[1], "cycles_batch360", "360 images")]:
        v = [o[key] for o in opts]
        ys = np.arange(len(opts))
        ax.barh(ys, v, color=plotstyle.SERIES[0], height=0.6)
        base = b["batch1" if key == "cycles_batch1" else "batch360"]
        for yv, val in zip(ys, v):
            ax.text(val * 1.08, yv, f"{val:,}  ({val / base:.2f}x)", va="center", fontsize=8, color=plotstyle.TEXT)
        if key == "cycles_batch360":
            ax.set_xscale("log")
            ax.set_xlim(right=max(v) * 12)
            ax.set_title(f"cycles, {title} (log scale)")
        else:
            for yv, val in zip(ys, v):
                ax.texts[yv].set_x(val + 30)
            ax.set_xlim(0, max(v) * 1.6)
            ax.set_title(f"cycles, {title}")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(opts)), names, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Phase 3: cycle cost of shuffling and masking (logits identical in every case)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sca_defense_cycles.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    sel = [o for o in opts if o["random_bits_batch360"] > 0]
    ys = np.arange(len(sel))
    v = [o["random_bits_batch360"] for o in sel]
    ax.barh(ys, v, color=plotstyle.SERIES[1], height=0.6)
    for yv, val in zip(ys, v):
        ax.text(val * 1.1, yv, f"{val:,}", va="center", fontsize=8, color=plotstyle.TEXT)
    ax.set_xscale("log")
    ax.set_xlim(right=max(v) * 15)
    ax.set_yticks(ys, [o["option"].replace(", ", ",\n", 1) for o in sel], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("random bits needed for 360 images (log scale)")
    ax.set_title("Randomness cost of the side-channel defenses")
    fig.tight_layout()
    fig.savefig(OUT / "sca_defense_random_bits.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
