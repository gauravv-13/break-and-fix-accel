"""Phase 4 experiments: cost of AES-128-CTR weight encryption.

Run:  .venv/bin/python -m src.experiments_phase4

Three scenarios:
  - 1 image: weights decrypted once, then one inference (920 cycles).
  - 360 images as one batch: weights load once (weight stationary), decrypted once
    (15,280 cycles).
  - 360 images one at a time: batch-1 mode (e.g. when Phase 3 needs fresh randomness per
    image, or the weights do not stay on chip), so the weights are re-read and decrypted
    for every image (360 x 920 = 331,200 cycles).
"""
import json
from pathlib import Path

import numpy as np

from src import accel, memory_encryption as me, plotstyle, quantize

OUT = Path(__file__).resolve().parent.parent / "results" / "phase4"
KEY = bytes(range(16))            # fixed demo key/nonce (reproducible); not secret here
NONCE = bytes([0xA5] * 8)
CPIS = (1.0, 1.3, 1.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q, Xq, y = quantize.load_npz()

    # functional check: DRAM holds ciphertext, the loader decrypts, results are identical
    ct = me.encrypt_weights(q, KEY, NONCE)
    plain = me.decrypt_weights(q, ct, KEY, NONCE)
    ref, st_b = accel.run_inference(Xq, q)
    lg, _ = accel.run_inference(Xq, plain)
    assert np.array_equal(lg, ref)
    from src import faults
    changed = float(np.mean(ct != faults.weight_memory(q)))

    nb = me.n_blocks_for(q)
    _, st_1 = accel.run_inference(Xq[:1], q)
    base = {"1 image": st_1["total_cycles"], "360 images, one batch": st_b["total_cycles"],
            "360 images, one at a time": 360 * st_1["total_cycles"]}
    sessions = {"1 image": 1, "360 images, one batch": 1, "360 images, one at a time": 360}
    _, keyexp, _ = me.load_rv32i_counts()

    rows = []
    for name in me.option_cycles_per_block(1.3):
        r = {"option": name}
        for cpi in CPIS:
            label, per_block, kind = me.option_cycles_per_block(cpi)[name]
            r["label"], r["kind"] = label, kind
            r[f"cycles_per_block_cpi{cpi}"] = per_block
            for sc in base:
                e = me.extra_cycles(name, nb, cpi, sessions[sc])
                r[f"{sc}|cpi{cpi}|extra"] = e
                r[f"{sc}|cpi{cpi}|slowdown"] = (base[sc] + e) / base[sc]
        rows.append(r)
    res = {"n_blocks": nb, "weight_bytes": int(faults.n_bits(q) // 8), "baseline_cycles": base,
           "ctr_overhead_instr_assumed": me.CTR_OVERHEAD_INSTR, "key_expansion_instr_once": keyexp,
           "ciphertext_bytes_differing_from_plaintext": changed, "rows": rows}
    (OUT / "phase4_results.json").write_text(json.dumps(res, indent=2))
    write_table(res)
    make_plot(res)


def write_table(res):
    L = [f"Weights: {res['weight_bytes']} bytes = {res['n_blocks']} AES blocks. Main numbers at CPI 1.3 "
         f"(core options); range for CPI 1.0-1.5 in brackets.", "",
         "| option | cycles / block | extra cycles, 1 image (base 920) | extra, 360 images one batch (base 15,280) | extra, 360 images one at a time (base 331,200) |",
         "|---|---|---|---|---|"]
    for r in res["rows"]:
        cells = []
        for sc in res["baseline_cycles"]:
            e, s = r[f"{sc}|cpi1.3|extra"], r[f"{sc}|cpi1.3|slowdown"]
            cell = f"+{e:,.0f} ({s:.2f}x total)"
            if r["kind"] == "core" and r["option"] != "aes_processor_rv64":
                cell += f" [{r[f'{sc}|cpi1.0|slowdown']:.2f}-{r[f'{sc}|cpi1.5|slowdown']:.2f}x]"
            cells.append(cell)
        L.append(f"| {r['label']} | {r['cycles_per_block_cpi1.3']:,.1f} | " + " | ".join(cells) + " |")
    L += ["", f"One-time key expansion on RV32I: {res['key_expansion_instr_once']:,} instructions "
          f"(per key, not per inference). CTR overhead assumed {res['ctr_overhead_instr_assumed']} "
          f"instructions per block on the RV32I options."]
    (OUT / "phase4_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def make_plot(res):
    plt = plotstyle.setup()
    scen = list(res["baseline_cycles"].keys())
    rows = res["rows"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ys = np.arange(len(rows))
    h = 0.26
    for j, sc in enumerate(scen):
        v = np.array([r[f"{sc}|cpi1.3|slowdown"] for r in rows])
        lo = np.array([r[f"{sc}|cpi1.0|slowdown"] for r in rows])
        hi = np.array([r[f"{sc}|cpi1.5|slowdown"] for r in rows])
        ax.barh(ys + (j - 1) * h, v, height=h * 0.92, color=plotstyle.SERIES[j], label=sc,
                xerr=[v - lo, hi - v], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": plotstyle.MUTED})
        for yv, val, top in zip(ys + (j - 1) * h, v, hi):
            ax.text(max(val, top) * 1.12, yv, f"{val:,.2f}x", va="center", fontsize=7, color=plotstyle.TEXT)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 2000)
    ax.axvline(1, color=plotstyle.MUTED, linewidth=1)
    ax.set_yticks(ys, [r["label"].replace(" on RV32I", "\non RV32I").replace("RV32I + ", "RV32I +\n").replace(" (", "\n(") for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("total time with decryption / time without (log scale; whiskers: CPI 1.0-1.5)")
    ax.set_title("Phase 4: cost of decrypting AES-128-CTR weights before inference")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "memory_encryption_cost.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
