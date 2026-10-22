"""Phase 2 experiments: random vs targeted bit flips, and the checksum defense.

Run:  .venv/bin/python -m src.experiments_phase2

Speed note: accuracy during the experiments is computed with the fast NumPy INT8
reference (quantize.int8_forward). Phase 1 tests prove it gives bit-identical logits to
the systolic model; this script also re-checks that on the faulty models (asserts below).
"""
import json
from pathlib import Path

import numpy as np

from src import accel, bfa, faults, model, plotstyle, quantize
from src.defense_checksum import ChecksumChecker, rows

OUT = Path(__file__).resolve().parent.parent / "results" / "phase2"
N_LIST = [1, 2, 5, 10, 20, 50, 100]
N_TRIALS = 20            # accuracy trials per N (as in the plan)
N_DETECT_TRIALS = 1000   # more trials for the detection-rate estimate (cheap)
BFA_SEEDS = [0, 1, 2, 3, 4]
ATTACK_BATCH = 128


def acc(w, Xq, y):
    return model.accuracy(quantize.int8_forward(Xq, w)[0], y)


def check_accel_matches(w, Xq, y, checker=None):
    logits, st = accel.run_inference(Xq, w, checker=checker)
    assert np.array_equal(logits, quantize.int8_forward(Xq, w)[0])
    return st


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q, Xq, y = quantize.load_npz()
    total_bits = faults.n_bits(q)
    clean_acc = acc(q, Xq, y)
    checkers = {k: ChecksumChecker(q, k) for k in ["parity", "crc8"]}
    res = {"clean_acc": clean_acc, "total_weight_bits": total_bits}

    # ---------- 1) random flips: accuracy ----------
    rand = {}
    for N in N_LIST:
        accs = []
        for t in range(N_TRIALS):
            rng = np.random.default_rng(10_000 * N + t)
            pos = rng.choice(total_bits, size=N, replace=False)
            w = faults.flip_bits(q, pos)
            accs.append(acc(w, Xq, y))
            if t == 0:
                check_accel_matches(w, Xq, y)
        rand[N] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                   "min": float(np.min(accs)), "accs": accs}
        print(f"random N={N:3d}: acc {rand[N]['mean']:.4f} +- {rand[N]['std']:.4f} (min {rand[N]['min']:.4f})")
    res["random"] = rand

    # ---------- 2) targeted progressive bit search ----------
    X_train, _, y_train, _ = model.load_data()
    runs = []
    for s in BFA_SEEDS:
        idx = np.random.default_rng(s).choice(len(y_train), ATTACK_BATCH, replace=False)
        Xa = quantize.quantize_input(X_train[idx], q["s_in"])
        r = bfa.run_bfa(q, Xa, y_train[idx], Xq, y)
        check_accel_matches(r["final_weights"], Xq, y)
        # detection after every prefix of the flip sequence
        det = {}
        for kind, c in checkers.items():
            det[kind] = [c.count_bad_rows(faults.flip_bits(q, r["positions"][:i])) > 0
                         for i in range(1, len(r["positions"]) + 1)]
        # at the end of the attack: how many of the rows the attacker changed get flagged?
        changed = faults.flip_bits(q, r["positions"])
        touched = [(l, k_, nt) for (l, k_, nt, row), (_, _, _, row0) in zip(rows(changed), rows(q))
                   if not np.array_equal(row, row0)]
        flagged = {kind: c.count_bad_rows(changed) for kind, c in checkers.items()}
        bits = [p % 8 for p in r["positions"]]
        layers = [0 if p // 8 < q["W1q"].size else 1 for p in r["positions"]]
        runs.append({"seed": s, "positions": r["positions"], "test_acc": r["test_acc"],
                     "attack_loss": r["attack_loss"], "n_flips": len(r["positions"]),
                     "reached_chance": r["test_acc"][-1] <= 0.11,
                     "bit_index_counts": {b: bits.count(b) for b in range(8)},
                     "flips_in_layer1": layers.count(0), "flips_in_layer2": layers.count(1),
                     "detected_after_each_flip": det,
                     "rows_changed": len(touched), "rows_flagged": flagged})
        print(f"BFA seed {s}: {len(r['positions'])} flips -> acc {r['test_acc'][-1]:.4f}; "
              f"rows changed {len(touched)}, flagged parity {flagged['parity']} / crc8 {flagged['crc8']}")
    res["targeted"] = runs

    # ---------- 3) defense: detection of random flips ----------
    detect = {k: {} for k in checkers}
    silent = {k: {} for k in checkers}
    for N in N_LIST:
        hits = {k: 0 for k in checkers}
        silent_accs = {k: [] for k in checkers}
        for t in range(N_DETECT_TRIALS):
            rng = np.random.default_rng(1_000_000 + 10_000 * N + t)
            w = faults.flip_bits(q, rng.choice(total_bits, size=N, replace=False))
            for k, c in checkers.items():
                if c.count_bad_rows(w) > 0:
                    hits[k] += 1
                else:
                    silent_accs[k].append(acc(w, Xq, y))
        for k in checkers:
            detect[k][N] = hits[k] / N_DETECT_TRIALS
            silent[k][N] = {"n_missed": len(silent_accs[k]),
                            "worst_acc_when_missed": min(silent_accs[k]) if silent_accs[k] else None}
        print(f"detect N={N:3d}: parity {detect['parity'][N]:.3f}  crc8 {detect['crc8'][N]:.3f}")
    res["detection_random"] = detect
    res["missed_random"] = silent

    # ---------- 4) defense: several flips inside ONE row (what an informed attacker would try) ----------
    row_list = list(rows(q))
    multi = {k: {} for k in checkers}
    for m in [2, 3, 4, 5, 6]:
        hits = {k: 0 for k in checkers}
        for t in range(2000):
            rng = np.random.default_rng(2_000_000 + 10_000 * m + t)
            layer, k_, nt, r = row_list[rng.integers(len(row_list))]
            row_bits = len(r) * 8
            if m > row_bits:
                continue
            chosen = rng.choice(row_bits, size=m, replace=False)
            pos = [faults.position_of(layer, k_, nt * 8 + b // 8, b % 8, q) for b in chosen]
            w = faults.flip_bits(q, pos)
            for kind, c in checkers.items():
                hits[kind] += c.count_bad_rows(w) > 0
        for kind in checkers:
            multi[kind][m] = hits[kind] / 2000
        print(f"{m} flips in one row: parity {multi['parity'][m]:.3f}  crc8 {multi['crc8'][m]:.3f}")
    res["detection_same_row"] = multi

    # ---------- 5) defense: costs ----------
    cost = {}
    for batch_name, Xb in [("batch1", Xq[:1]), ("batch360", Xq)]:
        base = check_accel_matches(q, Xb, y)["total_cycles"]
        cost[batch_name] = {"no_check_cycles": base}
        for kind, c in checkers.items():
            st = check_accel_matches(q, Xb, y, checker=c)
            assert st["check_failures"] == 0
            cost[batch_name][kind] = {"cycles": st["total_cycles"],
                                      "extra_cycles": st["total_cycles"] - base,
                                      "extra_percent": 100 * (st["total_cycles"] - base) / base}
    for kind, c in checkers.items():
        cost[kind + "_memory"] = {"rows": len(c.stored), "extra_bits": c.memory_bits(),
                                  "extra_percent": 100 * c.memory_bits() / total_bits}
        lg, _ = accel.run_inference(Xq, q, checker=c)
        cost[kind + "_clean_acc"] = model.accuracy(lg, y)
    res["defense_cost"] = cost
    print(json.dumps(cost, indent=2))

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    write_table(res)
    make_plots(res)


def write_table(res):
    c = res["defense_cost"]
    lines = ["| | parity (1 bit/row) | CRC-8 (8 bits/row) |", "|---|---|---|"]
    lines.append(f"| extra memory bits | {c['parity_memory']['extra_bits']} "
                 f"({c['parity_memory']['extra_percent']:.2f}%) | {c['crc8_memory']['extra_bits']} "
                 f"({c['crc8_memory']['extra_percent']:.2f}%) |")
    for b, name in [("batch1", "1 image"), ("batch360", "360 images")]:
        lines.append(f"| extra cycles, {name} | +{c[b]['parity']['extra_cycles']} "
                     f"({c[b]['parity']['extra_percent']:.2f}%) | +{c[b]['crc8']['extra_cycles']} "
                     f"({c[b]['crc8']['extra_percent']:.2f}%) |")
    lines.append(f"| clean accuracy with check | {100*c['parity_clean_acc']:.2f}% | {100*c['crc8_clean_acc']:.2f}% |")
    for N in N_LIST:
        lines.append(f"| detection, {N} random flips | {100*res['detection_random']['parity'][N]:.1f}% | "
                     f"{100*res['detection_random']['crc8'][N]:.1f}% |")
    for m, v in res["detection_same_row"]["parity"].items():
        lines.append(f"| detection, {m} flips in one row | {100*v:.1f}% | "
                     f"{100*res['detection_same_row']['crc8'][m]:.1f}% |")
    tgt = res["targeted"]
    for kind in ["parity", "crc8"]:
        allp = [d for r in tgt for d in r["detected_after_each_flip"][kind]]
        res.setdefault("targeted_detect_summary", {})[kind] = sum(allp) / len(allp)
    lines.append(f"| detection, targeted (all flip prefixes, {len(tgt)} runs) | "
                 f"{100*res['targeted_detect_summary']['parity']:.1f}% | {100*res['targeted_detect_summary']['crc8']:.1f}% |")
    ch = sum(r["rows_changed"] for r in tgt)
    fl = {kind: sum(r["rows_flagged"][kind] for r in tgt) for kind in ["parity", "crc8"]}
    lines.append(f"| targeted: changed rows flagged at end of attack | {fl['parity']}/{ch} | {fl['crc8']}/{ch} |")
    (OUT / "defense_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_plots(res):
    plt = plotstyle.setup()
    blue, orange = plotstyle.SERIES[0], plotstyle.SERIES[1]

    fig, ax = plt.subplots(figsize=(6.4, 4))
    Ns = N_LIST
    m = np.array([res["random"][N]["mean"] for N in Ns]) * 100
    s = np.array([res["random"][N]["std"] for N in Ns]) * 100
    ax.fill_between(Ns, m - s, np.minimum(m + s, 100), color=blue, alpha=0.15, linewidth=0)
    ax.plot(Ns, m, marker="o", color=blue, label=f"random flips (mean ± 1 std, {N_TRIALS} trials)")
    for i, r in enumerate(res["targeted"]):
        xs = np.arange(1, len(r["test_acc"]))
        ax.plot(xs, np.array(r["test_acc"][1:]) * 100, color=orange, linewidth=1.2, alpha=0.85,
                label=f"targeted bit search ({len(res['targeted'])} runs)" if i == 0 else None)
    ax.axhline(10, color=plotstyle.MUTED, linewidth=1, linestyle="--")
    ax.text(1.05, 11.5, "chance (10%)", color=plotstyle.MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlabel("number of flipped bits in weight memory (log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title(f"Random vs targeted bit flips ({res['total_weight_bits']:,} weight bits)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.18))
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_flips.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ms = list(res["detection_same_row"]["parity"].keys())
    xs = np.arange(len(ms))
    for j, (kind, col, lab) in enumerate([("parity", blue, "parity (1 bit/row)"), ("crc8", orange, "CRC-8 (8 bits/row)")]):
        vals = [100 * res["detection_same_row"][kind][k] for k in ms]
        bars = ax.bar(xs + (j - 0.5) * 0.36, vals, width=0.34, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=plotstyle.TEXT)
    ax.set_xticks(xs, [f"{k} flips" for k in ms])
    ax.set_ylabel("trials detected (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Detection when several flips land in the SAME 8-weight row")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "detection_same_row.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
