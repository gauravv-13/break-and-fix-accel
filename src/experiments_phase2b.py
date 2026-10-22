"""Phase 2b experiments: adaptive attacker, checksums in attackable memory, SECDED.

Run:  .venv/bin/python -m src.experiments_phase2b
(needs results/phase2/results.json from experiments_phase2 for the 20 targeted runs)

Like Phase 2, accuracy is computed with the fast NumPy INT8 path; the script re-checks it
against the systolic model on some faulty models (check_accel_matches).
"""
import json
from pathlib import Path

import numpy as np

from src import accel, adaptive, faults, model, plotstyle, quantize
from src.defense_checksum import ChecksumChecker, rows
from src.defense_secded import SecdedChecker

OUT = Path(__file__).resolve().parent.parent / "results" / "phase2"
N_LIST = [1, 2, 5, 10, 20, 50, 100]
N_TRIALS = 1000
SEEDS = list(range(20))
ATTACK_BATCH = 128
DEFENSES = ["parity", "crc8", "secded"]
LABEL = {"none": "nothing (no constraint)", "parity": "parity", "crc8": "CRC-8", "secded": "SECDED"}


def acc(w, Xq, y):
    return model.accuracy(quantize.int8_forward(Xq, w)[0], y)


def check_accel_matches(w, Xq, checker=None):
    logits, st = accel.run_inference(Xq, w, checker=checker)
    if checker is None:
        assert np.array_equal(logits, quantize.int8_forward(Xq, w)[0])
    return logits, st


def make_checkers(q):
    return {"parity": ChecksumChecker(q, "parity"), "crc8": ChecksumChecker(q, "crc8"),
            "secded": SecdedChecker(q)}


class RowMap:
    """Find which rows a set of flipped bits touches, so only those rows are decoded."""

    def __init__(self, q, checker):
        self.rows = list(rows(q))
        self.byte_to_row = np.empty(faults.n_bits(q) // 8, dtype=np.int64)
        for i, (l, k, nt, r) in enumerate(self.rows):
            b0 = faults.position_of(l, k, nt * 8, 0, q) // 8
            self.byte_to_row[b0:b0 + len(r)] = i
        self.code_offsets = np.cumsum([0] + checker.widths)

    def touched(self, weight_bits, code_bits):
        a = self.byte_to_row[np.asarray(weight_bits, dtype=np.int64) // 8]
        b = np.searchsorted(self.code_offsets, np.asarray(code_bits, dtype=np.int64), side="right") - 1
        return sorted(set(a.tolist()) | set(b.tolist()))


def outcome(q, faulty, checker, rowmap, touched):
    """Classify every touched row. Returns a dict of counts."""
    c = {"false_alarm": 0, "true_alarm": 0, "corrected": 0, "miscorrected": 0, "missed": 0}
    for i in touched:
        l, k, nt, clean = rowmap.rows[i]
        row = faulty[faults.WEIGHT_KEYS[l]][k, nt * 8: nt * 8 + len(clean)]
        changed = not np.array_equal(row, clean)
        status, fixed = checker.decode_row(l, k, nt, row)
        right = np.array_equal(fixed, clean)
        if status == "detected":
            c["true_alarm" if changed else "false_alarm"] += 1
        elif not right:
            c["miscorrected" if status == "corrected" else "missed"] += 1
        elif changed:
            c["corrected"] += 1
    return c


def adaptive_section(q, Xq, y, checkers):
    X_train, _, y_train, _ = model.load_data()
    patterns = {"none": adaptive.global_patterns(q, None)}
    pattern_info = {}
    for d in DEFENSES:
        patterns[d] = adaptive.global_patterns(q, checkers[d])
        pattern_info[d] = {f"{nb}-byte rows": {"pattern_bits": adaptive.smallest_invisible(checkers[d], nb)[0],
                                               "n_patterns": len(adaptive.smallest_invisible(checkers[d], nb)[1])}
                           for nb in (8, 2)}
    print("invisible patterns:", json.dumps(pattern_info))
    clean_mem = faults.weight_memory(q)
    runs = {a: [] for a in patterns}
    for s in SEEDS:
        idx = np.random.default_rng(s).choice(len(y_train), ATTACK_BATCH, replace=False)
        Xa = quantize.quantize_input(X_train[idx], q["s_in"])
        for a, P in patterns.items():
            r = adaptive.run_pattern_attack(q, P, Xa, y_train[idx], Xq, y)
            fw = r["final_weights"]
            net = int(np.unpackbits(faults.weight_memory(fw) ^ clean_mem).sum())
            seen = {}
            for d, c in checkers.items():
                st = [st for _, st, _ in c.row_statuses(fw)]
                seen[d] = {"rows_detected": st.count("detected"), "rows_corrected": st.count("corrected")}
            fixed, _ = checkers["secded"].correct_weights(fw)
            if a != "none":   # the attack must be invisible to the defense it targets
                assert seen[a]["rows_detected"] == 0 and seen[a]["rows_corrected"] == 0
            if s == 0:
                check_accel_matches(fw, Xq)
            runs[a].append({"seed": s, "bits": r["bits"], "test_acc": r["test_acc"],
                            "bits_used": r["bits"][-1], "net_bits_changed": net,
                            "steps": len(r["patterns"]), "reached_chance": r["test_acc"][-1] <= 0.11,
                            "final_acc": r["test_acc"][-1], "seen_by": seen,
                            "acc_after_secded_correction": acc(fixed, Xq, y)})
        print(f"seed {s}: " + ", ".join(f"{a} {runs[a][-1]['bits_used']} bits" for a in patterns))
    summary = {}
    for a, rs in runs.items():
        b = [r["bits_used"] for r in rs if r["reached_chance"]]
        summary[a] = {"runs": len(rs), "reached_chance": len(b),
                      "bits_median": float(np.median(b)), "bits_mean": float(np.mean(b)),
                      "bits_std": float(np.std(b)), "bits_min": int(min(b)), "bits_max": int(max(b)),
                      "net_bits_median": float(np.median([r["net_bits_changed"] for r in rs])),
                      "detected_by": {d: sum(r["seen_by"][d]["rows_detected"] > 0 for r in rs) for d in DEFENSES},
                      "secded_touched": sum(r["seen_by"]["secded"]["rows_detected"] + r["seen_by"]["secded"]["rows_corrected"] > 0 for r in rs),
                      "acc_after_secded_median": float(np.median([r["acc_after_secded_correction"] for r in rs]))}
        print(a, summary[a])
    return {"pattern_info": pattern_info, "runs": runs, "summary": summary}


def checksum_memory_section(q, Xq, y, checkers):
    """Random flips over weight bits AND stored check bits."""
    nw = faults.n_bits(q)
    res = {}
    for d, c in checkers.items():
        rm = RowMap(q, c)
        total = nw + c.memory_bits()
        res[d] = {"weight_bits": nw, "code_bits": c.memory_bits()}
        for N in N_LIST:
            agg = {"trials_with_false_alarm": 0, "trials_with_any_alarm": 0,
                   "trials_with_silent_corruption": 0, "false_alarm_rows": 0, "alarm_rows": 0,
                   "trials_code_bits_only": 0, "trials_code_bits_only_alarm": 0}
            for t in range(N_TRIALS):
                rng = np.random.default_rng(3_000_000 + 10_000 * N + t)
                pos = rng.choice(total, size=N, replace=False)
                wb, cb = pos[pos < nw], pos[pos >= nw] - nw
                faulty = faults.flip_bits(q, wb)
                fc = c.with_flipped_stored_bits(cb)
                o = outcome(q, faulty, fc, rm, rm.touched(wb, cb))
                alarms = o["false_alarm"] + o["true_alarm"]
                agg["trials_with_false_alarm"] += o["false_alarm"] > 0
                agg["trials_with_any_alarm"] += alarms > 0
                agg["trials_with_silent_corruption"] += (o["missed"] + o["miscorrected"]) > 0
                agg["false_alarm_rows"] += o["false_alarm"]
                agg["alarm_rows"] += alarms
                if len(wb) == 0:
                    agg["trials_code_bits_only"] += 1
                    agg["trials_code_bits_only_alarm"] += alarms > 0
            res[d][N] = {
                "false_alarm_rate": agg["trials_with_false_alarm"] / N_TRIALS,
                "any_alarm_rate": agg["trials_with_any_alarm"] / N_TRIALS,
                "silent_corruption_rate": agg["trials_with_silent_corruption"] / N_TRIALS,
                "share_of_alarm_rows_that_are_false": agg["false_alarm_rows"] / max(agg["alarm_rows"], 1),
                "trials_only_code_bits_hit": agg["trials_code_bits_only"],
                "of_those_raised_alarm": agg["trials_code_bits_only_alarm"],
            }
            print(f"{d:6s} N={N:3d}: false-alarm trials {res[d][N]['false_alarm_rate']:.3f}, "
                  f"false share of alarm rows {res[d][N]['share_of_alarm_rows_that_are_false']:.3f}")
    return res


def secded_section(q, Xq, y, checkers):
    nw = faults.n_bits(q)
    out = {}
    # (a) m flips inside one row, weights only
    rm = RowMap(q, checkers["parity"])
    same_row = {d: {} for d in DEFENSES}
    for m in range(1, 7):
        counts = {d: {"corrected": 0, "detected": 0, "miscorrected": 0, "missed": 0} for d in DEFENSES}
        for t in range(2000):
            rng = np.random.default_rng(4_000_000 + 10_000 * m + t)
            i = int(rng.integers(len(rm.rows)))
            l, k, nt, r = rm.rows[i]
            local = rng.choice(len(r) * 8, size=m, replace=False)
            pos = faults.position_of(l, k, nt * 8, 0, q) + local
            faulty = faults.flip_bits(q, pos)
            for d in DEFENSES:
                o = outcome(q, faulty, checkers[d], rm, [i])
                key = "detected" if o["true_alarm"] else max(("corrected", "miscorrected", "missed"), key=lambda x: o[x])
                counts[d][key] += 1
        for d in DEFENSES:
            same_row[d][m] = {k_: v / 2000 for k_, v in counts[d].items()}
        print(f"{m} flips in one row: " + "  ".join(f"{d} {same_row[d][m]}" for d in DEFENSES))
    out["same_row"] = same_row

    # (b) random flips over the weights only: alarms and accuracy after SECDED correction
    rand = {}
    sec = checkers["secded"]
    rm_s = RowMap(q, sec)
    for N in N_LIST:
        a_none, a_fix, alarm, repaired, silent, silent_quiet = [], [], {d: 0 for d in DEFENSES}, 0, 0, 0
        for t in range(N_TRIALS):
            rng = np.random.default_rng(5_000_000 + 10_000 * N + t)
            pos = rng.choice(nw, size=N, replace=False)
            faulty = faults.flip_bits(q, pos)
            a_none.append(acc(faulty, Xq, y))
            for d in ("parity", "crc8"):
                alarm[d] += checkers[d].count_bad_rows(faulty) > 0
            o = outcome(q, faulty, sec, rm_s, rm_s.touched(pos, []))
            alarm["secded"] += o["true_alarm"] > 0
            repaired += (o["true_alarm"] + o["missed"] + o["miscorrected"]) == 0
            silent += (o["missed"] + o["miscorrected"]) > 0
            silent_quiet += (o["missed"] + o["miscorrected"]) > 0 and o["true_alarm"] == 0
            fixed, _ = sec.correct_weights(faulty) if (o["true_alarm"] + o["miscorrected"] + o["missed"]) else (q, None)
            a_fix.append(acc(fixed, Xq, y))
        rand[N] = {"acc_no_defense_mean": float(np.mean(a_none)), "acc_after_secded_mean": float(np.mean(a_fix)),
                   "acc_after_secded_min": float(np.min(a_fix)),
                   "secded_fully_repaired": repaired / N_TRIALS, "secded_silent_corruption": silent / N_TRIALS,
                   "secded_silent_corruption_no_alarm": silent_quiet / N_TRIALS,
                   "alarm_rate": {d: alarm[d] / N_TRIALS for d in DEFENSES}}
        print(f"random N={N:3d}: {rand[N]}")
    out["random_weights_only"] = rand

    # (c) the 20 non-adaptive targeted runs from Phase 2, seen by SECDED
    p2 = json.loads((OUT / "results.json").read_text())
    tgt = []
    rm_all = RowMap(q, sec)
    for r in p2["targeted"]:
        fw = faults.flip_bits(q, r["positions"])
        fixed, counts = sec.correct_weights(fw)
        o = outcome(q, fw, sec, rm_all, rm_all.touched(r["positions"], []))
        if r["seed"] == 0:
            logits, st = check_accel_matches(fw, Xq, checker=sec)
            assert np.array_equal(logits, quantize.int8_forward(Xq, fixed)[0])
        tgt.append({"seed": r["seed"], "acc_attacked": r["test_acc"][-1], "acc_after_secded": acc(fixed, Xq, y),
                    "rows_corrected": counts["corrected"], "rows_detected": counts["detected"],
                    "rows_miscorrected": o["miscorrected"], "rows_missed": o["missed"]})
    out["targeted_phase2"] = tgt
    a = [t["acc_after_secded"] for t in tgt]
    print(f"Phase-2 targeted runs after SECDED: acc median {np.median(a):.4f} min {min(a):.4f}, "
          f"runs with alarm {sum(t['rows_detected'] > 0 for t in tgt)}/{len(tgt)}")

    # (d) costs
    cost = {}
    for name, Xb in [("batch1", Xq[:1]), ("batch360", Xq)]:
        _, base = check_accel_matches(q, Xb)
        cost[name] = {"no_check": base["total_cycles"]}
        for d in DEFENSES:
            lg, st = accel.run_inference(Xb, q, checker=checkers[d])
            assert st["check_failures"] == 0 and st["corrections"] == 0
            cost[name][d] = st["total_cycles"]
    for d in DEFENSES:
        lg, _ = accel.run_inference(Xq, q, checker=checkers[d])
        cost[d] = {"memory_bits": checkers[d].memory_bits(),
                   "memory_percent": 100 * checkers[d].memory_bits() / nw,
                   "check_latency_per_tile": checkers[d].check_latency,
                   "clean_acc": model.accuracy(lg, y)}
    out["cost"] = cost
    print(json.dumps(cost))
    return out


def write_tables(res):
    ad = res["adaptive"]["summary"]
    L = ["## Adaptive attacker (20 runs each, attacker batch seeds 0-19)", "",
         "| attack | reached chance | bits flipped (median, range) | mean ± std | flagged by parity | by CRC-8 | by SECDED (detect) | acc after SECDED correction (median) |",
         "|---|---|---|---|---|---|---|---|"]
    names = {"none": "no constraint (single bits)", "parity": "parity-invisible (2-bit, same row)",
             "crc8": "CRC-8-invisible (4-bit, same row)", "secded": "SECDED-invisible (4-bit, same row)"}
    for a, s in ad.items():
        L.append(f"| {names[a]} | {s['reached_chance']}/{s['runs']} | {s['bits_median']:.0f} ({s['bits_min']}-{s['bits_max']}) | "
                 f"{s['bits_mean']:.1f} ± {s['bits_std']:.1f} | {s['detected_by']['parity']}/{s['runs']} | "
                 f"{s['detected_by']['crc8']}/{s['runs']} | {s['detected_by']['secded']}/{s['runs']} | "
                 f"{100*s['acc_after_secded_median']:.1f}% |")
    c = res["secded"]["cost"]
    L += ["", "## Cost of each defense", "",
          "| | parity | CRC-8 | SECDED |", "|---|---|---|---|",
          "| extra memory bits | " + " | ".join(f"{c[d]['memory_bits']} ({c[d]['memory_percent']:.2f}%)" for d in DEFENSES) + " |",
          "| check latency per tile (Phase 5 hardware: 1 cycle fits for all three) | " + " | ".join(str(c[d]["check_latency_per_tile"]) for d in DEFENSES) + " |"]
    for b, nm in [("batch1", "1 image"), ("batch360", "360 images")]:
        base = c[b]["no_check"]
        L.append(f"| extra cycles, {nm} (base {base}) | " + " | ".join(
            f"+{c[b][d]-base} ({100*(c[b][d]-base)/base:.2f}%)" for d in DEFENSES) + " |")
    L.append("| clean accuracy | " + " | ".join(f"{100*c[d]['clean_acc']:.2f}%" for d in DEFENSES) + " |")
    sr = res["secded"]["same_row"]
    L += ["", "## m random flips inside ONE row (2000 trials each): corrected / detected / mis-corrected / missed", "",
          "| m | parity | CRC-8 | SECDED |", "|---|---|---|---|"]
    for m in range(1, 7):
        L.append(f"| {m} | " + " | ".join(
            f"{100*sr[d][m]['corrected']:.1f} / {100*sr[d][m]['detected']:.1f} / {100*sr[d][m]['miscorrected']:.1f} / {100*sr[d][m]['missed']:.1f}"
            for d in DEFENSES) + " |")
    cm = res["checksum_memory"]
    L += ["", "## Check bits in attackable memory (1000 trials each)",
          "false-alarm rate = share of trials where a row is flagged although its weights are intact", "",
          "| random flips | parity | CRC-8 | SECDED |", "|---|---|---|---|"]
    for N in N_LIST:
        L.append(f"| {N} | " + " | ".join(
            f"{100*cm[d][N]['false_alarm_rate']:.1f}% ({100*cm[d][N]['share_of_alarm_rows_that_are_false']:.1f}% of alarms)"
            for d in DEFENSES) + " |")
    rw = res["secded"]["random_weights_only"]
    L += ["", "## Random flips in the weights only: SECDED repair (1000 trials each)", "",
          "| flips | acc, no defense (mean) | acc after SECDED (mean / worst) | fully repaired | alarm raised | silent corruption (any / with no alarm at all) |",
          "|---|---|---|---|---|---|"]
    for N in N_LIST:
        r = rw[N]
        L.append(f"| {N} | {100*r['acc_no_defense_mean']:.2f}% | {100*r['acc_after_secded_mean']:.2f}% / {100*r['acc_after_secded_min']:.2f}% | "
                 f"{100*r['secded_fully_repaired']:.1f}% | {100*r['alarm_rate']['secded']:.1f}% | "
                 f"{100*r['secded_silent_corruption']:.1f}% / {100*r['secded_silent_corruption_no_alarm']:.1f}% |")
    tg = res["secded"]["targeted_phase2"]
    L += ["", "## The 20 non-adaptive targeted runs (Phase 2) seen by SECDED", "",
          "| seed | acc after attack | acc after SECDED | rows 'corrected' (incl. mis-corrected) | rows detected (alarm) | rows mis-corrected |",
          "|---|---|---|---|---|---|"]
    for t in tg:
        L.append(f"| {t['seed']} | {100*t['acc_attacked']:.1f}% | {100*t['acc_after_secded']:.1f}% | {t['rows_corrected']} | "
                 f"{t['rows_detected']} | {t['rows_miscorrected']} |")
    (OUT / "phase2b_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def make_plots(res):
    plt = plotstyle.setup()
    col = {"none": plotstyle.SERIES[0], "parity": plotstyle.SERIES[1],
           "crc8": plotstyle.SERIES[2], "secded": plotstyle.SERIES[3]}
    runs = res["adaptive"]["runs"]

    # Plot 1: adaptive attacks, curves + bits needed
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for a, rs in runs.items():
        for i, r in enumerate(rs):
            ax.plot(r["bits"], np.array(r["test_acc"]) * 100, color=col[a], linewidth=1, alpha=0.45,
                    label=f"{LABEL[a]}" if i == 0 else None)
    ax.axhline(10, color=plotstyle.MUTED, linewidth=1, linestyle="--")
    ax.text(1, 12, "chance (10%)", color=plotstyle.MUTED, fontsize=8)
    ax.set_xlabel("bits flipped in weight memory")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Attacker that only uses flips the defense cannot see")
    leg = ax.legend(title="flips must be invisible to", loc="upper right")
    for line in leg.get_lines():
        line.set_alpha(1)
        line.set_linewidth(2)
    order = list(runs.keys())
    for j, a in enumerate(order):
        b = [r["bits_used"] for r in runs[a] if r["reached_chance"]]
        jitter = np.random.default_rng(j).uniform(-0.15, 0.15, len(b))
        ax2.scatter(b, j + jitter, s=18, color=col[a], alpha=0.8, linewidths=0)
        med = np.median(b)
        ax2.plot([med, med], [j - 0.3, j + 0.3], color=plotstyle.TEXT, linewidth=2)
        ax2.text(med + 0.8, j - 0.22, f"median {med:.0f}", ha="left", fontsize=8, color=plotstyle.TEXT)
    ax2.set_yticks(range(len(order)), ["no constraint" if a == "none" else LABEL[a] for a in order])
    ax2.set_ylim(-0.6, len(order) - 0.3)
    ax2.invert_yaxis()
    ax2.set_xlabel("bits flipped to reach chance")
    ax2.set_title("Bits needed (20 runs each)")
    ax2.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(OUT / "adaptive_attacks.png")
    plt.close(fig)

    # Plot 2: false alarms when check bits can be hit
    cm = res["checksum_memory"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for d in DEFENSES:
        v = [100 * cm[d][N]["false_alarm_rate"] for N in N_LIST]
        ax.plot(N_LIST, v, marker="o", color=col[d], label=f"{LABEL[d]} ({cm[d]['code_bits']} check bits)")
    ax.set_xscale("log")
    ax.set_xticks(N_LIST, [str(n) for n in N_LIST])
    ax.set_xlabel("random flips over weights + check bits (log scale)")
    ax.set_ylabel("trials with a false alarm (%)")
    ax.set_title("False alarms: a row is flagged but its weights are fine")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "false_alarms.png")
    plt.close(fig)

    # Plot 3: what happens to m flips inside one row, per defense
    sr = res["secded"]["same_row"]
    cats = [("corrected", "corrected", plotstyle.SERIES[2]), ("detected", "detected", plotstyle.SERIES[0]),
            ("miscorrected", "mis-corrected (silent)", plotstyle.SERIES[3]), ("missed", "missed (silent)", plotstyle.SERIES[1])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    ms = list(range(1, 7))
    for ax, d in zip(axes, DEFENSES):
        bottom = np.zeros(len(ms))
        for key, lab, c_ in cats:
            v = np.array([100 * sr[d][m][key] for m in ms])
            ax.bar(ms, v, bottom=bottom, color=c_, width=0.7, label=lab, edgecolor="white", linewidth=1)
            bottom += v
        ax.set_title(LABEL[d])
        ax.set_xticks(ms)
        ax.set_xlabel("flips in the same row")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of trials (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "same_row_outcomes.png")
    plt.close(fig)


def main():
    q, Xq, y = quantize.load_npz()
    checkers = make_checkers(q)
    res = {"adaptive": adaptive_section(q, Xq, y, checkers)}
    res["checksum_memory"] = checksum_memory_section(q, Xq, y, checkers)
    res["secded"] = secded_section(q, Xq, y, checkers)
    (OUT / "phase2b_results.json").write_text(json.dumps(res, indent=1, default=str))
    write_tables(res)
    make_plots(res)


if __name__ == "__main__":
    main()
