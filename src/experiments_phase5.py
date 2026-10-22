"""Phase 5: Verilog row checkers: simulate (Icarus Verilog) and synthesize (Yosys).

Run:  .venv/bin/python -m src.experiments_phase5

1. Writes test vectors from the Python models (src/hw_vectors.py) and runs the
   self-checking testbench hw/tb/tb_row_checkers.v for 8-byte and 2-byte rows.
2. Synthesizes every checker and one PE multiply-accumulate (hw/rtl/pe_mac.v) with Yosys,
   mapped to 2-input gates (AND, NAND, OR, NOR, XOR, XNOR, ANDNOT, ORNOT), and reports
   cell counts and logic depth (longest path in gate levels, Yosys `ltp`).
3. Delay comparison: the PE's multiply-accumulate must finish in one clock cycle in our
   systolic model, so its depth sets the clock. A checker whose depth fits within one PE
   cycle can finish in 1 cycle (result registered at the end of the cycle). Depth counts
   gate levels only: it ignores wire delay and that an XOR gate is slower than a NAND, so
   it is a rough comparison, not timing analysis.
"""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from src import hw_vectors, plotstyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase5"
RTL = ROOT / "hw" / "rtl"
TB = ROOT / "hw" / "tb" / "tb_row_checkers.v"
BUILD = ROOT / "hw" / "build"
GATES = "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT"
ASSUMED_LATENCY = {"parity": 1, "crc8": 1, "secded": 2}    # Phases 2 / 2b
DESIGNS = [
    ("parity", "row_parity", {"N_BYTES": 8}, "8-byte row"),
    ("parity", "row_parity", {"N_BYTES": 2}, "2-byte row"),
    ("crc8", "row_crc8", {"N_BYTES": 8}, "8-byte row"),
    ("crc8", "row_crc8", {"N_BYTES": 2}, "2-byte row"),
    ("secded", "row_secded", {"N_BYTES": 8, "R": 7}, "8-byte row"),
    ("secded", "row_secded", {"N_BYTES": 2, "R": 5}, "2-byte row"),
    ("pe_mac", "pe_mac", {}, "one PE: 8x8 multiply + 32-bit add"),
]
# The PE reference uses Yosys's Kogge-Stone carry unit (the shallowest PE we got: 51 gate
# levels vs 60 with the default ripple-style flow; Booth did not help depth). Checkers use
# the default flow.
PE_FLOW = "kogge-stone"


def run_testbenches():
    BUILD.mkdir(parents=True, exist_ok=True)
    counts = hw_vectors.write_vectors()
    out = {}
    for nb, R in ((8, 7), (2, 5)):
        exe = BUILD / f"tb_{nb}B"
        srcs = [str(TB)] + [str(RTL / f) for f in ("row_parity.v", "row_crc8.v", "row_secded.v")]
        subprocess.run(["iverilog", "-g2005", "-P", f"tb_row_checkers.N_BYTES={nb}",
                        "-P", f"tb_row_checkers.R={R}", "-o", str(exe)] + srcs, check=True)
        res = subprocess.run(["vvp", "-n", str(exe), f"+DIR={hw_vectors.VEC_DIR}"],
                             capture_output=True, text=True, check=True)
        line = [l for l in res.stdout.splitlines() if l.startswith(("PASS", "FAIL"))][-1]
        out[f"{nb}B"] = {"result": line, "passed": line.startswith("PASS"),
                         "vectors": counts[f"parity_{nb}B"]}
        print(line)
    return out


def synth(module, params, flow="default"):
    src = RTL / f"{module}.v"
    chp = "".join(f"chparam -set {k} {v} {module}; " for k, v in params.items())
    if flow == "kogge-stone":
        mapping = (f"synth -flatten -top {module} -run begin:fine; opt -fast -full; "
                   "techmap -map +/techmap.v -map +/choices/kogge-stone.v; opt -fast; ")
    else:
        mapping = f"synth -flatten -top {module}; "
    script = (f"read_verilog {src}; {chp}{mapping}abc -g {GATES}; opt_clean; "
              "tee -o /dev/stdout stat; tee -o /dev/stdout ltp -noff")
    res = subprocess.run(["yosys", "-q", "-p", script], capture_output=True, text=True, check=True)
    txt = res.stdout
    total = int(re.search(r"^\s*(\d+)\s+cells\s*$", txt, re.M).group(1))
    types = {m.group(2): int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\s+\$_(\w+)_\s*$", txt, re.M)}
    depth = int(re.search(r"Longest topological path in \S+ \(length=(\d+)\)", txt).group(1))
    return {"cells": total, "cell_types": types, "depth": depth}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tool in ("iverilog", "vvp", "yosys"):
        assert shutil.which(tool), f"{tool} not found (install with Homebrew)"
    tb = run_testbenches()
    syn = []
    for kind, module, params, desc in DESIGNS:
        flow = PE_FLOW if kind == "pe_mac" else "default"
        r = synth(module, params, flow)
        r.update({"kind": kind, "module": module, "params": params, "desc": desc, "flow": flow})
        syn.append(r)
        print(f"{module} {params}: {r['cells']} cells, depth {r['depth']}  {r['cell_types']}")
    pe = next(r for r in syn if r["kind"] == "pe_mac")
    for r in syn:
        if r["kind"] == "pe_mac":
            continue
        r["depth_vs_pe"] = r["depth"] / pe["depth"]
        r["logic_cycles"] = max(1, math.ceil(r["depth"] / pe["depth"]))
        r["assumed_latency"] = ASSUMED_LATENCY[r["kind"]]
        r["cells_vs_64_pe"] = r["cells"] / (64 * pe["cells"])
    # per-tile latency the hardware supports: worst of the 8-byte and 2-byte versions
    latency = {k: max(r["logic_cycles"] for r in syn if r["kind"] == k) for k in ASSUMED_LATENCY}
    pe_default = synth("pe_mac", {}, "default")
    res = {"testbench": tb, "synthesis": syn, "pe_mac": pe, "pe_mac_default_flow": pe_default,
           "hw_latency_per_tile": latency,
           "assumed_latency_per_tile": ASSUMED_LATENCY,
           "tools": {"iverilog": subprocess.run(["iverilog", "-V"], capture_output=True, text=True).stdout.splitlines()[0],
                     "yosys": subprocess.run(["yosys", "-V"], capture_output=True, text=True).stdout.strip()}}
    (OUT / "phase5_results.json").write_text(json.dumps(res, indent=2))
    write_table(res)
    make_plot(res)
    return res


def write_table(res):
    pe = res["pe_mac"]
    L = [f"Yosys, mapped to 2-input gates ({GATES}). Depth = longest path in gate levels.",
         f"Reference (clock): one PE multiply-accumulate, Kogge-Stone carry = {pe['cells']} cells, "
         f"depth {pe['depth']} (64 PEs = {64 * pe['cells']:,} cells). Default Yosys flow: "
         f"{res['pe_mac_default_flow']['cells']} cells, depth {res['pe_mac_default_flow']['depth']}.", "",
         "| checker | row | cells | XOR/XNOR cells | depth (gate levels) | depth / PE depth | cycles needed | assumed in Phase 2/2b | cells vs 64 PEs |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res["synthesis"]:
        if r["kind"] == "pe_mac":
            continue
        x = r["cell_types"].get("XOR", 0) + r["cell_types"].get("XNOR", 0)
        L.append(f"| {r['kind']} | {r['desc']} | {r['cells']} | {x} | {r['depth']} | {r['depth_vs_pe']:.2f} | "
                 f"{r['logic_cycles']} | {r['assumed_latency']} | {100 * r['cells_vs_64_pe']:.2f}% |")
    L += ["", "Testbench: " + "; ".join(f"{k}: {v['result']}" for k, v in res["testbench"].items())]
    (OUT / "phase5_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def make_plot(res):
    plt = plotstyle.setup()
    rs = [r for r in res["synthesis"] if r["kind"] != "pe_mac" and r["params"]["N_BYTES"] == 8]
    pe = res["pe_mac"]
    names = [r["kind"].replace("crc8", "CRC-8").replace("secded", "SECDED") for r in rs] + ["one PE\n(MAC)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    cells = [r["cells"] for r in rs] + [pe["cells"]]
    depth = [r["depth"] for r in rs] + [pe["depth"]]
    cols = [plotstyle.SERIES[1], plotstyle.SERIES[2], plotstyle.SERIES[3], plotstyle.SERIES[0]]
    for ax, vals, title in [(a1, cells, "2-input gate cells"), (a2, depth, "logic depth (gate levels)")]:
        bars = ax.bar(names, vals, color=cols, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v}", ha="center", fontsize=9, color=plotstyle.TEXT)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.15)
    a2.axhline(pe["depth"], color=plotstyle.MUTED, linestyle="--", linewidth=1)
    a2.text(-0.4, pe["depth"] * 1.03, "one clock cycle (PE depth)", fontsize=8, color=plotstyle.MUTED)
    fig.suptitle("Phase 5: row checkers for an 8-weight row (Yosys), vs one PE", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "checker_synthesis.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
