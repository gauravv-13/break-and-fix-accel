"""Benchmark RV32I software AES-128 vs. custom AES instruction extensions on `RV32ISimulator`."""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.aes.aes import aes128_encrypt_block
from .sim import RV32ISimulator, encode_i

ELF_DEFAULT_PATH = Path(__file__).resolve().parent / "c_src" / "aes_rv32.elf"
HALT_STUB_ADDR = 0x8001F000
STACK_TOP_ADDR = 0x80020000
EBREAK_INSTR = encode_i(0x73, 0, 0x0, 0, 1)


def call_elf_function(
    sim: RV32ISimulator,
    func_name: str,
    a0: int = 0,
    a1: int = 0,
    a2: int = 0,
) -> Dict[str, int]:
    """Call a C function in the loaded ELF and return exact instruction counts for that call."""
    if func_name not in sim.symbols:
        raise KeyError(f"Symbol {func_name!r} not found in ELF symbols")

    sim.write_u32(HALT_STUB_ADDR, EBREAK_INSTR)
    sim.regs[2] = STACK_TOP_ADDR   # sp (x2)
    sim.regs[1] = HALT_STUB_ADDR   # ra (x1) -> returns to EBREAK
    sim.regs[10] = a0 & 0xFFFFFFFF  # a0 (x10)
    sim.regs[11] = a1 & 0xFFFFFFFF  # a1 (x11)
    sim.regs[12] = a2 & 0xFFFFFFFF  # a2 (x12)
    sim.pc = sim.symbols[func_name]
    sim.halted = False
    sim.reset_counts()
    sim.run()

    counts = dict(sim.counts)
    # Subtract the synthetic halting EBREAK at HALT_STUB_ADDR so only the function's own
    # instructions (including its final `ret` / `jalr`) are counted.
    if counts["system"] > 0:
        counts["system"] -= 1
    counts["total"] = sum(counts.values())
    return counts


def run_rv32_aes_variant(
    variant: str,
    plaintext: bytes,
    key: bytes,
    elf_path: str | Path = ELF_DEFAULT_PATH,
) -> tuple[bytes, Dict[str, int], Dict[str, int]]:
    """Run one AES-128 variant on the RV32I simulator and return (ciphertext, block_counts, key_exp_counts)."""
    sim = RV32ISimulator()
    sim.load_elf(elf_path)

    if variant == "ttable":
        call_elf_function(sim, "init_ttables")

    addr_key = sim.symbols["g_key"]
    addr_pt = sim.symbols["g_pt"]
    addr_rk = sim.symbols["g_rk"]

    func_map = {
        "tablefree": ("aes128_tablefree_block", "g_ct_tablefree"),
        "ttable": ("aes128_ttable_block", "g_ct_ttable"),
        "zkne": ("aes128_zkne_block", "g_ct_zkne"),
        "custom0": ("aes128_custom0_block", "g_ct_custom0"),
    }
    if variant not in func_map:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {list(func_map)}")

    func_name, ct_sym = func_map[variant]
    addr_ct = sim.symbols[ct_sym]

    sim.write_bytes(addr_key, key)
    sim.write_bytes(addr_pt, plaintext)

    key_exp_counts = call_elf_function(sim, "aes128_expand_key", a0=addr_key, a1=addr_rk)
    block_counts = call_elf_function(sim, func_name, a0=addr_ct, a1=addr_pt, a2=addr_rk)
    ciphertext = sim.read_bytes(addr_ct, 16)
    return ciphertext, block_counts, key_exp_counts


def run_all_rv32i_benchmarks(output_dir: str | Path = "results/rv32i") -> Dict[str, Any]:
    """Benchmark all AES-128 implementations on RV32ISimulator and save tables and plots."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # FIPS-197 Appendix B test vector
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    expected_ct = aes128_encrypt_block(pt, key)

    variants = [
        ("tablefree", "Software AES (Table-Free)"),
        ("ttable", "Software AES (T-Table)"),
        ("zkne", "RV32I + Zkne (aes32esmi/esi)"),
        ("custom0", "RV32I + Custom-0 (aes_round_col)"),
    ]
    cpi_assumptions = [1.0, 1.3, 1.5]

    results: Dict[str, Any] = {
        "compiler": "riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O2",
        "cpi_assumptions": cpi_assumptions,
        "key_expansion": {},
        "variants": {},
    }

    rows_for_table = []
    for var_id, label in variants:
        ct, blk_counts, ke_counts = run_rv32_aes_variant(var_id, pt, key)
        if ct != expected_ct:
            raise AssertionError(f"{var_id} ciphertext {ct.hex()} != expected {expected_ct.hex()}")

        results["key_expansion"] = ke_counts
        total_blk = blk_counts["total"]
        total_with_ke = total_blk + ke_counts["total"]

        cycles_blk = {f"cpi_{cpi:.1f}": round(total_blk * cpi, 1) for cpi in cpi_assumptions}
        cycles_with_ke = {f"cpi_{cpi:.1f}": round(total_with_ke * cpi, 1) for cpi in cpi_assumptions}

        entry = {
            "label": label,
            "block_counts": blk_counts,
            "total_block_instructions": total_blk,
            "total_with_key_expansion": total_with_ke,
            "cycles_per_block": cycles_blk,
            "cycles_with_key_expansion": cycles_with_ke,
        }
        results["variants"][var_id] = entry
        rows_for_table.append((var_id, label, blk_counts, total_blk, total_with_ke, cycles_blk))

    # Compute speedup relative to table-free baseline
    baseline_instr = results["variants"]["tablefree"]["total_block_instructions"]
    for var_id in results["variants"]:
        t_instr = results["variants"][var_id]["total_block_instructions"]
        results["variants"][var_id]["speedup_vs_tablefree"] = round(baseline_instr / t_instr, 2)

    # Save CSV
    with open(out_path / "aes_rv32i_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant_id",
            "implementation",
            "alu",
            "load",
            "store",
            "branch",
            "jump",
            "custom",
            "total_instr_per_block",
            "total_instr_incl_key_exp",
            "cycles_cpi_1.0",
            "cycles_cpi_1.3",
            "cycles_cpi_1.5",
            "speedup_vs_tablefree",
        ])
        for var_id, label, bc, tot_blk, tot_ke, cyc in rows_for_table:
            writer.writerow([
                var_id,
                label,
                bc["alu"],
                bc["load"],
                bc["store"],
                bc["branch"],
                bc["jump"],
                bc["custom"],
                tot_blk,
                tot_ke,
                cyc["cpi_1.0"],
                cyc["cpi_1.3"],
                cyc["cpi_1.5"],
                results["variants"][var_id]["speedup_vs_tablefree"],
            ])

    # Save Markdown table
    md_lines = [
        "# RV32I AES-128 Instruction Counts & Cycle Estimates (Per 16-Byte Block)",
        "",
        "Compiled with `/opt/homebrew/bin/riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O2` and executed on `src/rv32i/sim.py`.",
        f"One-time key expansion (`aes128_expand_key`) takes **{results['key_expansion']['total']} instructions** "
        f"(ALU: {results['key_expansion']['alu']}, Load: {results['key_expansion']['load']}, "
        f"Store: {results['key_expansion']['store']}, Branch: {results['key_expansion']['branch']}, "
        f"Jump: {results['key_expansion']['jump']}).",
        "",
        "| Implementation | ALU | Load | Store | Branch | Jump | Custom | Total Instr / Block | Cycles (CPI=1.0) | Cycles (CPI=1.3) | Cycles (CPI=1.5) | Speedup vs. Table-Free |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for var_id, label, bc, tot_blk, tot_ke, cyc in rows_for_table:
        spd = results["variants"][var_id]["speedup_vs_tablefree"]
        md_lines.append(
            f"| {label} | {bc['alu']} | {bc['load']} | {bc['store']} | {bc['branch']} | "
            f"{bc['jump']} | {bc['custom']} | **{tot_blk}** | {cyc['cpi_1.0']:.0f} | "
            f"{cyc['cpi_1.3']:.1f} | {cyc['cpi_1.5']:.1f} | **{spd:.2f}x** |"
        )
    with open(out_path / "aes_rv32i_table.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    with open(out_path / "aes_rv32i_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot bar chart: Left = Stacked instruction mix per block; Right = Estimated cycles across CPIs
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=160)
    short_labels = [
        "SW Table-Free\n(RV32I)",
        "SW T-Table\n(RV32I)",
        "RV32I + Zkne\n(aes32esmi/esi)",
        "RV32I + Custom-0\n(aes_round_col)",
    ]
    x = np.arange(len(rows_for_table))

    categories = [
        ("alu", "ALU", "#3b82f6"),
        ("load", "Load", "#10b981"),
        ("store", "Store", "#f59e0b"),
        ("branch", "Branch", "#ef4444"),
        ("jump", "Jump", "#8b5cf6"),
        ("custom", "Custom AES", "#ec4899"),
    ]
    bottoms = np.zeros(len(rows_for_table))
    for cat_key, cat_label, color in categories:
        vals = np.array([r[2][cat_key] for r in rows_for_table], dtype=np.float64)
        axes[0].bar(x, vals, width=0.52, bottom=bottoms, label=cat_label, color=color)
        bottoms += vals

    for idx, tot in enumerate(bottoms):
        spd = results["variants"][rows_for_table[idx][0]]["speedup_vs_tablefree"]
        axes[0].text(
            idx,
            tot + max(bottoms) * 0.02,
            f"{int(tot)}\n({spd:.2f}x)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    axes[0].set_title("Instruction Mix per 16-Byte AES-128 Block")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, fontsize=8.5)
    axes[0].set_ylabel("Instructions Executed per Block")
    axes[0].set_ylim(0, max(bottoms) * 1.18)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8.5)

    # Panel 2: Grouped bar chart for CPI = 1.0, 1.3, 1.5
    width = 0.24
    cpi_colors = ["#2563eb", "#d97706", "#dc2626"]
    for i, (cpi, col) in enumerate(zip(cpi_assumptions, cpi_colors)):
        cyc_vals = [r[5][f"cpi_{cpi:.1f}"] for r in rows_for_table]
        offset = (i - 1) * width
        bars = axes[1].bar(x + offset, cyc_vals, width=width, color=col, label=f"CPI = {cpi:.1f}")
        for b_rect in bars:
            h = b_rect.get_height()
            axes[1].text(
                b_rect.get_x() + b_rect.get_width() / 2.0,
                h + max(bottoms) * 1.5 * 0.015,
                f"{int(round(h))}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    axes[1].set_title("Estimated Clock Cycles per 16-Byte Block\n(5-Stage Pipeline CPI = 1.0, 1.3, 1.5)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, fontsize=8.5)
    axes[1].set_ylabel("Estimated Cycles per 16-Byte Block")
    axes[1].set_ylim(0, max(bottoms) * 1.5 * 1.16)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path / "aes_rv32i_comparison.png")
    plt.close(fig)

    return results


if __name__ == "__main__":
    res = run_all_rv32i_benchmarks()
    print("Key expansion instructions:", res["key_expansion"])
    for k, v in res["variants"].items():
        print(f"{k:10s} -> {v['total_block_instructions']:4d} instr/block ({v['speedup_vs_tablefree']:.2f}x), cycles: {v['cycles_per_block']}")
