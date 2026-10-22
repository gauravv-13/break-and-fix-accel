"""Tests verifying simulated RV32I AES-128 ELF implementations against Python AES-128."""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.aes.aes import aes128_encrypt_block
from src.rv32i.sim import RV32ISimulator
from src.rv32i.benchmark import ELF_DEFAULT_PATH, run_rv32_aes_variant


def test_elf_full_start_execution() -> None:
    """Load `aes_rv32.elf` from `_start` and verify all 4 ciphertext buffers match FIPS-197 Appendix B."""
    sim = RV32ISimulator()
    sim.load_elf(ELF_DEFAULT_PATH)
    sim.run()

    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    expected = aes128_encrypt_block(pt, key)

    for sym in ("g_ct_tablefree", "g_ct_ttable", "g_ct_zkne", "g_ct_custom0"):
        got = sim.read_bytes(sim.symbols[sym], 16)
        assert got == expected, f"{sym} mismatch: {got.hex()} != {expected.hex()}"


def test_all_variants_match_python_aes_on_fips_and_random_vectors() -> None:
    """Check table-free, T-table, RV32 Zkne, and Custom-0 against Python AES on FIPS C.1 and 15 random blocks."""
    vectors = [
        (
            bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
            bytes.fromhex("00112233445566778899aabbccddeeff"),
        ),
        (
            bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
            bytes.fromhex("3243f6a8885a308d313198a2e0370734"),
        ),
    ]
    rng = np.random.default_rng(seed=20261022)
    for _ in range(15):
        vectors.append((rng.bytes(16), rng.bytes(16)))

    expected_counts = {
        "tablefree": 2792,
        "ttable": 921,
        "zkne": 264,
        "custom0": 197,
    }

    for variant, exp_instr in expected_counts.items():
        for key, pt in vectors:
            expected_ct = aes128_encrypt_block(pt, key)
            got_ct, blk_counts, ke_counts = run_rv32_aes_variant(variant, pt, key)
            assert got_ct == expected_ct
            # Data-independent execution count (Zkt property)
            assert blk_counts["total"] == exp_instr
            assert ke_counts["total"] == 1471
