"""Phase 4: weights stored in DRAM encrypted with AES-128 in counter (CTR) mode.

Functional part: the flat INT8 weight memory (faults.weight_memory, 2,368 bytes =
148 AES blocks, last block partly used) is XORed with a keystream
AES_K(nonce || counter). Decryption at load time is the same XOR. The block cipher is the
AES track's own src/aes/aes.py (checked against FIPS-197 and `cryptography` there).
Biases and scales are not encrypted here (only the INT8 weight memory, as in Phase 2).

Cost part: extra cycles per inference to produce the keystream for all 148 blocks.
  - RV32I core options use the MEASURED instruction counts per block from
    results/rv32i/aes_rv32i_results.json (AES track, pre-expanded round keys), times an
    assumed CPI (1.0 / 1.3 / 1.5, the AES track's range; 1.3 is the main value), plus
    CTR_OVERHEAD_INSTR per block (assumed, not measured).
  - The 64-bit ../AES_processor core reports 54 cycles per CTR block (its README); used
    as a reference row.
  - Dedicated AES engine beside the accelerator (assumptions): one round per cycle,
    10 rounds + 1 cycle for the XOR = 11 cycles per block ("iterative"), or a fully
    unrolled 10-stage pipeline taking one block per cycle with 11 cycles latency
    ("pipelined").
  - Main model: all weights are decrypted BEFORE the inference starts (no overlap) on the
    same clock as the accelerator. Key expansion (1,471 instructions on RV32I) happens
    once per key, not per inference, and is reported separately.
"""
import json
from pathlib import Path

import numpy as np

from src import faults
from src.aes.aes import aes128_encrypt_block

RV32I_RESULTS = Path(__file__).resolve().parent.parent / "results" / "rv32i" / "aes_rv32i_results.json"
BLOCK = 16
CTR_OVERHEAD_INSTR = 17      # assumed per block: 4 lw + 4 xor + 4 sw + counter add + loop (~3)
AES_PROCESSOR_CYCLES_PER_BLOCK = 54   # ../AES_processor README (RV64 + Zkne, CTR, measured there)
ENGINE_ROUNDS = 10


def keystream(key, nonce, n_bytes):
    """AES-128-CTR keystream: block i = AES_K(nonce (8 bytes) || i (8 bytes, big-endian))."""
    assert len(key) == 16 and len(nonce) == 8
    n_blocks = -(-n_bytes // BLOCK)
    ks = b"".join(aes128_encrypt_block(nonce + i.to_bytes(8, "big"), key) for i in range(n_blocks))
    return np.frombuffer(ks[:n_bytes], dtype=np.uint8)


def encrypt_weights(weights, key, nonce):
    """Ciphertext of the flat weight memory (what sits in DRAM), as uint8."""
    mem = faults.weight_memory(weights)
    return mem ^ keystream(key, nonce, mem.size)


def decrypt_weights(weights_template, ciphertext, key, nonce):
    """Decrypt at load time; returns a weights dict (scales/biases from the template)."""
    plain = ciphertext ^ keystream(key, nonce, ciphertext.size)
    return faults.from_memory(weights_template, plain)


def n_blocks_for(weights):
    return -(-faults.n_bits(weights) // 8 // BLOCK)


def load_rv32i_counts(path=RV32I_RESULTS):
    d = json.loads(Path(path).read_text())
    out = {k: {"label": v["label"], "instr_per_block": v["total_block_instructions"]}
           for k, v in d["variants"].items()}
    return out, d["key_expansion"]["total"], d["cpi_assumptions"]


def option_cycles_per_block(cpi=1.3):
    """Cycles per keystream block for every option (dict name -> (label, cycles, kind))."""
    counts, _, _ = load_rv32i_counts()
    opts = {}
    for k, v in counts.items():
        label = v["label"] if v["label"].startswith("RV32I") else v["label"] + " on RV32I"
        opts[k] = (label, (v["instr_per_block"] + CTR_OVERHEAD_INSTR) * cpi, "core")
    opts["aes_processor_rv64"] = ("../AES_processor RV64 + Zkne (54 cycles/block, its README)",
                                  float(AES_PROCESSOR_CYCLES_PER_BLOCK), "core")
    opts["engine_iterative"] = ("Dedicated AES engine, iterative (11 cycles/block)", float(ENGINE_ROUNDS + 1), "engine")
    opts["engine_pipelined"] = ("Dedicated AES engine, pipelined (1 block/cycle)", 1.0, "engine")
    return opts


def extra_cycles(option, n_blocks, cpi=1.3, sessions=1):
    """Extra cycles to decrypt n_blocks before each of `sessions` inferences (no overlap)."""
    label, per_block, kind = option_cycles_per_block(cpi)[option]
    if option == "engine_pipelined":
        per_session = n_blocks + ENGINE_ROUNDS       # first block after 11 cycles, then 1/cycle
    else:
        per_session = per_block * n_blocks
    return sessions * per_session
