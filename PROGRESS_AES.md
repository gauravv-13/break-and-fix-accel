# AES & RV32I Track Progress Notebook (`PROGRESS_AES.md`)

## Part A: AES-128 From Scratch & Simulated Power Side-Channel Analysis (CPA)

### What Was Built
- `src/aes/aes.py`: Clean, textbook AES-128 implementation from scratch (`sub_bytes`, `shift_rows`, `mix_columns`, `add_round_key`, `key_expansion`, `aes128_encrypt_block`, and `aes128_encrypt_rounds`).
- `tests/aes/test_aes.py`: Verified against FIPS-197 Appendix A.1 (key schedule of `2b7e15...`), Appendix B (full cipher example + Round 1 intermediate states), Appendix C.1 (`000102...0f` vector), and 1,000 random `(key, plaintext)` pairs checked against Python's `cryptography` library.
- `src/aes/traces.py`: Simulated Hamming-weight power trace generator (`HW(Sbox(p XOR k)) + Normal(0, sigma)`) padded with 4 pre-operation and 4 post-operation non-leaking time samples (24 time samples per trace).
- `src/aes/cpa.py` & `tests/aes/test_cpa.py`: Vectorized Pearson Correlation Power Analysis across all 16 bytes and 256 key guesses (supporting both known Point-of-Interest and full time-series PoI discovery).
- `src/aes/experiments.py` & `results/aes/`: Generated reproducible plots (`correlation_vs_guess.png`, `success_rate_vs_traces.png`, `traces_for_90pct_vs_sigma.png`) and data (`cpa_summary.csv`, `cpa_results.json`).
- `docs/aes_notes.md`: Plain-language explanation of why CPA works and summary of experimental numbers.

### Key Numbers
- **Verification**: 100% match on FIPS-197 Appendix A.1, B, C.1, and 1,000/1,000 random blocks vs. `cryptography`.
- **Single-byte CPA demo ($N = 100$ traces, $\sigma = 1.0$, key byte 0 = `0x2b`)**:
  - Correct key guess (`0x2b`) correlation: $|r| = 0.8454$ (Rank 1).
  - Best wrong key guess correlation: $|r| = 0.4164$.
  - Full key recovery at $N = 100, \sigma = 1.0$: 16/16 bytes correct, all Rank 1.
- **Traces needed for $\ge 90\%$ full-key recovery (all 16 bytes simultaneously correct, averaged over 20 runs)**:
  - $\sigma = 0.5$ ($\text{SNR} = 8.0$): **15 traces**
  - $\sigma = 1.0$ ($\text{SNR} = 2.0$): **30 traces**
  - $\sigma = 2.0$ ($\text{SNR} = 0.5$): **80 traces**
  - $\sigma = 4.0$ ($\text{SNR} = 0.125$): **275 traces**

### Known Problems / Limitations
- The leakage model assumes an unprotected byte-serial Hamming-weight leakage without clock jitter or Boolean masking. Adding first-order Boolean masking or randomizing the S-box execution order (shuffling) would break first-order CPA or increase the required traces quadratically.

### Three Questions I Should Be Able to Answer (Part A)
1. **Why can CPA attack AES-128 one byte at a time ($16 \times 256 = 4{,}096$ guesses) instead of guessing all $2^{128}$ keys at once?**
   Because in Round 1, `AddRoundKey` followed by `SubBytes` computes $\text{S-box}(p_i \oplus k_i)$ on each byte independently before `MixColumns` mixes bytes across columns.
2. **Why does a key guess that is off by only 1 bit still have nearly zero correlation with the true power trace?**
   Because the AES S-box is strongly non-linear (based on multiplicative inversion in $\text{GF}(2^8)$), so flipping 1 input bit scrambles the output bits across different plaintexts, making the predicted Hamming weight uncorrelated with the actual Hamming weight.
3. **Why does doubling the noise standard deviation $\sigma$ require roughly $3.5\times\text{–}4\times$ as many traces for key recovery?**
   Because the Signal-to-Noise Ratio is $\text{SNR} = \text{Var}(\text{HW}) / \sigma^2 = 2 / \sigma^2$, and the sample variance of the Pearson correlation estimator decreases as $1/N$, meaning $N \propto 1/\text{SNR} \propto \sigma^2$.

---

## Part B: RV32I Instruction Simulator & Software vs. Custom-Instruction AES-128

### Compiler & Toolchain Route Used
- Checked `which riscv64-unknown-elf-gcc` -> found `/opt/homebrew/bin/riscv64-unknown-elf-gcc` (GCC 16.1.0).
- Compiled bare-metal C AES-128 (`src/rv32i/c_src/aes_rv32.c`) with:
  `/opt/homebrew/bin/riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O2 -nostdlib -ffreestanding -T src/rv32i/c_src/link.ld -o src/rv32i/c_src/aes_rv32.elf src/rv32i/c_src/aes_rv32.c`
- Loaded the 32-bit ELF into `src/rv32i/sim.py` using `pyelftools` (`ELFFile`).

### What Was Built
- `src/rv32i/sim.py`: Full RV32I base instruction-set simulator (32 registers with hardwired `x0 = 0`, 32-bit PC, sparse 64 KB page-based byte-addressable little-endian memory, ELF loader via `pyelftools`, instruction counters broken down into `alu`, `load`, `store`, `branch`, `jump`, `custom`, and `system`).
- Custom AES instruction support in `src/rv32i/sim.py` based on `../AES_processor`:
  1. **RV32 Zkne (`aes32esmi` / `aes32esi`)**: The 32-bit ratified RISC-V Scalar Crypto counterpart to `../AES_processor/rtl/zkne_unit.v`.
  2. **Custom-0 (`0001011`) `aes_round_col` / `aes_final_col`**: A 32-bit column-round custom instruction modeled directly on `../AES_processor/rtl/aes_round64.v` (`SubBytes + MixColumns + AddRoundKey` on one 32-bit column in 1 cycle).
- `tests/rv32i/test_sim.py` & `tests/rv32i/test_aes_rv32i.py`: Hand-written RV32I instruction tests (covering `x0`, `LUI`/`AUIPC`, sign extension, logical/arithmetic shifts, signed/unsigned ALU & branches, byte/halfword/word loads & stores, and `JAL`/`JALR`) plus end-to-end verification that all 4 RV32I AES-128 variants match `src/aes/aes.py` on FIPS-197 Appendix B, Appendix C.1, and 15 random blocks with constant-time (`Zkt`) instruction counts.
- `src/rv32i/benchmark.py` & `results/rv32i/`: Generated `aes_rv32i_counts.csv`, `aes_rv32i_table.md`, `aes_rv32i_results.json`, and `aes_rv32i_comparison.png`.

### Key Numbers (Per 16-Byte Block, Pre-Expanded Round Keys)
- **One-Time Key Expansion (`aes128_expand_key`)**: **1,471 instructions** (ALU: 1,090, Load: 85, Store: 179, Branch: 86, Jump: 31).
- **Per-Block Encryption Breakdown & 5-Stage Pipeline Cycle Estimates**:

| Implementation | ALU | Load | Store | Branch | Jump | Custom | Total Instr / Block | Cycles (CPI=1.0) | Cycles (CPI=1.3) | Cycles (CPI=1.5) | Speedup vs. Table-Free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Software AES (Table-Free)** | 1,616 | 665 | 317 | 193 | 1 | 0 | **2,792** | 2,792.0 | 3,629.6 | 4,188.0 | **1.00x** |
| **Software AES (T-Table)** | 677 | 217 | 17 | 9 | 1 | 0 | **921** | 921.0 | 1,197.3 | 1,381.5 | **3.03x** |
| **RV32I + Zkne (`aes32esmi`/`esi`)** | 42 | 48 | 4 | 9 | 1 | 160 | **264** | 264.0 | 343.2 | 396.0 | **10.58x** |
| **RV32I + Custom-0 (`aes_round_col`)** | 15 | 48 | 4 | 9 | 1 | 120 | **197** | 197.0 | 256.1 | 295.5 | **14.17x** |

*(For reference, the 64-bit `../AES_processor` RV64+Zkne core encrypts a 16-byte CTR block in **54 cycles/block** when round keys are kept in registers `x6..x27`.)*

### Known Problems / Notes
- Initially, writing `xtime_u8` with a C ternary `(x & 0x80) ? 0x1b : 0x00` caused GCC `-O2` to emit a conditional branch on the MSB of `x`, causing instruction counts to vary by a few instructions across plaintexts (violating constant-time `Zkt` execution). Replacing it with a branchless bitmask `((0U - (x >> 7)) & 0x1bU)` made `aes128_tablefree_block` strictly constant-time (2,792 instructions on every block).
- T-table AES (`aes128_ttable_block`) is $3.03\times$ faster than table-free software AES, but requires 4 KB of lookup tables (`Te0..Te3`) and performs data-dependent memory indexing (`Te0[s0 & 0xff]`), which is vulnerable to cache-timing attacks on cores with data caches.

### Three Questions I Should Be Able to Answer (Part B)
1. **Why is the T-table software implementation ($3.03\times$ fewer instructions) much faster than table-free AES on RV32I, and what is its security drawback?**
   T-tables precompute `SubBytes + ShiftRows + MixColumns` into four 1 KB lookup tables (`Te0..Te3`), replacing dozens of byte-level GF($2^8$) shifts and XORs per round with 16 32-bit loads and XORs. Its drawback is 4 KB of table memory and secret-dependent memory addresses that leak key bytes via cache-timing side channels.
2. **How do the RV32 Zkne (`aes32esmi`/`aes32esi`) and Custom-0 (`aes_round_col`) instructions achieve $10.58\times$ and $14.17\times$ speedups over table-free software AES?**
   They move the S-box lookup and GF($2^8$) `MixColumns` matrix multiplication into combinational hardware inside the ALU/crypto execution stage, eliminating all 160 S-box memory loads per block and collapsing hundreds of bit-manipulation instructions into 160 (`Zkne`) or 120 (`Custom-0`) single-cycle register-to-register instructions.
3. **Why do we evaluate CPI = 1.0, 1.3, and 1.5 for a 5-stage RV32I pipeline instead of assuming 1 instruction = 1 cycle?**
   In a classic 5-stage pipeline (`IF-ID-EX-MEM-WB`), `LW`/`LBU` instructions followed immediately by a dependent ALU instruction incur a 1-cycle load-use stall, and taken branches incur a 2-cycle flush penalty when resolved in `EX` (as in `../AES_processor/rtl/processor.v`), raising the effective Cycles Per Instruction (CPI) above 1.0.
