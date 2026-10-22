# RV32I AES-128 Instruction Counts & Cycle Estimates (Per 16-Byte Block)

Compiled with `/opt/homebrew/bin/riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O2` and executed on `src/rv32i/sim.py`.
One-time key expansion (`aes128_expand_key`) takes **1471 instructions** (ALU: 1090, Load: 85, Store: 179, Branch: 86, Jump: 31).

| Implementation | ALU | Load | Store | Branch | Jump | Custom | Total Instr / Block | Cycles (CPI=1.0) | Cycles (CPI=1.3) | Cycles (CPI=1.5) | Speedup vs. Table-Free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Software AES (Table-Free) | 1760 | 665 | 317 | 49 | 1 | 0 | **2792** | 2792 | 3629.6 | 4188.0 | **1.00x** |
| Software AES (T-Table) | 677 | 217 | 17 | 9 | 1 | 0 | **921** | 921 | 1197.3 | 1381.5 | **3.03x** |
| RV32I + Zkne (aes32esmi/esi) | 42 | 48 | 4 | 9 | 1 | 160 | **264** | 264 | 343.2 | 396.0 | **10.58x** |
| RV32I + Custom-0 (aes_round_col) | 15 | 48 | 4 | 9 | 1 | 120 | **197** | 197 | 256.1 | 295.5 | **14.17x** |
