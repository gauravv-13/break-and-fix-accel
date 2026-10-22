Yosys, mapped to 2-input gates (AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT). Depth = longest path in gate levels.
Reference (clock): one PE multiply-accumulate, Kogge-Stone carry = 920 cells, depth 51 (64 PEs = 58,880 cells). Default Yosys flow: 849 cells, depth 60.

| checker | row | cells | XOR/XNOR cells | depth (gate levels) | depth / PE depth | cycles needed | assumed in Phase 2/2b | cells vs 64 PEs |
|---|---|---|---|---|---|---|---|---|
| parity | 8-byte row | 64 | 64 | 7 | 0.14 | 1 | 1 | 0.11% |
| parity | 2-byte row | 16 | 16 | 5 | 0.10 | 1 | 1 | 0.03% |
| crc8 | 8-byte row | 200 | 193 | 10 | 0.20 | 1 | 1 | 0.34% |
| crc8 | 2-byte row | 55 | 44 | 8 | 0.16 | 1 | 1 | 0.09% |
| secded | 8-byte row | 351 | 243 | 13 | 0.25 | 1 | 2 | 0.60% |
| secded | 2-byte row | 110 | 65 | 10 | 0.20 | 1 | 2 | 0.19% |

Testbench: 8B: PASS: 2380 vectors x 3 checkers, 8-byte rows; 2B: PASS: 540 vectors x 3 checkers, 2-byte rows
