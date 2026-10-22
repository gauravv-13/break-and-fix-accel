Weights: 2368 bytes = 148 AES blocks. Main numbers at CPI 1.3 (core options); range for CPI 1.0-1.5 in brackets.

| option | cycles / block | extra cycles, 1 image (base 920) | extra, 360 images one batch (base 15,280) | extra, 360 images one at a time (base 331,200) |
|---|---|---|---|---|
| Software AES (Table-Free) on RV32I | 3,651.7 | +540,452 (588.45x total) [452.88-678.82x] | +540,452 (36.37x total) [28.21-41.81x] | +194,562,576 (588.45x total) [452.88-678.82x] |
| Software AES (T-Table) on RV32I | 1,219.4 | +180,471 (197.16x total) [151.90-227.34x] | +180,471 (12.81x total) [10.09-14.63x] | +64,969,632 (197.16x total) [151.90-227.34x] |
| RV32I + Zkne (aes32esmi/esi) | 365.3 | +54,064 (59.77x total) [46.20-68.81x] | +54,064 (4.54x total) [3.72-5.08x] | +19,463,184 (59.77x total) [46.20-68.81x] |
| RV32I + Custom-0 (aes_round_col) | 278.2 | +41,174 (45.75x total) [35.43-52.64x] | +41,174 (3.69x total) [3.07-4.11x] | +14,822,496 (45.75x total) [35.43-52.64x] |
| ../AES_processor RV64 + Zkne (54 cycles/block, its README) | 54.0 | +7,992 (9.69x total) | +7,992 (1.52x total) | +2,877,120 (9.69x total) |
| Dedicated AES engine, iterative (11 cycles/block) | 11.0 | +1,628 (2.77x total) | +1,628 (1.11x total) | +586,080 (2.77x total) |
| Dedicated AES engine, pipelined (1 block/cycle) | 1.0 | +158 (1.17x total) | +158 (1.01x total) | +56,880 (1.17x total) |

One-time key expansion on RV32I: 1,471 instructions (per key, not per inference). CTR overhead assumed 17 instructions per block on the RV32I options.
