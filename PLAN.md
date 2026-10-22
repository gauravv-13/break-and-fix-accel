# Plan: six phases

## Phase 1: Build the target
1. Train a small neural network (for example 64 -> 32 -> 10) on scikit-learn's built-in digits dataset.
2. Convert it to 8-bit integers (INT8) and report how much accuracy drops.
3. Write a cycle-by-cycle model of an 8x8 weight-stationary systolic array that runs every matrix multiply in the network. Check each output against NumPy exactly, and report cycles and how busy the array is per layer.
4. Write tests for all of the above.

## Phase 2: Flip bits like RowHammer
1. Treat the INT8 weights as if they sit in DRAM and simulate bit flips.
2. Compare random flips with a targeted search that flips the most damaging bit each round (the idea behind the Bit-Flip Attack paper, Rakin et al., ICCV 2019).
3. Plot accuracy against the number of flipped bits for both.
4. Add a defense to the accelerator model, such as a checksum per weight row that is checked when weights load. Measure what it catches and what it costs in extra memory and cycles.

## Phase 3: Steal secrets from power use
1. Warm-up: write AES-128 and check it against the FIPS-197 test vectors. Simulate power traces for the first round (Hamming weight of the S-box output plus Gaussian noise) and recover the 16-byte key with correlation power analysis (CPA).
2. Attack the accelerator the same way: simulate power traces of the multiply units on known inputs and recover the secret first-layer weights (the idea behind the CSI NN paper, Batina et al., USENIX Security 2019).
3. Plot how many traces each attack needs at different noise levels.
4. Add a defense (shuffling the multiply order, or masking). Show how much harder the attack gets and what it costs.

## Phase 4: Encrypt the weights with AES
1. Store the weights in DRAM encrypted with AES-128 in counter mode, decrypted as they load into the accelerator.
2. Compare three setups: software AES on a simple RV32I core, AES with a custom AES-round instruction, and a dedicated AES engine beside the accelerator. If my own AES-round instruction code (aes_round64.v) is in this folder, base the custom-instruction setup on it.
3. Estimate the extra time each setup adds per inference. List every assumption. Show a table and a plot.
4. Bonus: write a small RV32I instruction simulator in Python so the software-AES numbers come from real instruction counts instead of estimates.

## Phase 5: Turn a defense into hardware
1. Write the Phase 2 checksum checker in Verilog.
2. Test it with a self-checking testbench in Icarus Verilog until it passes.
3. Synthesize it with Yosys and report a rough gate count.

## Phase 6: Dashboard and report
1. Build `results/dashboard.html`: one self-contained page that opens in a browser with no server. I should be able to switch each attack and defense on or off, move sliders for noise and number of bit flips, and see the results change. Use precomputed results from the earlier phases.
2. Search the web and write `docs/report.md` (about 5 pages): problem, setup, results with plots, related work (real, well-known papers only; mark anything you couldn't verify), and a "gaps and future work" section for my advisor.
