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

## Phase 2b / 2c (added later)
- 2b: adaptive attacker that only uses flips the defense cannot see; check bits in attackable memory (false alarms); SECDED as a third defense; 20 targeted runs. DONE.
- 2c: keyed MAC per row. PARTIAL: module built, untested, no MAC experiments (see PROGRESS.md).

## Phase 3: Side-channel defenses for the accelerator (defense cost only, no new attacks)
1. The side-channel evidence is the AES CPA results in `results/aes/` from the merged AES track. No new attack code.
2. Add two defenses to the accelerator model: random shuffling of the MAC (multiply-accumulate) order, and first-order Boolean masking of the multiply step.
3. For each, measure the cost: extra cycles, extra hardware (random bits per inference, extra registers), and check that clean accuracy is unchanged.
4. In PROGRESS.md, add a short analytical note (cited published work, clearly marked analytical, not measured) on how much each defense is expected to raise the number of traces an attack needs.

## Phase 4: Memory encryption cost
1. Weights stored in DRAM encrypted with AES-128 in counter mode, decrypted as they load.
2. Use the RV32I instruction counts in `results/rv32i/` (table-free, T-table, Zkne, custom instruction) instead of re-estimating; add a dedicated AES engine option with stated assumptions.
3. Note in PROGRESS.md how the AES track's custom instruction differs from my 64-bit `aes_round64.v` in `../AES_processor` (read only, never change it).
4. Report extra time per inference for 1 image and for 360 images, in a table and a plot.

## Phase 5: Hardware
1. Write the parity, CRC-8 and SECDED row checkers in Verilog.
2. Install Icarus Verilog and Yosys with Homebrew. Test each checker with a self-checking testbench, synthesize with Yosys, report cell counts.
3. Compare the real check delays with the cycle assumptions from Phases 2 and 2b; update the numbers if they differ.

## Phase 6: Dashboard and report
1. `results/dashboard.html`: one self-contained page using precomputed results, with toggles for each defense and sliders for noise and number of flips.
2. `docs/report.md`: problem, setup, results with plots, related work (real papers only), limits (including the partial Phase 2c and the analytical-only entries), and "gaps and future work" for my advisor.

## Rules for the remaining phases
- Follow the "Finishing a phase" checklist in CLAUDE.md after each phase.
- If a safety filter stops a step, do not retry it: write what was skipped in PROGRESS.md and move on.
