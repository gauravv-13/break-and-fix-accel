# Project notebook (PROGRESS.md)

How to reproduce everything so far:
```
.venv/bin/python -m src.experiments_phase1   # Phase 1 numbers + plots
.venv/bin/python -m src.experiments_phase2   # Phase 2 numbers + plots
.venv/bin/python -m src.experiments_phase2b  # Phase 2b (needs Phase 2's results.json)
.venv/bin/python -m pytest                   # all tests
```
All randomness is seeded (seed 0 unless stated otherwise).

---

## Phase 1: Build the target (2026-10-22) - DONE

### What was built
- `src/model.py`: float MLP 64 -> 32 -> 10 with ReLU (scikit-learn `MLPClassifier`, adam,
  seed 0) on the sklearn digits dataset. Pixels scaled 0..16 -> 0..1. Stratified 80/20
  split with seed 0 (1437 train / 360 test images).
- `src/quantize.py`: symmetric per-layer INT8 quantization. One scale per weight matrix
  (max|W|/127) and one per activation tensor (input, hidden), calibrated on the training
  set. INT8 x INT8 multiplies, int32 accumulators, int32 biases. After layer 1: bias,
  ReLU, then requantize back to INT8 (0..127). Layer-2 int32 outputs are the logits.
  Everything is saved in `results/phase1/model_int8.npz` (weights, biases, scales,
  float and INT8 test data, labels).
- `src/systolic.py`: cycle-by-cycle model of an 8x8 weight-stationary systolic array.
  Each PE holds one weight; activations move right, partial sums move down, rows are
  skewed by their row index. Big matrices are cut into 8x8 tiles (zero-padded at edges),
  and tiles along the input dimension are summed in 32-bit accumulators.
  Cycles per tile with M input vectors: load 8 + fill 7 + compute M + drain 7 = M + 22.
- `src/accel.py`: `run_inference(xq, weights, checker=None) -> (logits, stats)` runs the
  whole INT8 network on the array model. Bias/ReLU/requantize happen in a
  "post-processing unit" whose cycles are not counted.
- `src/experiments_phase1.py`: makes all numbers, tables and plots below.
- `tests/test_phase1.py`: 17 tests (exact match vs NumPy for 12 shapes including sizes
  not divisible by 8 and extreme values -128, tile cycle counts, busy-cycle counts,
  accelerator accuracy == NumPy INT8 accuracy, logits identical).

### Key numbers
| | test accuracy |
|---|---|
| float | 96.94% (349/360) |
| INT8 (NumPy) | 96.94% |
| INT8 on the systolic model | 96.94% (logits bit-identical to NumPy) |

- Accuracy drop from INT8: 0.00 percentage points. Note: 2 of 360 individual predictions
  did change, but one went right->wrong and the other wrong->right, so they cancel.
- Cycles and utilization (utilization = busy PE-cycles / (64 x total cycles)):

| layer | K x N | batch | tiles | total cycles | utilization |
|---|---|---|---|---|---|
| 1 | 64 x 32 | 360 | 32 | 12,224 | 94.2% |
| 2 | 32 x 10 | 360 | 8 | 3,056 | 58.9% |
| 1 | 64 x 32 | 1 | 32 | 736 | 4.3% |
| 2 | 32 x 10 | 1 | 8 | 184 | 2.7% |

  Whole test set: 15,280 cycles. One image: 920 cycles.
  Layer 2 is less busy because its 10 outputs fill only 10 of 16 columns in its two
  tile-columns (6 columns are padding). With batch 1, the array spends almost all its
  time loading weights, filling and draining (22 of 23 cycles per tile).

### Plots / tables
- `results/phase1/utilization_per_layer.png`
- `results/phase1/cycles_vs_size.png` (simulated cycles for square n x n matmuls; matches
  the formula ceil(n/8)^2 x (n + 22) exactly, checked in the script)
- `results/phase1/cycles_table.md`, `results/phase1/cycles_vs_size.csv`,
  `results/phase1/summary.json`

### Known problems / simplifications
- Requantization multiplies by a float (m1). Real hardware would use a fixed-point
  multiplier + shift; results could differ by 1 LSB for a few values.
- No overlap between tiles: the next tile's weights are not pre-loaded while the current
  tile computes (no double buffering). Real arrays do overlap, so our cycle counts are
  pessimistic, especially for batch 1.
- Post-processing (bias, ReLU, requantize) and moving data in/out of memory cost 0 cycles
  in the model.
- The test set is small (360 images), so a single image is worth 0.28 percentage points.

### Three questions I should be able to answer
1. Why is utilization 94% for the whole test set but only about 4% for one image, even
   though the array and the weights are the same?
2. What does "weight stationary" mean, and why do the inputs of row i need to be delayed
   by i cycles?
3. How do the INT8 scales work: how is a float weight turned into an int8, and why is the
   bias stored as int32 with scale s_in * s_w?

---

## Phase 2: Simulated bit flips in weight memory + checksum defense (2026-10-22) - DONE

### What was built
- `src/faults.py`: `flip_bits(weights, positions)` treats all INT8 weights as one flat
  memory (layer 1 then layer 2, row-major; bit 0 = LSB, bit 7 = sign) of 18,944 bits and
  returns a new, flipped copy.
- `src/bfa.py`: targeted progressive bit search (idea from Rakin et al., ICCV 2019). Each
  round: straight-through gradient of the loss w.r.t. the integer weights on a 128-image
  attacker batch (taken from the TRAINING set), top-10 |gradient| weights per layer,
  try all 8 bits of each (160 tries), keep the flip that raises the real INT8 loss most.
  Stops at test accuracy <= 11% or 100 flips. Run 5 times with different attacker
  batches (seeds 0-4).
- `src/defense_checksum.py`: one checksum per weight row (the up-to-8 weights loaded into
  the array in one cycle; 320 rows). Two kinds: parity (1 bit/row) and CRC-8, poly 0x07
  (8 bits/row). The accelerator (`run_inference(..., checker=...)`) checks every row as
  it loads and counts failures.
- `src/experiments_phase2.py`: all numbers and plots below.
- `tests/test_phase2.py`: 15 tests (flip twice = original; exact bit changed; every
  single-bit flip in every row caught by both checksums (exhaustive); CRC-8 catches every
  2-bit error in 8-byte and 2-byte rows (exhaustive); parity misses a 2-bit error (on
  purpose); clean accuracy unchanged with the check; accelerator flags the same rows as
  the direct check; memory overhead; BFA gradient sign agrees with real loss change;
  BFA loss rises every round).

### Key numbers
Clean INT8 accuracy: 96.94%.

Random flips (20 trials each):

| flips | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| mean acc (%) | 96.93 | 96.97 | 96.69 | 96.96 | 96.33 | 95.58 | 93.06 |
| std (pp) | 0.14 | 0.21 | 0.73 | 0.38 | 0.82 | 1.13 | 2.03 |
| worst trial (%) | 96.39 | 96.67 | 94.44 | 96.11 | 93.33 | 92.22 | 88.33 |

(Update in Phase 2b: the targeted search was rerun with 20 attacker batches; see the
Phase 2b entry for the 20-run numbers. Runs 0-4 gave exactly the same results again.)

Targeted bit search: reached chance level (10.3-10.8%) in all 5 runs, after
**33, 30, 29, 30, 22 flips** (median 30). That is 0.16% of the weight bits. The first flip
alone drops accuracy to ~93%. 100 random flips only drop it to 93%.
Of the 144 chosen flips: 99 hit bit 7 (sign), 37 bit 6, 6 bit 5, 2 bit 4, none lower;
33 were in layer 1 and 111 in layer 2.

Defense:

| | parity (1 bit/row) | CRC-8 (8 bits/row) |
|---|---|---|
| extra memory | 320 bits (1.69%) | 2,560 bits (13.51%) |
| extra cycles, 1 image (920 base) | +40 (4.35%) | +40 (4.35%) |
| extra cycles, 360 images (15,280 base) | +40 (0.26%) | +40 (0.26%) |
| clean accuracy with check | 96.94% (unchanged) | 96.94% (unchanged) |
| random flips detected (1000 trials each, N = 1..100) | 100% except N=2: 99.9% | 100% |
| 2 / 4 / 6 flips in the same row detected | 0% / 0% / 0% | 100% / 99.1% / 99.5% |
| 3 / 5 flips in the same row detected | 100% / 100% | 100% / 100% |
| targeted attack detected (from the very first flip) | 5/5 runs | 5/5 runs |
| rows changed by the targeted attack that get flagged | 120/132 (91%) | 132/132 |

The one parity miss in the random test (N=2, both flips in one row) did not change
accuracy. Parity missed 12 rows in the targeted attack because the search sometimes flips
two bits of the same weight (e.g. bit 7 then bit 5) or two weights in the same row; the
model as a whole was still flagged because other rows failed.

### Plots / tables
- `results/phase2/accuracy_vs_flips.png` (random mean ± std vs the 5 targeted runs)
- `results/phase2/detection_same_row.png` (parity vs CRC-8 for 2-6 flips in one row)
- `results/phase2/defense_table.md`, `results/phase2/results.json` (every trial)

### Known problems / simplifications
- Accuracy in the experiments is computed with the fast NumPy INT8 path, not the
  cycle model, for speed. They are proven bit-identical in tests, and the script
  re-checks it on faulty models (random trial 0 of each N, and every BFA final model).
- Checksums are assumed to be stored somewhere the attacker cannot flip. Flips in the
  checksum bits themselves are not simulated (they would cause false alarms).
- Extra cycles assume a 1-cycle pipelined XOR-tree checker for BOTH parity and CRC-8
  (an assumption, not measured; Phase 5 will build the checker in Verilog and give a real
  gate count). A byte-serial CRC would be much slower.
- The defense only DETECTS. What happens next (reload clean weights, stop) is not modelled.
- The attacker here does not know about the defense. An attacker who knows parity is
  used can flip 2 bits in one row and pass it; CRC-8 still caught 99.1-100% of such
  multi-flip rows, but it is not a cryptographic check: a determined attacker who knows
  the CRC could find undetected 4-bit patterns (we saw ~0.5-0.9% slip through).
- Random-flip effect is small partly because the hidden layer saturates at 127, which
  caps the damage of one huge weight. Our model is tiny (2,368 weights), so the "flips
  needed" numbers are not directly comparable with the paper's ImageNet results.
- The BFA spends ~10 flips on a plateau (~88%) where it raises loss on confident
  class-8 images without changing many predictions: maximising loss is not the same as
  maximising errors.

### Three questions I should be able to answer
1. Why do 30 chosen flips destroy the model while 100 random flips barely hurt it? Why
   does the search almost always pick bit 7 or bit 6?
2. Why does parity catch 3 flips in a row but miss 2 or 4? What does CRC-8 do better,
   and what does it cost in memory?
3. Why is the extra-cycle cost 4.35% for one image but only 0.26% for 360 images, even
   though both add the same 40 cycles?

---

## Phase 2b: Adaptive attacker, attackable check bits, SECDED, 20 targeted runs (2026-10-22) - DONE

### What was built
- `src/defense_checksum.py` refactored: a shared `RowCodeChecker` base class (stored code
  word per row, `decode_row`, `with_flipped_stored_bits` to attack the check bits,
  `bit_syndromes` for the adaptive attacker). Parity/CRC-8 behave exactly as before.
- `src/defense_secded.py`: SECDED (extended Hamming) per row: (72,64) for 8-weight rows,
  (22,16) for 2-weight rows. Corrects 1 flipped bit (data or check bit), detects 2.
  The accelerator now loads the CORRECTED row (systolic `check_row` hook accepts
  `(status, row)`), and counts corrections.
- `src/adaptive.py`: the adaptive attacker. All three codes are linear, so a flip
  pattern is invisible exactly when the XOR of its bits' syndromes is 0. It enumerates
  the smallest invisible patterns per row: parity = any 2 bits (2,016 per 8-byte row);
  CRC-8 = 4-bit patterns (5,049 per 8-byte row, 16 per 2-byte row); SECDED = 4-bit
  patterns (7,025 / 64). Search: rank every allowed pattern by a gradient estimate, try
  the top 200 for real, keep the one that raises the loss most, repeat.
  Control: the same search with single bits and no constraint, so we compare like with
  like.
- `src/experiments_phase2.py`: targeted Phase-2 search now runs 20 times (seeds 0-19).
- `src/experiments_phase2b.py`: all numbers/plots below.
- `tests/test_phase2b.py`: 15 new tests (SECDED corrects every single data-bit and
  check-bit flip in every row, detects every double flip (exhaustive); accelerator
  repairs 30 single flips with bit-identical output; clean accuracy unchanged; no
  invisible CRC-8/SECDED pattern below 4 bits (exhaustive over 1-, 2-, 3-bit patterns);
  300 sampled "invisible" patterns per defense really pass the real checker; the
  adaptive attack stays invisible and still hurts; a flipped stored checksum bit gives
  exactly one false alarm). Full suite: 47 passed.

### Key numbers
**Item 4: Phase-2 targeted search, 20 runs:** all 20 reached chance; flips needed
11-35, median 28.5, mean 26.8 ± 6.0. Of 536 flips: 394 bit 7, 123 bit 6, 17 bit 5,
2 bit 4; 120 in layer 1, 416 in layer 2. Parity flagged 388/456 changed rows (85%),
CRC-8 456/456; both flagged every run from its first flip.

**Item 1: adaptive attacker (20 runs each, same attacker batches):**

| attack (flips must be invisible to) | reached chance | bits flipped: median (range), mean ± std |
|---|---|---|
| nothing (control, single bits) | 20/20 | 20 (9-27), 19.4 ± 5.2 |
| parity (2 bits in one row) | 20/20 | 18 (16-26), 18.2 ± 2.4 |
| CRC-8 (4-bit invisible patterns) | 20/20 | 36 (24-48), 35.2 ± 5.5 |
| SECDED (4-bit invisible patterns) | 20/20 | 36 (28-48), 35.6 ± 5.2 |

Every adaptive run was invisible to the defense it targeted (0 rows flagged, checked
by assertion). Parity does not slow the attacker at all (18 vs 20 bits); CRC-8 and SECDED
only about double the bits needed (36 vs 20). Cross-checks: the CRC-8-invisible attack is
also invisible to parity (CRC-8's polynomial makes every invisible pattern have an even
number of bits) but SECDED detects it in 20/20 runs; the SECDED-invisible attack is caught
by CRC-8 in 20/20 runs. So no single code here stops an attacker who knows it.
(Note the control needs fewer bits than the Phase-2 search, 20 vs 28.5, because it
ranks every bit instead of only the top-10 weights per layer.)

**Item 2: check bits in attackable memory (1000 trials per point):** a "false alarm" is a
row flagged although its weights are intact.

| random flips | parity (320 check bits) | CRC-8 (2560) | SECDED (2496) |
|---|---|---|---|
| 1 | 1.2% | 11.1% | 0.0% |
| 5 | 6.9% | 44.7% | 0.0% |
| 20 | 26.1% | 90.7% | 0.5% |
| 100 | 71.6% | 100.0% | 12.7% |

About 1.7% (parity) and 11-12% (CRC-8) of all alarm rows are false: roughly each code's
share of the total memory, as expected. SECDED fixes a single flipped check bit, so it
only false-alarms when two flips hit the check bits of the same row.

**Item 3: SECDED vs parity vs CRC-8:**

| | parity | CRC-8 | SECDED |
|---|---|---|---|
| extra memory | 320 bits (1.69%) | 2,560 (13.51%) | 2,496 (13.18%) |
| extra cycles, 1 image (920 base) | +40 (4.35%) | +40 (4.35%) | +80 (8.70%) |
| extra cycles, 360 images | +40 (0.26%) | +40 (0.26%) | +80 (0.52%) |
| 1 flip in a row | detected | detected | **corrected** |
| 2 flips in a row | missed 100% | detected | detected |
| 3 flips in a row | detected | detected | 25% detected, **75% mis-corrected** |
| 4 / 6 flips in a row | missed 100% | 99.4 / 99.3% detected | 99.0 / 99.2% detected |
| clean accuracy | 96.94% | 96.94% | 96.94% |

Random flips in the weights only (1000 trials): SECDED fully repairs 100% of trials at 1
flip, 86% at 10, 57% at 20, 1% at 50. Mean accuracy after SECDED stays 96.05-96.94% (no
defense: 92.85% at 100 flips). Some row gets silently written with a wrong value in 1.4% of
trials at 20 flips, 13% at 50 and 62% at 100 (rows with 3+ flips get mis-corrected), but in
almost all of those another row raises an alarm; silent corruption with no alarm at all
is at most 1.1% (at 20 flips).

Phase-2 targeted runs (not adaptive) seen by SECDED: 3/20 runs fully repaired to 96.94%
with no alarm; the other 17 raise an alarm. Accuracy after correction: median 84.2%
(25.6-96.9%), because rows with 2+ flips can only be detected, not fixed; 5 rows in total
(4 runs) were mis-corrected.

### Plots / tables
- `results/phase2/accuracy_vs_flips.png` (updated: 20 targeted runs)
- `results/phase2/adaptive_attacks.png` (accuracy vs bits for the 4 attacks + bits needed)
- `results/phase2/false_alarms.png`
- `results/phase2/same_row_outcomes.png` (corrected / detected / mis-corrected / missed)
- `results/phase2/phase2b_tables.md`, `results/phase2/phase2b_results.json`
- (`detection_same_row.png` and `defense_table.md` from Phase 2 are regenerated too)

### Known problems / simplifications
- The adaptive attacker assumes it can flip any chosen set of bits exactly, including
  2-4 bits in one 64-bit row at once. Real RowHammer gives far less control over which
  bits flip; this is a worst-case (strong attacker) model.
- The adaptive search only uses the smallest invisible patterns (2 bits for parity, 4 for
  CRC-8/SECDED), and only from the top 200 gradient-ranked patterns per round. A better
  search could need even fewer bits, so these counts are upper limits.
- SECDED's 2-cycle check latency is an assumption (1 for syndrome, 1 for correction),
  not measured. Phase 5 will build a checker in Verilog.
- On an alarm the model does nothing (no reload, no stop); "accuracy after SECDED"
  leaves detected-but-uncorrectable rows as they are.
- False alarms only consider random flips; an attacker could flip check bits on purpose
  to cause alarms (a denial-of-service), not studied.
- Not tried yet: combining two codes (e.g. CRC-8 + SECDED), or a keyed/cryptographic
  check (MAC) that an attacker cannot compute. Our results suggest that's the real fix
  against an adaptive attacker.
- Plot palette: the colorblind validator script needs Node.js, which isn't installed; I
  used the palette's first four colors in its documented, pre-validated order instead.

### Three questions I should be able to answer
1. Why can an attacker who knows the code always find flips it can't see? Why does
   that take 2 bits for parity but 4 bits for CRC-8 and SECDED?
2. Why does SECDED have almost no false alarms when its check bits get hit, while
   CRC-8 (about the same number of check bits) has the most?
3. SECDED corrects 1 flip and detects 2. What goes wrong with 3 flips in one row, and why
   is "mis-corrected" worse than "detected"?

---

## Phase 2c (partial): keyed MAC defense module, no experiments (2026-10-22) - NOT FINISHED

### Status, plainly
- `src/defense_mac.py` is **built but untested**: there are no unit tests for it, and
  **no MAC experiments were run** (no random-fault, same-row, false-alarm or attacker
  measurements). It was only smoke-run once to check that it imports and runs.
- The planned five-defense comparison table and plots were **not made**.
- Work stopped because my responses were halted three times by a safety filter while
  writing the MAC experiment code, so I did not continue that part.

### What was built
- `src/defense_mac.py`: a truncated MAC per weight row. tag = first t bits (t = 8 or 16)
  of AES-CMAC(key, address || row bytes), where address = (layer, k, nt). HMAC-SHA256 is
  an option. Uses the `cryptography` library (added to `requirements.txt`). The key comes
  from a fixed seed (2026) for reproducibility; a real chip would use a secret random key.
- `src/defense_checksum.py`: the base class now has `encode_at(layer, k, nt, row)`, so a
  code can depend on the row's address (needed by the MAC). Parity, CRC-8 and SECDED
  behave exactly as before; the full suite (47 tests) still passes.
- Cost assumptions written in the module (not measured): one AES-128 block per row,
  1 round per cycle, + 1 cycle to compare = 11 cycles per row. A pipelined engine adds
  11 cycles per tile (40 tiles per inference = +440 cycles); a one-row-at-a-time engine
  would add 8 x 11 - 8 = 80 cycles per tile. HMAC-SHA256 (2 SHA-256 compressions,
  64 rounds each) would be 130 cycles per row. Memory: 8 bits/row = 2,560 bits (13.5% of
  the 18,944 weight bits), 16 bits/row = 5,120 bits (27.0%).

### Worked math (ANALYTICAL, not measured)
- With a good t-bit MAC and a key the attacker does not know, any change to a row gives a
  tag that looks random, so the changed row passes the check with probability 2^-t:
  1/256 for t = 8, 1/65,536 for t = 16.
- If an attack changes r different rows, all of them pass at once with probability
  2^-(t*r). Example: r = 10 rows with t = 8 gives 2^-80.
- If an attacker could retry after each alarm, the number of tries until one row passes
  is geometric with success chance p = 2^-t, so on average 1/p = 2^t tries per row:
  about 256 tries (t = 8) or 65,536 tries (t = 16), each failed try raising an alarm.
- This is why an alarm must trigger a response. If the system ignores alarms, a short
  tag only slows a patient attacker down; if it reloads the weights or stops after a few
  alarms, the attacker gets only a handful of tries instead of 2^t.

### Future work
- Alarm response: model what happens on an alarm (reload clean weights, or stop the
  accelerator after a few alarms) and measure its cost.
- Unit tests for `defense_mac.py`, and the MAC measurements under the same random-fault
  seeds as Phase 2b, plus the five-defense comparison table.

### RISC-V note: speeding up the MAC check
Each row check is one AES-128 block (CMAC). If a small RISC-V control core did this in
software on plain RV32I, every AES round needs many table lookups, XORs and shifts, so the
check would cost far more than the 11 cycles assumed above. Two ways to speed it up:
- A custom AES-round instruction (like my `aes_round64.v` idea in PLAN.md Phase 4), or
  the standard RISC-V scalar crypto extension for AES (Zkne), cuts each round to a few
  instructions.
- A small dedicated AES engine beside the accelerator (the 11-cycles-per-row assumption
  above) takes the check off the core completely.
The real instruction counts for software AES vs the AES-round instruction will come from
`results/rv32i/` once the AES/RV32I track is merged; they are not available on this
branch yet, so no numbers are claimed here.

### Three questions I should be able to answer
1. Why can't an attacker who doesn't know the key find flips the MAC can't see, when it
   could for parity, CRC-8 and SECDED?
2. With an 8-bit tag, why is the chance of one changed row passing 1/256, and why does
   changing 10 rows make it 2^-80?
3. Why does a MAC only help if the system responds to alarms?

---

## Merge note (2026-10-22)
The AES/RV32I track was merged (see PROGRESS_AES.md). `requirements-aes.txt` was folded
into `requirements.txt` (only new package: pyelftools) and deleted; full suite 63 passed.
`src/rv32i/c_src/aes_rv32.elf` stays tracked because the RV32I tests load it.
Small mismatch found in the AES track: PROGRESS_AES.md lists the table-free AES row as
ALU 1,616 / Branch 193, but `results/rv32i/aes_rv32i_table.md` says ALU 1,760 /
Branch 49. Both total 2,792 instructions per block; I use the total (from the results
file) and did not change the AES track's files.

---

## Phase 3: Side-channel defenses for the accelerator, cost only (2026-10-22) - DONE

No new attack code. The side-channel evidence is the AES track's simulated CPA
(`results/aes/`): full 16-byte key recovery (90% success) needed 15 / 30 / 80 / 275
traces at noise sigma = 0.5 / 1 / 2 / 4, i.e. traces grow roughly like sigma^2.

### What was built
- `src/systolic.py`: `matmul` now also accepts int32 weights (32-bit masked shares);
  products and partial sums wrap modulo 2^32 like 32-bit registers. int8 behaviour is
  unchanged.
- `src/defense_sca.py`:
  - **Shuffling**: a fresh random permutation of the input index k and output index n
    per inference (or per batch). Weights load as W[pk][:, pn], inputs as X[:, pk],
    outputs are put back in order. Logits are bit-identical.
  - **First-order Boolean masking of the multiply step**: weights stored as two 8-bit
    Boolean shares (w XOR m, m); at each load they are re-randomised (8 bits), extended
    to 32 bits with the upper 24 bits masked (24 bits), and converted to arithmetic
    shares with Goubin's Boolean-to-arithmetic conversion (32 bits). Each PE multiplies
    the public activation by both shares and keeps two partial-sum chains; the two
    column outputs are added only in the post-processing unit. 32-bit shares are needed
    because 8-bit arithmetic shares would wrap mod 256 (wrong) or be non-uniform (leak).
- `src/experiments_phase3.py`: all options simulated on the cycle model and checked to
  give logits bit-identical to the undefended accelerator.
- `tests/test_phase3.py`: 9 tests (int32 matmul wraps like 32-bit hardware; Goubin
  conversion correct incl. edge values; shares recombine to the weights and are fresh on
  every load; shuffled and masked inference bit-identical with exact cycle counts;
  random-bit accounting). Full suite: 72 passed.

### Key numbers (baseline: 920 cycles for 1 image, 15,280 for 360 images)

| option | cycles, 1 image | cycles, 360 images | random bits, 1 image | random bits, 360 images |
|---|---|---|---|---|
| shuffle, one order per batch | 920 (+0%) | 15,280 (+0%) | 604 | 604 |
| shuffle, new order per image (reload per image) | 920 (+0%) | 331,200 (21.7x) | 604 | 217,440 |
| mask, 2 multipliers/PE, masks per batch | 1,000 (+8.7%) | 15,360 (+0.5%) | 151,552 | 151,552 |
| mask, 2 multipliers/PE, new masks per image (reload) | 1,000 (+8.7%) | 360,000 (23.6x) | 151,552 | 54,558,720 |
| mask, 1 widened multiplier (2 passes), masks per batch | 1,920 (2.09x) | 30,640 (2.0x) | 151,552 | 151,552 |
| mask, 1 widened multiplier, new masks per image | 1,920 (2.09x) | 691,200 (45.2x) | 151,552 | 54,558,720 |
| mask, 2 multipliers/PE, in-PE refresh per image [formula] | 1,000 (+8.7%) | 15,360 (+0.5%) | 151,552 | 27,355,136 |

Clean accuracy is 96.94% in every case (logits bit-identical, checked by assertion).

Extra hardware:
- Shuffling: a permutation table of 544 bits (64 x 6 + 32 x 5, largest layer), a
  Fisher-Yates generator, and address logic. 604 random bits per inference with one exact
  draw per step (minimum possible: log2(n!) = 553 bits).
- Masking (2 multipliers per PE): +88 register bits per PE (two 32-bit weight shares
  instead of one 8-bit weight, plus a second 32-bit partial sum) = +5,632 bits in the
  8x8 array; 2 multipliers of 8x32 per PE instead of one 8x8; weight memory doubled
  (+18,944 bits of shares); a Goubin converter at the load port. 64 random bits per
  weight per load (8 refresh + 24 upper-bit mask + 32 Goubin), i.e. 512 random bits per
  cycle while a tile row of 8 weights loads.

Cycle ASSUMPTIONS (not measured): shuffling permutation generated in the background
(0 cycles; +138 cycles per inference if not overlapped); Goubin converter pipelined at
the load port with 2 cycles latency per tile (two 32-bit subtractions in series).

### The big catch: randomness per batch vs per image
In a weight-stationary array the weights load once and then serve every image in the
batch. If the order or the masks are only renewed per weight load, all 360 images in a
batch see the SAME order / masks, so traces from one batch are not protected against
each other (an attacker could treat each share like an ordinary fixed weight). Fresh
randomness per image costs either a weight reload per image (about 22-24x slower for 360
images, 45x for the serial masked datapath) or, for masking, refresh adders in every PE
and up to 64 x 32 = 2,048 random bits per cycle (27.4 million bits for 360 images).

### Analytical note: expected effect on the number of traces (ANALYTICAL, NOT MEASURED)
No side-channel attack was run on the accelerator. This is what published work predicts.
- **Shuffling.** If the targeted operation is spread uniformly over m possible time
  positions, its correlation at any one fixed sample drops by a factor m, so a
  single-point CPA needs about m^2 times more traces. An attacker who adds up (integrates)
  the m candidate positions only loses a factor sqrt(m) in correlation, so about m times
  more traces. The literature usually evaluates shuffling this way ("integrated DPA",
  formulas from Mangard, Oswald, Popp 2007), and Veyrat-Charvillon et al. (ASIACRYPT 2012)
  warn that simplified shuffles (e.g. a random start index) are much weaker than full
  random permutations like ours. In our array a layer-1 weight can land in any of 32
  tiles (and 64 x 32 positions in total), so m is roughly between 32 and 2,048. Purely as an
  illustration, if the accelerator leaked like the AES model (30 traces at sigma = 1), that
  would mean very roughly 30 x 32 = 960 (integrating attacker, small m) up to
  30 x 32^2 = 30,720 traces (single-point attacker). Big caveats: the 64 PEs work in
  parallel, so a weight's leakage is already mixed with 63 others; and with one order per
  batch, shuffling gives no protection inside a batch.
- **Masking.** With a correct first-order masking, each share alone is independent of the
  weight, so first-order CPA (like the AES-track attack) should fail; the attacker needs a
  second-order attack that combines two leakages. Chari et al. (CRYPTO 1999) showed that
  the number of traces grows exponentially with the number of shares as noise grows;
  for 2 shares this is commonly summarised as roughly sigma^4 instead of sigma^2. So the
  gap to the unmasked case grows like sigma^2 as noise grows (no absolute number is
  claimed). Caveats: hardware glitches can make masked circuits leak at first order
  (Mangard, Pramstaller, Oswald, CHES 2005); our post-processing (bias, ReLU, requantize)
  is NOT masked, so hidden activations leak; masks reused across a batch break the
  argument (see above).
- For comparison, published masked NN accelerators report large costs: MaskedNet
  (Dubey, Cammarota, Aysu, HOST 2020) reports 2.8x latency and 2.3x area; BoMaNet
  (same authors, ICCAD 2020, fully Boolean-masked) reports 3.5% latency and 5.9x area on
  a Spartan-6 FPGA. These are their numbers, on different designs, not comparable 1:1.

References (existence and venue checked online; content summaries as noted):
- S. Chari, C. Jutla, J. R. Rao, P. Rohatgi, "Towards Sound Approaches to Counteract
  Power-Analysis Attacks", CRYPTO 1999, LNCS 1666 (exact scaling statement taken from
  how it is commonly summarised; full text not re-read).
- L. Goubin, "A Sound Method for Switching between Boolean and Arithmetic Masking",
  CHES 2001, LNCS 2162, pp. 3-15.
- S. Mangard, E. Oswald, T. Popp, "Power Analysis Attacks: Revealing the Secrets of
  Smart Cards", Springer 2007 (shuffling/integration formulas; book section not re-read).
- S. Mangard, N. Pramstaller, E. Oswald, "Successfully Attacking Masked AES Hardware
  Implementations", CHES 2005, LNCS 3659, pp. 157-171.
- N. Veyrat-Charvillon, M. Medwed, S. Kerckhof, F.-X. Standaert, "Shuffling against
  Side-Channel Attacks: A Comprehensive Study with Cautionary Note", ASIACRYPT 2012.
- A. Dubey, R. Cammarota, A. Aysu, "MaskedNet: The First Hardware Inference Engine Aiming
  Power Side-Channel Protection", HOST 2020; and "BoMaNet: Boolean Masking of an Entire
  Neural Network", ICCAD 2020 (overhead numbers from the abstracts).
- L. Batina, S. Bhasin, D. Jap, S. Picek, "CSI NN: Reverse Engineering of Neural Network
  Architectures Through Electromagnetic Side Channel", USENIX Security 2019.

### Plots / tables
- `results/phase3/sca_defense_cycles.png`, `results/phase3/sca_defense_random_bits.png`
- `results/phase3/phase3_table.md`, `results/phase3/phase3_results.json`

### Known problems / simplifications
- Costs only: no leakage was simulated or measured for the defended accelerator.
- Cycle numbers for the converter and permutation generator are assumptions.
- The "in-PE refresh" row is a formula, not simulated.
- Only the multiply step is masked; the shares' sum, bias, ReLU and requantization are
  computed unmasked in the post-processing unit.
- Random bits are counted, but the cost of a random number generator that can deliver
  hundreds or thousands of bits per cycle is not modelled.

### Three questions I should be able to answer
1. Why does shuffling with one order per batch protect almost nothing inside a batch, and
   why does a fresh order per image make 360 images about 22x slower in our array?
2. Why do the masked shares need to be 32 bits wide, when the weights are only 8 bits?
3. Why does masking force a second-order attack, and why does that help more when the
   measurements are noisier?

---

## Phase 4: Memory encryption cost (2026-10-22) - DONE

### What was built
- `src/memory_encryption.py`: the flat INT8 weight memory (2,368 bytes = 148 AES blocks)
  is stored in "DRAM" encrypted with AES-128-CTR (keystream block i =
  AES_K(nonce || i)), using the AES track's own `src/aes/aes.py`. The loader decrypts with
  the same XOR. Cost model per option (below).
- `src/experiments_phase4.py`: functional check (decrypt -> accelerator logits
  bit-identical) and the cost table/plot.
- `tests/test_phase4.py`: 6 tests (keystream equals `cryptography`'s AES-CTR; encrypt ->
  decrypt -> identical inference, and >95% of DRAM bytes change; a flipped ciphertext bit
  flips exactly the same plaintext bit; 148 blocks; instruction counts are read from
  `results/rv32i/`; cycle formulas). Full suite: 78 passed.

### Assumptions (stated, not measured)
- RV32I options: instructions per block taken from `results/rv32i/aes_rv32i_results.json`
  (MEASURED by the AES track on its RV32I simulator, pre-expanded round keys):
  table-free 2,792, T-table 921, Zkne 264, Custom-0 197. Plus 17 instructions per block for
  the CTR part (load/XOR/store 16 bytes, counter, loop), an ESTIMATE. Cycles =
  instructions x CPI, CPI 1.3 main value (1.0-1.5 range, as in the AES track).
- `../AES_processor` RV64 + Zkne: 54 cycles per CTR block, as reported in its README.
- Dedicated engine: iterative = 1 round per cycle, 11 cycles per block; pipelined =
  10-stage unrolled, 1 block per cycle after 11 cycles of latency.
- All weights are decrypted before the inference starts (no overlap), same clock as the
  accelerator. Key expansion (1,471 RV32I instructions) happens once per key, not per
  inference. Only the weights are encrypted (not biases/scales).
- "360 images one at a time" = batch-1 mode, weights re-read and re-decrypted for every
  image (what Phase 3's per-image randomness would require).

### Key numbers (extra cycles; total time as a multiple of no encryption; CPI 1.3)

| option | cycles/block | 1 image (base 920) | 360 images, one batch (base 15,280) | 360 images, one at a time (base 331,200) |
|---|---|---|---|---|
| table-free software AES, RV32I | 3,652 | +540,452 (588x) | +540,452 (36.4x) | +194.6 M (588x) |
| T-table software AES, RV32I | 1,219 | +180,471 (197x) | +180,471 (12.8x) | +65.0 M (197x) |
| RV32I + Zkne | 365 | +54,064 (59.8x) | +54,064 (4.5x) | +19.5 M (59.8x) |
| RV32I + Custom-0 (aes_round_col) | 278 | +41,174 (45.8x) | +41,174 (3.7x) | +14.8 M (45.8x) |
| ../AES_processor RV64 + Zkne | 54 | +7,992 (9.7x) | +7,992 (1.5x) | +2.9 M (9.7x) |
| dedicated engine, iterative | 11 | +1,628 (2.8x) | +1,628 (1.1x) | +586,080 (2.8x) |
| dedicated engine, pipelined | 1 | +158 (1.17x) | +158 (1.01x) | +56,880 (1.17x) |

(Full table with CPI 1.0-1.5 ranges: `results/phase4/phase4_table.md`.)

Main takeaways: on a small RV32I core even the best instruction-set option makes one
inference ~46x slower, because decrypting 2.4 KB of weights takes far longer than our tiny
920-cycle inference. Batching hides it (weights decrypted once per batch). Only a
dedicated engine keeps the overhead small for single images. Also: CTR encryption gives
no integrity. A flipped ciphertext bit flips exactly the same weight bit (tested), so the
Phase 2 bit-flip attack works just as well on encrypted weights. Encryption has to be
combined with a row check (Phase 2/2b) or a MAC (Phase 2c, partial).

### How the AES track's custom instruction differs from my aes_round64.v
(read `../AES_processor/rtl/aes_round64.v`, not changed)
- `aes_round64.v` is a 64-bit, single-cycle custom-0 instruction for an RV64 pipeline. Its
  own header marks it LEGACY and NON-STANDARD: it treats a 64-bit register as the whole
  AES state (real AES has a 128-bit state), so its "ShiftRows" only swaps bytes 1<->5 and
  3<->7 inside 8 bytes, then applies MixColumns to two 32-bit columns and XORs a 64-bit
  round key. It does 8 S-boxes + 2 MixColumns per instruction but does not compute real
  AES-128 on its own.
- The AES track's `aes_round_col` (RV32, custom-0, funct3 = 0) works on ONE 32-bit column:
  rd = MixColumns(SubBytes(rs1)) XOR rs2 (4 S-boxes + 1 MixColumns). `aes_final_col`
  (funct3 = 1) skips MixColumns for the last round, and two byte-packing helpers
  (`aes_shift_even`, funct3 = 2: low half from rs1, high half from rs2; `aes_shift_diag`,
  funct3 = 3: alternating bytes from rs1 and rs2) let ShiftRows be done correctly across
  the full 128-bit state held in four registers. Result: real AES-128,
  checked against FIPS-197 and random blocks (197 instructions per block).
- So: same idea (S-box + MixColumns + AddRoundKey in one instruction), but half the width
  per instruction, and correct AES because ShiftRows is handled across all four columns.
  `../AES_processor` itself now uses the standard Zkne instructions (`zkne_unit.v`) for
  real AES; its README reports 54 cycles per CTR block on the RV64 core.

### Plots / tables
- `results/phase4/memory_encryption_cost.png`, `results/phase4/phase4_table.md`,
  `results/phase4/phase4_results.json`

### Known problems / simplifications
- The 17-instruction CTR overhead and the engine cycle counts are assumptions; the
  RV32I counts are measured by the AES track, but CPI is assumed.
- No overlap: a CTR keystream does not depend on the data, so a real design could compute
  it ahead of time (while the previous layer or image runs) and hide much of the cost.
  Not modelled.
- Clock speeds of core, engine and accelerator are assumed equal.
- Nonce/counter management (a new nonce whenever the weights are re-encrypted) is not
  modelled; reusing a CTR nonce with different weights would leak their XOR.

### Three questions I should be able to answer
1. Why is decryption almost free for 360 images in one batch but very expensive for one
   image, with the same AES code?
2. Why doesn't AES-CTR encryption stop the Phase 2 bit-flip attack?
3. What is the difference between my aes_round64 instruction and the AES track's
   aes_round_col, and why does only one of them compute real AES-128?
