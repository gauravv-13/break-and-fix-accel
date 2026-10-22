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
