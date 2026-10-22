# Project notebook (PROGRESS.md)

How to reproduce everything so far:
```
.venv/bin/python -m src.experiments_phase1   # Phase 1 numbers + plots
.venv/bin/python -m src.experiments_phase2   # Phase 2 numbers + plots
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
