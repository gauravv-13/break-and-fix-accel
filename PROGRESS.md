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
