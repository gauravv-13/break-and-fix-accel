## Adaptive attacker (20 runs each, attacker batch seeds 0-19)

| attack | reached chance | bits flipped (median, range) | mean ± std | flagged by parity | by CRC-8 | by SECDED (detect) | acc after SECDED correction (median) |
|---|---|---|---|---|---|---|---|
| no constraint (single bits) | 20/20 | 20 (9-27) | 19.4 ± 5.2 | 20/20 | 20/20 | 10/20 | 88.2% |
| parity-invisible (2-bit, same row) | 20/20 | 18 (16-26) | 18.2 ± 2.4 | 0/20 | 20/20 | 20/20 | 10.0% |
| CRC-8-invisible (4-bit, same row) | 20/20 | 36 (24-48) | 35.2 ± 5.5 | 0/20 | 0/20 | 20/20 | 10.3% |
| SECDED-invisible (4-bit, same row) | 20/20 | 36 (28-48) | 35.6 ± 5.2 | 0/20 | 20/20 | 0/20 | 10.3% |

## Cost of each defense

| | parity | CRC-8 | SECDED |
|---|---|---|---|
| extra memory bits | 320 (1.69%) | 2560 (13.51%) | 2496 (13.18%) |
| check latency per tile (assumed) | 1 | 1 | 2 |
| extra cycles, 1 image (base 920) | +40 (4.35%) | +40 (4.35%) | +80 (8.70%) |
| extra cycles, 360 images (base 15280) | +40 (0.26%) | +40 (0.26%) | +80 (0.52%) |
| clean accuracy | 96.94% | 96.94% | 96.94% |

## m random flips inside ONE row (2000 trials each): corrected / detected / mis-corrected / missed

| m | parity | CRC-8 | SECDED |
|---|---|---|---|
| 1 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 100.0 / 0.0 / 0.0 | 100.0 / 0.0 / 0.0 / 0.0 |
| 2 | 0.0 / 0.0 / 0.0 / 100.0 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 100.0 / 0.0 / 0.0 |
| 3 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 25.1 / 74.9 / 0.0 |
| 4 | 0.0 / 0.0 / 0.0 / 100.0 | 0.0 / 99.4 / 0.0 / 0.7 | 0.0 / 99.0 / 0.0 / 1.0 |
| 5 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 100.0 / 0.0 / 0.0 | 0.0 / 32.6 / 67.5 / 0.0 |
| 6 | 0.0 / 0.0 / 0.0 / 100.0 | 0.0 / 99.3 / 0.0 / 0.7 | 0.0 / 99.2 / 0.0 / 0.9 |

## Check bits in attackable memory (1000 trials each)
false-alarm rate = share of trials where a row is flagged although its weights are intact

| random flips | parity | CRC-8 | SECDED |
|---|---|---|---|
| 1 | 1.2% (1.2% of alarms) | 11.1% (11.1% of alarms) | 0.0% (0.0% of alarms) |
| 2 | 3.7% (1.9% of alarms) | 24.0% (12.6% of alarms) | 0.0% (0.0% of alarms) |
| 5 | 6.9% (1.4% of alarms) | 44.7% (11.3% of alarms) | 0.0% (0.0% of alarms) |
| 10 | 16.9% (1.9% of alarms) | 71.2% (11.9% of alarms) | 0.1% (0.9% of alarms) |
| 20 | 26.1% (1.7% of alarms) | 90.7% (11.6% of alarms) | 0.5% (0.9% of alarms) |
| 50 | 49.9% (1.6% of alarms) | 99.9% (11.1% of alarms) | 4.0% (1.2% of alarms) |
| 100 | 71.6% (1.7% of alarms) | 100.0% (10.5% of alarms) | 12.7% (1.1% of alarms) |

## Random flips in the weights only: SECDED repair (1000 trials each)

| flips | acc, no defense (mean) | acc after SECDED (mean / worst) | fully repaired | alarm raised | silent corruption (any / with no alarm at all) |
|---|---|---|---|---|---|
| 1 | 96.93% | 96.94% / 96.94% | 100.0% | 0.0% | 0.0% / 0.0% |
| 2 | 96.90% | 96.94% / 96.67% | 99.7% | 0.3% | 0.0% / 0.0% |
| 5 | 96.77% | 96.94% / 95.83% | 96.0% | 3.9% | 0.1% / 0.1% |
| 10 | 96.59% | 96.93% / 94.17% | 86.2% | 13.8% | 0.0% / 0.0% |
| 20 | 96.21% | 96.90% / 93.61% | 57.2% | 41.7% | 1.4% / 1.1% |
| 50 | 95.07% | 96.72% / 86.39% | 1.1% | 98.7% | 13.2% / 0.2% |
| 100 | 92.85% | 96.05% / 88.06% | 0.0% | 100.0% | 61.5% / 0.0% |

## The 20 non-adaptive targeted runs (Phase 2) seen by SECDED

| seed | acc after attack | acc after SECDED | rows 'corrected' (incl. mis-corrected) | rows detected (alarm) | rows mis-corrected |
|---|---|---|---|---|---|
| 0 | 10.3% | 86.4% | 25 | 4 | 0 |
| 1 | 10.8% | 87.8% | 26 | 2 | 0 |
| 2 | 10.3% | 92.2% | 27 | 1 | 0 |
| 3 | 10.8% | 81.4% | 22 | 4 | 0 |
| 4 | 10.6% | 93.1% | 20 | 1 | 0 |
| 5 | 10.6% | 39.7% | 11 | 6 | 0 |
| 6 | 10.0% | 96.9% | 11 | 0 | 0 |
| 7 | 10.3% | 47.8% | 11 | 5 | 0 |
| 8 | 10.6% | 96.9% | 28 | 0 | 0 |
| 9 | 10.3% | 93.6% | 13 | 1 | 0 |
| 10 | 10.6% | 93.6% | 24 | 1 | 0 |
| 11 | 10.6% | 80.6% | 24 | 5 | 0 |
| 12 | 10.3% | 30.0% | 10 | 7 | 0 |
| 13 | 10.8% | 49.4% | 19 | 6 | 1 |
| 14 | 10.8% | 32.8% | 16 | 7 | 1 |
| 15 | 10.8% | 81.9% | 20 | 3 | 0 |
| 16 | 10.6% | 25.6% | 16 | 6 | 2 |
| 17 | 10.3% | 28.1% | 14 | 7 | 1 |
| 18 | 10.6% | 96.9% | 29 | 0 | 0 |
| 19 | 10.6% | 88.1% | 22 | 2 | 0 |
