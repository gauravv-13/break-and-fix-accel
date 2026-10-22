# Break-and-fix lab: a tiny ML accelerator, attacked and defended

An honours project on security for ML accelerators. I built a small **simulated** ML
accelerator, attacked it in simulation (RowHammer-style weight bit flips, power side
channels), added defenses, and measured what each defense costs in memory, cycles,
randomness and logic.

Everything runs on my own small model in Python (plus a few Verilog modules). Nothing
targets real hardware or real systems.

- **Report:** [`docs/report.md`](docs/report.md) (about 5 pages: problem, setup, results, related work, limits, future work)
- **Dashboard:** [`results/dashboard.html`](results/dashboard.html). Download it and open it in a browser; it works offline. It has toggles for each defense and sliders for flipped bits and noise.
- **Lab notebook:** [`PROGRESS.md`](PROGRESS.md) (every phase, every number, every assumption and known problem); AES/RV32I track: [`PROGRESS_AES.md`](PROGRESS_AES.md)

---

## What's inside

| Phase | What | Key result (measured unless marked) |
|---|---|---|
| 1 | INT8 MLP (64-32-10) on scikit-learn digits + cycle-accurate 8x8 weight-stationary systolic array | 96.94% accuracy (float = INT8 = accelerator, bit-identical logits); 920 cycles per image |
| 2 | Random vs targeted weight bit flips (Bit-Flip Attack idea); parity and CRC-8 row checks | 100 random flips: 93% accuracy. Targeted search: chance level after 11-35 flips (median 28.5, 20 runs) |
| 2b | Adaptive attacker that only uses flips the check can't see; check bits in attackable memory; SECDED | No linear code stops an attacker who knows it (parity: 18 bits, CRC-8 / SECDED: 36 bits, never flagged). False alarms at 1 flip: parity 1.2%, CRC-8 11.1%, SECDED 0% |
| 2c | Keyed MAC per row (AES-CMAC, 8/16-bit tags) | **Partial:** module built but untested, no MAC experiments; only analytical numbers (2^-t per changed row) |
| 3 | Shuffling and first-order Boolean masking (Goubin conversion, 32-bit shares) in the array, cost only | Bit-identical results. Masking: +8.7% cycles for 1 image, doubled weight memory, 151,552 random bits per load. Fresh randomness per image costs about 22-24x for 360 images |
| 4 | AES-128-CTR encrypted weights, using measured RV32I instruction counts | 1 image: 588x slower (software AES) ... 45.8x (custom instruction) ... 1.17x (pipelined AES engine). CTR gives no integrity: bit flips pass straight through |
| 5 | Parity, CRC-8, SECDED checkers in Verilog (Icarus + Yosys) | All pass a self-checking testbench; 64 / 200 / 351 gates, 7 / 10 / 13 gate levels vs 51 for one PE. SECDED needs 1 cycle, not the assumed 2 |
| 6 | Dashboard and report | `results/dashboard.html`, `docs/report.md` |

The AES/RV32I track (merged from a separate branch) adds AES-128 from scratch
(FIPS-197 verified), simulated CPA on AES (15 / 30 / 80 / 275 traces for 90% key
recovery at noise 0.5 / 1 / 2 / 4), and an RV32I instruction-set simulator that counts
instructions for software and custom-instruction AES.

## Honest limits

- It's all simulation: no real DRAM, no measured power traces, no silicon. The model is
  tiny (2,368 weights, 360 test images).
- Entries marked **analytical** in the report and dashboard (MAC detection probabilities,
  the effect of shuffling/masking on trace counts) are formulas or published results, not
  measurements. No side-channel attack was run on the accelerator itself.
- Several cycle costs are stated assumptions (masking converter, AES engines, CPI 1.3,
  MAC engine). Only the row checkers' delay was checked in hardware, and only as gate
  depth (no cell library).
- Phase 2c (MAC) is unfinished; see `PROGRESS.md`.

## Repository layout

```
src/            Python: model, quantization, systolic array, attacks, defenses, experiments
  aes/          AES-128 from scratch, simulated traces, CPA (AES track)
  rv32i/        RV32I simulator, bare-metal C AES variants, benchmark (AES track)
tests/          pytest suite (86 tests), incl. tests/aes and tests/rv32i
hw/rtl/         Verilog: row_parity.v, row_crc8.v, row_secded.v, pe_mac.v
hw/tb/          self-checking testbench
hw/vectors/     test vectors generated from the Python models
results/phaseN/ plots, tables (.md) and raw results (.json) per phase; results/aes, results/rv32i
docs/           report.md, aes_notes.md
PLAN.md         the six phases; PROGRESS.md the notebook
```

## Setup

Python 3.11, in a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Optional, for Phase 5 (macOS): `brew install icarus-verilog yosys`. The RV32I tests use
the committed `src/rv32i/c_src/aes_rv32.elf`, so no RISC-V compiler is needed to run them.

## Reproduce

All randomness is seeded; rerunning gives the same numbers.

```bash
.venv/bin/python -m pytest                    # full test suite (86 tests)

.venv/bin/python -m src.experiments_phase1    # train, quantize, systolic array, plots
.venv/bin/python -m src.experiments_phase2    # random vs targeted flips, parity/CRC-8
.venv/bin/python -m src.experiments_phase2b   # adaptive attacker, false alarms, SECDED (~1 min)
.venv/bin/python -m src.experiments_phase3    # shuffling / masking cost
.venv/bin/python -m src.experiments_phase4    # AES-CTR weight encryption cost
.venv/bin/python -m src.experiments_phase5    # Verilog testbenches + Yosys synthesis (~2 min)
.venv/bin/python -m src.build_dashboard       # rebuild results/dashboard.html

.venv/bin/python -m src.aes.experiments       # AES CPA experiments (AES track)
.venv/bin/python -m src.rv32i.benchmark       # RV32I AES instruction counts (AES track)
```

Phase 1 must run first (it writes `results/phase1/model_int8.npz`); Phase 2b needs
Phase 2's results, and the dashboard reads all of them.

## Main references

Rakin, He, Fan, *Bit-Flip Attack*, ICCV 2019; Kim et al., *Flipping Bits in Memory
Without Accessing Them*, ISCA 2014; Batina et al., *CSI NN*, USENIX Security 2019;
Brier, Clavier, Olivier, *Correlation Power Analysis*, CHES 2004; Goubin, *Switching
between Boolean and Arithmetic Masking*, CHES 2001; Dubey, Cammarota, Aysu, *MaskedNet*
(HOST 2020) and *BoMaNet* (ICCAD 2020). Full list, with which ones were checked, in
[`docs/report.md`](docs/report.md).
