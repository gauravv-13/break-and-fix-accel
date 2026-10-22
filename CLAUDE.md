# Break-and-fix lab: a tiny ML accelerator, attacked and defended

I'm building a small simulated ML accelerator, attacking it (RowHammer-style bit flips, power side-channels), adding defenses, and measuring what each defense costs. The results go into my honours project on security for ML accelerators, so correct, reproducible results matter more than speed.

- PLAN.md has the six phases. Do them in order.
- PROGRESS.md is the project notebook (create it if it's missing). Read it at the start of every session and continue from where it says we stopped.

## Rules
- Python only, in a virtual environment at `.venv`. Keep `requirements.txt` up to date.
- Layout: code in `src/`, pytest tests in `tests/`, plots and tables in `results/phaseN/`, writing in `docs/`.
- Fix random seeds so every number can be reproduced.
- Never fake, hard-code, or quietly tune a number to make a result look better. If something doesn't work, say so plainly in PROGRESS.md.
- Use scikit-learn's built-in digits dataset. No dataset downloads.
- Install system tools (Icarus Verilog, Yosys) with the OS package manager (Homebrew on macOS). Apart from that, don't change anything outside this folder.
- Explain things to me in simple words. I'm not an experienced programmer.

## Finishing a phase
A phase is done only when all of these are true:
1. The full test suite passes.
2. PROGRESS.md has a new entry for the phase: what was built, key numbers, plots made, known problems, and three questions I should be able to answer about it.
3. The work is committed with git as `Phase N: <short summary>`.

Then give me a short summary in simple words. Only continue to the next phase if I asked for more than one.
