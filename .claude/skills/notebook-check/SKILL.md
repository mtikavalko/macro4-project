---
name: notebook-check
description: Re-execute Macro4Project_v2.ipynb and report only the outputs that changed versus the committed version. Use after editing anything in aiyagari/ or the notebook, before committing, or when asked whether the results still hold.
---

# Notebook check

From the repository root:

```
python3 tools/nbcheck.py Macro4Project_v2.ipynb
```

It runs a copy of the notebook in memory (the file on disk is untouched) in about 10 seconds. Then it compares every number in the text outputs against the committed outputs.

Run-to-run noise is masked: durations like `4.09s` and the pixel size in `<Figure size 1000x400 ...>`, which depends on the machine's DPI. Plot images aren't compared.

Exit code 0 means no changes. 1 means something changed or a cell raised an error.

## Reporting back

- **No changes:** say so in one line, with the cell count.
- **Changes:** list each changed cell with its old and new lines. If you need to explain why a number moved, look at the cell with `python3 tools/nbtext.py Macro4Project_v2.ipynb <cell>`. Don't open the .ipynb directly.
- **Errors:** the first `[n] ERROR` is the root cause. Later `NameError`s in other cells are usually fallout from it.

Say whether the changes match what the code edit should have done. A calibration change moves almost every table: in a test, changing `p_ue` from 0.70 to 0.65 moved the parameter, equilibrium, welfare and diagnostics cells. A pure refactor should move nothing.

## Flags

- `--atol 1e-5` and `--rtol 1e-6` (the defaults): pandas prints 6 decimals, so `atol` absorbs last-digit rounding. Tighten them only if asked.
- `--max-lines 40`: the maximum number of diff lines printed per cell.
- `--write`: saves the re-executed notebook over the original. Only use it when the user asks to refresh the committed outputs. It rewrites every embedded PNG, so the git diff is large.

## Setup

The notebook needs the packages in `requirements.txt`, and the checker needs `nbformat`, `nbclient` and `ipykernel`:

```
pip install -r requirements.txt nbformat nbclient ipykernel
```

Don't run this on `Macro4Project.ipynb` (v1). It imports numba, which the package dropped, so it fails at cell 1.
