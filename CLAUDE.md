# macro4-project

An Aiyagari (1994) model of Finland's 2024–2025 asset-tested benefit reforms. README.md covers the model, the package layout and the FIONA data workflow.

## Don't read notebooks directly

The .ipynb files embed their plots as base64 PNGs. `Macro4Project.ipynb` holds about 774k characters of image data against 26k of text output. Opening it with Read, cat or grep fills the context with image bytes.

Use `tools/nbtext.py`, which prints code and text output and replaces images with `[image/png omitted]`:

- `python3 tools/nbtext.py Macro4Project_v2.ipynb --outline` prints one line per cell, numbered.
- `python3 tools/nbtext.py Macro4Project_v2.ipynb 9` prints cell 9's source and output.
- `python3 tools/nbtext.py Macro4Project_v2.ipynb 4-9` prints a range.

Which notebook is which:

- `Macro4Project_v2.ipynb` is the live project notebook.
- `Macro4Project.ipynb` is the frozen v1 from the initial commit. It needs numba and the pre-refactor package, so it no longer runs. Treat it as read-only reference.

## Checking results

After changing `aiyagari/` or the v2 notebook, use the `notebook-check` skill (`python3 tools/nbcheck.py Macro4Project_v2.ipynb`). It re-executes the notebook in about 10 seconds and prints only the numbers that moved.

For the register-data estimators, run `python3 examples/fiona_estimation_template.py --selftest`. It should end with `SELFTEST PASSED`.

## Constraints on aiyagari/

The package gets copied as plain files into Statistics Finland's FIONA environment: no install step and no internet. Keep it to numpy, scipy and pandas. Don't add numba, compiled extensions or new dependencies.
