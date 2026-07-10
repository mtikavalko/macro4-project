# macro4-project

Asset-tested benefits in an Aiyagari (1994) economy, calibrated to Finland
and the Orpo government's 2024–2025 social-security reforms (housing
allowance asset test, housing allowance cut, staggering of the
earnings-related unemployment allowance).

- `Macro4Project_v2.ipynb` — the project notebook: calibration, the three
  policy regimes, welfare evaluation, diagnostics.
- `aiyagari/` — the model package (pure numpy/scipy/pandas):
  - `labor.py` — labor-state processes: the original 2-state chain, and a
    general builder combining a Rouwenhorst AR(1) for earnings with
    employment / unemployment / non-participation flows.
  - `model.py` — the economy and the endogenous-grid household solver.
  - `solvers.py` — stationary distribution (verified sparse LU / ARPACK),
    general equilibrium (Illinois false position), β calibration.
  - `calibration.py` — `CalibrationTargets` (JSON-serializable data
    moments), regime parameter builders, model-vs-data moment comparison,
    SMM fitting.
  - `welfare.py` — consumption-equivalent welfare and diagnostics.
- `examples/fiona_workflow.py` — end-to-end data-calibration workflow.

## Requirements

`numpy`, `scipy`, `pandas` (`matplotlib` only for the notebook plots).
No compilation, no JIT, no internet access needed at runtime.

## Using the model in FIONA (Statistics Finland remote access)

The package is designed to be copied as-is into restricted research
environments: it is plain Python files with no install step and no
dependencies beyond the standard scientific stack that FIONA's Python
distribution ships.

1. Transfer the `aiyagari/` folder (and optionally
   `examples/fiona_workflow.py`) into your FIONA project directory via the
   normal material import procedure.
2. Estimate the calibration moments from the registers, e.g.
   - **FOLK / employment statistics** → annual transition rates
     `p_eu, p_ue, p_en, p_ne, p_un, p_nu`, the unemployment rate and the
     participation rate;
   - **income register panel** → persistence `rho_e` and dispersion
     `sigma_e` of log labor earnings (plus `var_log_earnings` as a check);
   - **wealth survey (HVT)** → mean/median wealth, percentiles, Gini, and
     the share of benefit recipients under the €10k / over €50k asset-test
     thresholds;
   - **Kela benefit registers** → replacement rates and housing-allowance
     levels.
3. Put the estimates into a `CalibrationTargets` (directly or via
   `CalibrationTargets.from_json`), build the model with
   `params_from_targets(targets, regime=...)`, and compare
   `compare_moments(model, targets)`.
4. Fit free parameters (β, `sigma_e`, flow rates, …) to the moments with
   `smm_calibrate`, then evaluate the reform regimes (`"pre"`, `"atest"`,
   `"orpo"`) and welfare with `aiyagari.welfare`.

A rich-process general-equilibrium solve takes seconds on a normal
workstation, so SMM loops over a handful of parameters are practical.

## Reproducing the notebook

```
pip install -r requirements.txt
jupyter notebook Macro4Project_v2.ipynb
```
