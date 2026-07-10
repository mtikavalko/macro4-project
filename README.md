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
2. Estimate the calibration moments from the registers (see the variable
   list below).
3. Put the estimates into a `CalibrationTargets` (directly or via
   `CalibrationTargets.from_json`), build the model with
   `params_from_targets(targets, regime=...)`, and compare
   `compare_moments(model, targets)`.
4. Fit free parameters (β, `sigma_e`, flow rates, …) to the moments with
   `smm_calibrate`, then evaluate the reform regimes (`"pre"`, `"atest"`,
   `"orpo"`) and welfare with `aiyagari.welfare`.

A rich-process general-equilibrium solve takes seconds on a normal
workstation, so SMM loops over a handful of parameters are practical.
The estimation scripts only need to output aggregate moments into a
targets JSON, which is also convenient for FIONA's output-clearance
rules.

### Required register data and variables

Request the datasets through the
[Taika research data catalogue](https://taika.stat.fi/en/); exact variable
names shift between module versions, so confirm them in Taika's variable
descriptions when writing the data-permit application.

**1. Labor-market flows — FOLK basic module (`folk_perus`), person panel**

| Variable (concept) | Used for |
|---|---|
| pseudonymized person id (`shnro`), year | panel linking |
| main type of activity (`ptoim1`) | classify person-years as E / U / N |
| age, sex | restrict to working age (e.g. 18–64) |
| household-dwelling unit id (asuntokunta) | link persons to households for the wealth block |

Year-to-year `ptoim1` transitions give the annual E/U/N matrix directly
(the model period is one year): `p_eu, p_ue, p_en, p_ne, p_un, p_nu`,
plus `urate` and `participation`. Note that register-based activity
status differs from LFS/ILO definitions — use one definition
consistently for both the flows and the rate targets.

**2. Earnings process — FOLK income module (`folk_tulo`)**

| Variable (concept) | Used for |
|---|---|
| annual wage and salary earnings (palkkatulot) | log-earnings AR(1): `rho_e`, `sigma_e`, `var_log_earnings`; mean earnings → `eur_avg_wage` |
| entrepreneurial income | include/exclude self-employed consistently |
| education, occupation (from `folk_perus`) | residualize age/education profiles before fitting the AR(1) |

**3. Benefits — income components in `folk_tulo` (or a Kela extract)**

| Variable (concept) | Used for |
|---|---|
| earnings-related unemployment allowance | `rr_ui_pre/post` = benefit ÷ own pre-unemployment wage, person by person |
| Kela basic allowance / labour market subsidy | `ui_floor_eur`, `b_n_eur` |
| general housing allowance (asumistuki) | `ha_pre_eur` / `ha_post_eur` among unemployed and inactive recipients |

The statutory parameters (asset-test thresholds EUR 10k/50k, staggering
percentages) come from law, not data.

**4. Wealth distribution — Households' assets statistics (`vtutk`),
household-level waves (2016/2019/2023)**

| Variable (concept) | Used for |
|---|---|
| net wealth (nettovarallisuus) | `mean/median_wealth_eur`, `wealth_p10/p90_eur`, `wealth_gini` |
| financial assets (rahoitusvarat: deposits, listed shares, funds) | asset-test exposure: share of benefit-recipient households below EUR 10k / above EUR 50k — this is what the statutory test screens |
| debts | net positions, `borrow_limit_eur` |

Estimate the asset-test exposure under both definitions (net worth and
financial assets): the model holds all wealth in one asset, so to
reproduce the *empirical* bite share you can remap `atest_lo_eur` /
`atest_hi_eur` to the net-worth levels matching the financial-asset
exposure — one line in the targets, no code change.

**Not needed from FIONA:** `alpha`, `delta`, `ky` (public national
accounts), `eta` (set a priori), `beta` (fitted with `smm_calibrate`
after the data moments are in).

## Reproducing the notebook

```
pip install -r requirements.txt
jupyter notebook Macro4Project_v2.ipynb
```
