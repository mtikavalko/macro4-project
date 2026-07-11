"""
FIONA estimation template: register microdata -> CalibrationTargets JSON.

Four estimation blocks, one per CalibrationTargets group:

  1. labor-market flows        FOLK basic  (folk_perus)
  2. earnings process          FOLK income (folk_tulo), employed only
  3. benefit levels            FOLK income components (or a Kela extract)
  4. wealth distribution       Households' assets statistics (vtutk)

Inside FIONA you only need to
  (a) point CONFIG["..."] at the data files,
  (b) fix the variable names in VAR (confirm them in the Taika catalogue,
      https://taika.stat.fi/en/ - the names below are common but shift
      between module versions), and
  (c) check the PTOIM_TO_EUN activity-code mapping.

Then:  python fiona_estimation_template.py
writes targets_estimated.json, ready for examples/fiona_workflow.py.

Self-test (runs anywhere, no data needed): generates a synthetic
FOLK-like panel with known parameters and verifies that every estimator
recovers them:  python fiona_estimation_template.py --selftest

Only numpy / pandas / (aiyagari) are used - no statsmodels, no internet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiyagari import CalibrationTargets
from aiyagari.calibration import _weighted_percentile, _weighted_gini

# ============================================================== CONFIG
CONFIG = {
    # data files (csv or parquet; export sas7bdat to one of these first)
    "folk_perus": "FILL_IN/folk_perus.parquet",
    "folk_tulo":  "FILL_IN/folk_tulo.parquet",
    "vtutk":      "FILL_IN/vtutk_2023.parquet",

    "working_age": (18, 64),
    "min_annual_wage_eur": 1_000.0,   # drop marginal jobs from the AR(1)
    "pre_reform_years": range(2018, 2024),   # calibration window
    # deflator to the base year (index, e.g. CPI); None = skip deflation
    "cpi": None,                       # e.g. {2018: 0.90, ..., 2023: 1.0}
}

# variable names - CONFIRM IN TAIKA before running
VAR = {
    "pid": "shnro",            # pseudonymized person id
    "year": "vuosi",
    "activity": "ptoim1",      # main type of activity
    "age": "ika",
    "wage": "palk",            # annual wage and salary earnings, EUR
    "educ": "koulutusaste",    # education level (optional; "" to skip)
    # benefit income components, EUR/year
    "ui_earnings_related": "ansiopvraha",
    "ui_basic": "peruspvraha_tmtuki",   # basic allowance + labour mkt subsidy
    "housing_allowance": "asumistuki",
    # wealth statistics (household level)
    "hh_id": "knro",
    "net_wealth": "nettovarallisuus",
    "fin_assets": "rahoitusvarat",
    "weight": "paino",         # survey weight
    "hh_unemployed": "tyoton_jasen",    # any unemployed member (build if absent)
}

# ptoim1 codes -> E/U/N.  CONFIRM the coding in Taika: typically
# 11 = employed, 12 = unemployed, everything else out of the labor force.
PTOIM_TO_EUN = {11: "E", 12: "U"}


def read_any(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _deflate(nominal, years):
    cpi = CONFIG["cpi"]
    if cpi is None:
        return nominal
    return nominal / years.map(cpi)


# ==================================================== 1. labor flows
def estimate_flows(perus):
    """Annual E/U/N transition rates and stock shares from folk_perus."""
    v = VAR
    lo, hi = CONFIG["working_age"]
    d = perus[(perus[v["age"]] >= lo) & (perus[v["age"]] <= hi)].copy()
    d["state"] = d[v["activity"]].map(PTOIM_TO_EUN).fillna("N")

    d = d.sort_values([v["pid"], v["year"]])
    g = d.groupby(v["pid"])
    d["state_next"] = g["state"].shift(-1)
    d["year_next"] = g[v["year"]].shift(-1)
    pairs = d[d["year_next"] == d[v["year"]] + 1]

    P = pd.crosstab(pairs["state"], pairs["state_next"], normalize="index")
    P = P.reindex(index=list("EUN"), columns=list("EUN"), fill_value=0.0)

    shares = d["state"].value_counts(normalize=True)
    lf = shares.get("E", 0.0) + shares.get("U", 0.0)
    return {
        "p_eu": float(P.loc["E", "U"]), "p_en": float(P.loc["E", "N"]),
        "p_ue": float(P.loc["U", "E"]), "p_un": float(P.loc["U", "N"]),
        "p_ne": float(P.loc["N", "E"]), "p_nu": float(P.loc["N", "U"]),
        "urate": float(shares.get("U", 0.0) / lf) if lf > 0 else np.nan,
        "participation": float(lf),
    }


# ================================================ 2. earnings process
def _residualize(logw, age, year, educ=None):
    """OLS residuals of log wages on an age profile and year dummies."""
    cols = [np.ones(len(logw)), age, age ** 2]
    cols += [(year == y).astype(float) for y in np.unique(year)[1:]]
    if educ is not None:
        cols += [(educ == e).astype(float) for e in np.unique(educ)[1:]]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, logw, rcond=None)
    return logw - X @ beta


def estimate_earnings_process(perus, tulo):
    """AR(1) of residualized log real wages among the employed."""
    v = VAR
    d = perus.merge(tulo, on=[v["pid"], v["year"]], how="inner")
    lo, hi = CONFIG["working_age"]
    emp = d[(d[v["activity"]].map(PTOIM_TO_EUN) == "E")
            & (d[v["age"]] >= lo) & (d[v["age"]] <= hi)
            & (d[v["wage"]] > CONFIG["min_annual_wage_eur"])].copy()

    emp["wage_real"] = _deflate(emp[v["wage"]], emp[v["year"]])
    emp["logw"] = np.log(emp["wage_real"])
    educ = emp[v["educ"]].values if v["educ"] and v["educ"] in emp else None
    emp["res"] = _residualize(emp["logw"].values,
                              emp[v["age"]].values.astype(float),
                              emp[v["year"]].values, educ)

    emp = emp.sort_values([v["pid"], v["year"]])
    g = emp.groupby(v["pid"])
    emp["res_lag"] = g["res"].shift(1)
    emp["year_lag"] = g[v["year"]].shift(1)
    consec = emp[emp["year_lag"] == emp[v["year"]] - 1]

    rho = float(np.cov(consec["res"], consec["res_lag"])[0, 1]
                / consec["res_lag"].var())
    return {
        "rho_e": rho,
        "sigma_e": float(emp["res"].std()),
        "var_log_earnings": float(emp["res"].var()),
        "eur_avg_wage": float(emp["wage_real"].mean()),
    }


# ======================================================= 3. benefits
def estimate_benefits(perus, tulo):
    """
    Replacement rates and benefit levels, in real (deflated) euros when
    CONFIG["cpi"] is set - consistent with the wage anchor.

    rr_ui = earnings-related allowance (annualized) / own previous-year
    wage, for people employed in t-1 and unemployed in t.  The housing
    allowance level is estimated over ALL non-employed recipients, because
    the model pays the same asset-tested b_ha in the unemployed and
    inactive states.
    """
    v = VAR
    d = perus.merge(tulo, on=[v["pid"], v["year"]], how="inner")
    # same working-age window as the flow and earnings blocks - otherwise
    # the non-employed HA pool picks up retirees, students and minors that
    # are outside the model's labor process
    lo, hi = CONFIG["working_age"]
    d = d[(d[v["age"]] >= lo) & (d[v["age"]] <= hi)].copy()
    d["state"] = d[v["activity"]].map(PTOIM_TO_EUN).fillna("N")
    for col in (v["wage"], v["ui_earnings_related"], v["ui_basic"],
                v["housing_allowance"]):
        d[col] = _deflate(d[col], d[v["year"]])
    d = d.sort_values([v["pid"], v["year"]])
    g = d.groupby(v["pid"])
    d["state_prev"] = g["state"].shift(1)
    d["wage_prev"] = g[v["wage"]].shift(1)
    d["year_prev"] = g[v["year"]].shift(1)

    new_u = d[(d["state"] == "U") & (d["state_prev"] == "E")
              & (d["year_prev"] == d[v["year"]] - 1)
              & (d["wage_prev"] > CONFIG["min_annual_wage_eur"])
              & (d[v["ui_earnings_related"]] > 0)]
    rr_ui = float((new_u[v["ui_earnings_related"]]
                   / new_u["wage_prev"]).median())

    unemp = d[d["state"] == "U"]
    nonemp = d[d["state"] != "E"]
    ha_rec = nonemp[nonemp[v["housing_allowance"]] > 0]
    return {
        "rr_ui_pre": rr_ui,
        "ui_floor_eur": float(
            unemp.loc[unemp[v["ui_basic"]] > 0, v["ui_basic"]].median()),
        "ha_pre_eur": float(ha_rec[v["housing_allowance"]].mean()),
    }


# ========================================================= 4. wealth
def estimate_wealth(vtutk):
    """Weighted wealth distribution and asset-test exposure (vtutk)."""
    v = VAR
    x = vtutk[v["net_wealth"]].values.astype(float)
    w = vtutk[v["weight"]].values.astype(float) \
        if v["weight"] in vtutk else np.ones(len(vtutk))

    out = {
        "mean_wealth_eur": float(np.average(x, weights=w)),
        "median_wealth_eur": _weighted_percentile(x, w, 50),
        "wealth_p10_eur": _weighted_percentile(x, w, 10),
        "wealth_p90_eur": _weighted_percentile(x, w, 90),
        "wealth_gini": _weighted_gini(x, w),
    }
    # asset-test exposure: financial assets of benefit-recipient households
    if v["hh_unemployed"] in vtutk and v["fin_assets"] in vtutk:
        rec = vtutk[vtutk[v["hh_unemployed"]] > 0]
        fa = rec[v["fin_assets"]].values.astype(float)
        wr = rec[v["weight"]].values.astype(float) \
            if v["weight"] in rec else np.ones(len(rec))
        out["exposure_fin_below_10k"] = float(wr[fa < 10_000].sum() / wr.sum())
        out["exposure_fin_above_50k"] = float(wr[fa > 50_000].sum() / wr.sum())
    return out


# ======================================================== assembly
def build_targets(flows, earnings, benefits, wealth, out_path):
    """
    Combine the estimated blocks into a CalibrationTargets JSON.

    Post-reform benefit levels are derived from the ESTIMATED pre-reform
    values by applying the statutory Orpo-reform factors, so the reform
    experiment stays anchored to the same data (leaving them at the class
    defaults would silently compare against unrelated placeholder levels).
    Once post-reform register years are available, estimate them directly
    instead.
    """
    from aiyagari.calibration import UI_REFORM_FACTOR, HA_REFORM_FACTOR
    t = CalibrationTargets(
        eur_avg_wage=earnings["eur_avg_wage"],
        p_eu=flows["p_eu"], p_ue=flows["p_ue"], p_en=flows["p_en"],
        p_ne=flows["p_ne"], p_un=flows["p_un"], p_nu=flows["p_nu"],
        urate=flows["urate"], participation=flows["participation"],
        rho_e=earnings["rho_e"], sigma_e=earnings["sigma_e"],
        var_log_earnings=earnings["var_log_earnings"],
        rr_ui_pre=benefits["rr_ui_pre"],
        rr_ui_post=benefits["rr_ui_pre"] * UI_REFORM_FACTOR,
        ui_floor_eur=benefits["ui_floor_eur"],
        ha_pre_eur=benefits["ha_pre_eur"],
        ha_post_eur=benefits["ha_pre_eur"] * HA_REFORM_FACTOR,
        mean_wealth_eur=wealth["mean_wealth_eur"],
        median_wealth_eur=wealth["median_wealth_eur"],
        wealth_p10_eur=wealth["wealth_p10_eur"],
        wealth_p90_eur=wealth["wealth_p90_eur"],
        wealth_gini=wealth["wealth_gini"],
        # statutory, not estimated: asset-test thresholds
    )
    t.to_json(out_path)
    return t


def main():
    perus = read_any(CONFIG["folk_perus"])
    tulo = read_any(CONFIG["folk_tulo"])
    vtutk = read_any(CONFIG["vtutk"])

    yrs = CONFIG["pre_reform_years"]
    perus = perus[perus[VAR["year"]].isin(yrs)]
    tulo = tulo[tulo[VAR["year"]].isin(yrs)]

    flows = estimate_flows(perus)
    earnings = estimate_earnings_process(perus, tulo)
    benefits = estimate_benefits(perus, tulo)
    wealth = estimate_wealth(vtutk)

    out = Path(__file__).with_name("targets_estimated.json")
    t = build_targets(flows, earnings, benefits, wealth, out)
    print(f"targets written to {out}")
    for block in (flows, earnings, benefits, wealth):
        for k, val in block.items():
            print(f"  {k:<26s} {val:,.4f}")
    return t


# ======================================================== self-test
def _selftest():
    """Synthetic FOLK-like panel with known parameters; verify recovery."""
    rng = np.random.default_rng(0)
    N, T = 30_000, 6
    TRUE = dict(p_eu=0.07, p_ue=0.65, p_en=0.02, p_ne=0.12, p_un=0.10,
                p_nu=0.02, rho=0.95, sigma=0.40, rr_ui=0.45)

    P = np.array([  # E U N
        [1 - TRUE["p_eu"] - TRUE["p_en"], TRUE["p_eu"], TRUE["p_en"]],
        [TRUE["p_ue"], 1 - TRUE["p_ue"] - TRUE["p_un"], TRUE["p_un"]],
        [TRUE["p_ne"], TRUE["p_nu"], 1 - TRUE["p_ne"] - TRUE["p_nu"]]])
    states = np.zeros((N, T), dtype=int)
    states[:, 0] = rng.choice(3, N, p=[0.72, 0.07, 0.21])
    z = rng.normal(0, TRUE["sigma"], N)          # latent log productivity
    age0 = rng.integers(20, 58, N)
    rows = []
    for t in range(T):
        if t > 0:
            u = rng.random(N)
            cum = np.cumsum(P[states[:, t - 1]], axis=1)
            states[:, t] = (u[:, None] > cum).sum(axis=1)
            z = TRUE["rho"] * z + rng.normal(
                0, TRUE["sigma"] * np.sqrt(1 - TRUE["rho"] ** 2), N)
        age = age0 + t
        wage = np.where(states[:, t] == 0,
                        np.exp(10.4 + 0.03 * age - 3e-4 * age ** 2 + z), 0.0)
        prev_wage = np.array([r["palk"] for r in rows[-N:]]) \
            if t > 0 else np.zeros(N)
        ui = np.where((states[:, t] == 1) & (prev_wage > 0),
                      TRUE["rr_ui"] * prev_wage, 0.0)
        basic = np.where((states[:, t] == 1) & (ui == 0), 9_600.0, 0.0)
        ha = np.where((states[:, t] != 0) & (rng.random(N) < 0.6), 3_900.0, 0)
        for i in range(N):
            rows.append(dict(shnro=i, vuosi=2018 + t, age=age[i],
                             ptoim1=(11, 12, 22)[states[i, t]],
                             palk=wage[i], ansiopvraha=ui[i],
                             peruspvraha_tmtuki=basic[i], asumistuki=ha[i]))
    df = pd.DataFrame(rows)
    VAR.update(age="age", educ="")
    perus = df[["shnro", "vuosi", "age", "ptoim1"]]
    tulo = df[["shnro", "vuosi", "palk", "ansiopvraha",
               "peruspvraha_tmtuki", "asumistuki"]]

    flows = estimate_flows(df)
    earn = estimate_earnings_process(perus, tulo)
    ben = estimate_benefits(perus, tulo)

    wealth_df = pd.DataFrame({
        "knro": np.arange(5_000),
        "nettovarallisuus": np.exp(rng.normal(11.5, 1.1, 5_000)) - 5_000,
        "rahoitusvarat": np.exp(rng.normal(9.0, 1.5, 5_000)),
        "paino": rng.uniform(0.5, 2.0, 5_000),
        "tyoton_jasen": rng.random(5_000) < 0.1})
    wea = estimate_wealth(wealth_df)

    # deflation consistency: with a CPI of 0.5 everywhere, real benefit
    # levels must double
    from aiyagari.calibration import UI_REFORM_FACTOR, HA_REFORM_FACTOR
    CONFIG["cpi"] = {y: 0.5 for y in range(2018, 2018 + T)}
    ben_real = estimate_benefits(perus, tulo)
    CONFIG["cpi"] = None

    checks = [
        ("p_eu", flows["p_eu"], TRUE["p_eu"], 0.01),
        ("p_ue", flows["p_ue"], TRUE["p_ue"], 0.02),
        ("p_ne", flows["p_ne"], TRUE["p_ne"], 0.01),
        ("rho_e", earn["rho_e"], TRUE["rho"], 0.03),
        ("sigma_e", earn["sigma_e"], TRUE["sigma"], 0.03),
        ("rr_ui_pre", ben["rr_ui_pre"], TRUE["rr_ui"], 0.02),
        ("ui_floor", ben["ui_floor_eur"], 9_600.0, 1.0),
        ("ha_pre", ben["ha_pre_eur"], 3_900.0, 1.0),
        ("ui_floor deflated", ben_real["ui_floor_eur"], 19_200.0, 1.0),
        ("ha_pre deflated", ben_real["ha_pre_eur"], 7_800.0, 1.0),
    ]
    ok = True
    for name, got, want, tol in checks:
        passed = abs(got - want) < tol
        ok &= passed
        print(f"  {name:<12s} est {got:9.4f}  true {want:9.4f}  "
              f"{'OK' if passed else 'FAIL'}")

    out = Path(__file__).with_name("targets_selftest.json")
    t = build_targets(flows, earn, ben, wea, out)
    t2 = CalibrationTargets.from_json(out)
    assert t2 == t
    out.unlink()
    print("  JSON round-trip OK")

    # post-reform levels must follow the estimated pre values via the
    # statutory factors, not the class defaults
    assert abs(t.rr_ui_post - ben["rr_ui_pre"] * UI_REFORM_FACTOR) < 1e-12
    assert abs(t.ha_post_eur - ben["ha_pre_eur"] * HA_REFORM_FACTOR) < 1e-12
    print(f"  post-reform derived OK (rr_ui {t.rr_ui_post:.4f}, "
          f"ha {t.ha_post_eur:,.0f} EUR/yr)")
    assert ok, "SELFTEST FAILED"
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
