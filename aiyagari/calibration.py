"""
Calibration interface: Finnish targets, euro anchoring, and a data-driven
workflow for restricted microdata environments (Statistics Finland FIONA).

Workflow for actual data calibration
------------------------------------
1. Outside or inside FIONA, estimate the empirical moments from register
   data (FOLK for labor-market flows and participation, the income
   register for the earnings process, the wealth survey for the asset
   distribution) and put them in a `CalibrationTargets` - directly or via
   JSON (`CalibrationTargets.from_json` / `.to_json`).
2. Build a model with `params_from_targets(targets, regime=...)`; the
   labor process (employed productivity states x unemployed with
   earnings-related benefits x non-participation) is constructed from the
   estimated flow rates and the AR(1) earnings process.
3. Solve and compare `model_moments(mod, targets)` with the data via
   `compare_moments`, or fit free parameters with `smm_calibrate`.

All default values below are approximations from public sources (see the
project notebook); they are placeholders to be replaced with
register-based estimates inside FIONA.

Macro anchors: unemployment 8.4 % (2024), participation ~79 % (15-64),
K/Y ~ 3.3, average gross wage ~ EUR 48,000/yr, mean household net wealth
~ EUR 230,000.

Policy (Orpo government reforms 2024-2025), as mapped into the model:
  1. Earnings-related allowance staggering (2 Sep 2024: 80 % after 40
     benefit days, 75 % after 170 days) + abolished child supplements:
     average UI replacement rate 45 % -> 38 % of the own gross wage.
  2. General housing allowance cut (1 Apr 2024: compensation 80 -> 70 %,
     deductible 42 -> 50 %): HA level -17 %.
  3. Housing allowance asset test (1 Jan 2025): linear phase-out of the
     HA between EUR 10,000 and EUR 50,000 of assets.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict, replace, field

from .labor import build_labor_process, two_state_process
from .model import AiyagariModel
from .solvers import solve_general_equilibrium


# ================================================================ targets
@dataclass
class CalibrationTargets:
    # --- euro anchor and technology --------------------------------------
    eur_avg_wage: float = 48_000.0   # gross average annual earnings
    ky: float = 3.3                  # annual capital-output ratio
    alpha: float = 0.38
    delta: float = 0.06
    eta: float = 2.0
    beta: float = 0.94926            # re-calibrate with calibrate_beta/SMM

    # --- labor-market flows, per year (estimate from FOLK spells) --------
    p_eu: float = 0.0734             # E -> U (=> urate 8.4 % with the flows below)
    p_ue: float = 0.70               # U -> E
    p_en: float = 0.02               # E -> N   (exit from labor force)
    p_ne: float = 0.10               # N -> E   (entry into employment)
    p_un: float = 0.10               # U -> N
    p_nu: float = 0.00               # N -> U

    # --- earnings process (estimate from income register panel) ----------
    n_e: int = 5                     # productivity states among employed
    rho_e: float = 0.96              # annual persistence of log earnings
    sigma_e: float = 0.45            # stationary s.d. of log earnings

    # --- benefits, euros per year where absolute --------------------------
    rr_ui_pre: float = 0.45          # UI replacement on own gross wage
    rr_ui_post: float = 0.38         # after the staggering reform
    ui_floor_eur: float = 9_600.0    # basic allowance ~ EUR 800/month
    ui_cap_eur: float = 36_000.0     # effective ER-allowance cap
    b_n_eur: float = 7_200.0         # transfer when out of the labor force
    ha_pre_eur: float = 3_888.0      # housing allowance ~ EUR 324/month
    ha_post_eur: float = 3_216.0     # after the April 2024 cut
    atest_lo_eur: float = 10_000.0   # asset test starts
    atest_hi_eur: float = 50_000.0   # allowance fully gone
    borrow_limit_eur: float = 12_000.0

    # --- distributional targets (fill from microdata; None = not targeted)
    urate: float = 0.084
    participation: float = 0.79
    mean_wealth_eur: float = 230_000.0
    median_wealth_eur: float = None
    wealth_p10_eur: float = None
    wealth_p90_eur: float = None
    wealth_gini: float = None
    var_log_earnings: float = None   # among the employed

    # ------------------------------------------------- unit conversion
    @property
    def w_ref(self):
        """Reference model wage implied by the K/Y target."""
        return (1.0 - self.alpha) * self.ky ** (self.alpha / (1.0 - self.alpha))

    @property
    def eur_per_unit(self):
        return self.eur_avg_wage / self.w_ref

    def eur(self, x_model):
        return x_model * self.eur_per_unit

    def model_units(self, x_eur):
        return x_eur / self.eur_per_unit

    # ------------------------------------------------------- JSON I/O
    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def replace(self, **kw):
        return replace(self, **kw)


DEFAULT = CalibrationTargets()


# ================================================== model construction
def labor_process_from_targets(t, reform=False):
    """Rich labor process (E x productivity, U with own-wage UI, N)."""
    return build_labor_process(
        n_e=t.n_e, rho_e=t.rho_e, sigma_e=t.sigma_e,
        p_eu=t.p_eu, p_ue=t.p_ue, p_en=t.p_en, p_ne=t.p_ne,
        p_un=t.p_un, p_nu=t.p_nu,
        rr_ui=(t.rr_ui_post if reform else t.rr_ui_pre),
        ui_floor=t.model_units(t.ui_floor_eur),
        ui_cap=t.model_units(t.ui_cap_eur),
        b_n=t.model_units(t.b_n_eur),
        w_anchor=t.w_ref,
    )


def params_from_targets(t, regime="pre", rich_labor=True, amax=40.0,
                        **grid_kwargs):
    """
    Keyword arguments for AiyagariModel under a policy regime:
      "pre"   - pre-reform benefit system, no asset test
      "atest" - pre-reform levels + the 2025 asset test
      "orpo"  - full reform package (staggering + HA cut + asset test)
    """
    if regime not in ("pre", "atest", "orpo"):
        raise ValueError(f"unknown regime {regime!r}")
    reform_levels = regime == "orpo"
    asset_test = regime in ("atest", "orpo")

    b_ha = t.model_units(t.ha_post_eur if reform_levels else t.ha_pre_eur)
    a_lo = t.model_units(t.atest_lo_eur)
    a_hi = t.model_units(t.atest_hi_eur)

    p = dict(
        beta=t.beta, eta=t.eta, alpha=t.alpha, delta=t.delta,
        b_ha=b_ha,
        phi_a=(b_ha / (a_hi - a_lo)) if asset_test else 0.0,
        a_thresh=a_lo,
        amin=-t.model_units(t.borrow_limit_eur),
        amax=amax,
        **grid_kwargs,
    )
    if rich_labor:
        p["labor"] = labor_process_from_targets(t, reform=reform_levels)
    else:  # original two-state economy
        rr = t.rr_ui_post if reform_levels else t.rr_ui_pre
        p.update(b_ui=rr * t.w_ref, peu=t.p_eu, pue=t.p_ue)
    # decent starting guess for the equilibrium K
    L = p["labor"].aggregate_labor() if rich_labor else \
        t.p_ue / (t.p_eu + t.p_ue)
    p["K0"] = t.ky ** (1.0 / (1.0 - t.alpha)) * L
    return p


# ======================================================== model moments
def _weighted_percentile(x, w, q):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    cw /= cw[-1]
    return float(np.interp(q / 100.0, cw, x[order]))


def _weighted_gini(x, w):
    order = np.argsort(x)
    x, w = x[order], w[order]
    w = w / w.sum()
    cx = np.cumsum(w * x)
    mu = cx[-1]
    if mu <= 0:
        return float("nan")
    return float(1.0 - np.sum(w * (np.concatenate([[0.0], cx[:-1]]) + cx)) / mu)


def model_moments(mod, t):
    """Model counterparts of the calibration targets, euros where relevant."""
    lab = mod.labor
    grid_eur = t.eur(mod.a_grid_dist)
    w_a = mod.dist.sum(axis=1)                     # wealth marginal

    m = {
        "K/Y": mod.K / mod.output(),
        "r": mod.r,
        "tau": mod.tau,
        "urate": lab.unemployment_rate(),
        "participation": lab.participation_rate(),
        "employment_rate": lab.employment_rate(),
        "mean_wealth_eur": float(np.dot(w_a, grid_eur)),
        "median_wealth_eur": _weighted_percentile(grid_eur, w_a, 50),
        "wealth_p10_eur": _weighted_percentile(grid_eur, w_a, 10),
        "wealth_p90_eur": _weighted_percentile(grid_eur, w_a, 90),
        "wealth_gini": _weighted_gini(grid_eur, w_a),
        "share_borrowing": float(mod.dist[mod.a_grid_dist < 0.0, :].sum()),
        "benefit_costs/Y": mod.aggregate_transfer() / mod.output(),
    }
    # asset-test exposure among HA-eligible states
    elig = mod.dist[:, lab.ha_elig]
    elig_mass = elig.sum()
    if elig_mass > 0:
        below = mod.a_grid_dist < t.model_units(t.atest_lo_eur)
        above = mod.a_grid_dist > t.model_units(t.atest_hi_eur)
        m["ha_share_assets_below_10k"] = float(elig[below].sum() / elig_mass)
        m["ha_share_assets_above_50k"] = float(elig[above].sum() / elig_mass)
    # labor income distribution among the employed
    emp = lab.is_employed
    if emp.sum() > 1:
        inc = mod.w * lab.e[emp]
        wts = mod.pi_stat[emp]
        m["labor_income_gini"] = _weighted_gini(inc, wts)
        lm = np.log(inc)
        mu = np.average(lm, weights=wts)
        m["var_log_earnings"] = float(np.average((lm - mu) ** 2, weights=wts))
    return m


def compare_moments(mod, t):
    """DataFrame of model moments next to the targets that are set."""
    import pandas as pd
    m = model_moments(mod, t)
    tgt = {
        "K/Y": t.ky, "urate": t.urate, "participation": t.participation,
        "mean_wealth_eur": t.mean_wealth_eur,
        "median_wealth_eur": t.median_wealth_eur,
        "wealth_p10_eur": t.wealth_p10_eur,
        "wealth_p90_eur": t.wealth_p90_eur,
        "wealth_gini": t.wealth_gini,
        "var_log_earnings": t.var_log_earnings,
    }
    rows = [(k, v, tgt.get(k)) for k, v in m.items()]
    return pd.DataFrame(rows, columns=["moment", "model", "target"])


# ============================================================== fitting
def smm_calibrate(t, param_names, bounds, moment_targets, regime="pre",
                  weights=None, rich_labor=True, model_kwargs=None,
                  maxiter=60, verbose=True):
    """
    Fit CalibrationTargets fields to data moments by simulated method of
    moments (Nelder-Mead on the weighted relative moment distance).

    param_names    : list of CalibrationTargets field names to vary,
                     e.g. ["beta", "sigma_e", "p_ue"]
    bounds         : list of (lo, hi) per parameter
    moment_targets : dict moment name (key of model_moments) -> data value,
                     e.g. {"K/Y": 3.3, "wealth_gini": 0.65, "urate": 0.084}
    Returns (fitted_targets, solved_model, scipy_result).
    """
    from scipy.optimize import minimize

    weights = weights or {k: 1.0 for k in moment_targets}
    model_kwargs = model_kwargs or {}
    x0 = np.array([getattr(t, p) for p in param_names], dtype=float)
    cache = {}

    def solve_for(x):
        t_try = t.replace(**dict(zip(param_names, x)))
        mod = AiyagariModel(**params_from_targets(
            t_try, regime=regime, rich_labor=rich_labor, **model_kwargs))
        solve_general_equilibrium(mod, verbose=False)
        return t_try, mod

    def objective(x):
        t_try, mod = solve_for(x)
        m = model_moments(mod, t_try)
        loss = 0.0
        for k, target in moment_targets.items():
            scale = abs(target) if target != 0 else 1.0
            loss += weights[k] * ((m[k] - target) / scale) ** 2
        cache["last"] = (t_try, mod)
        if verbose:
            pstr = ", ".join(f"{p}={v:.4f}" for p, v in zip(param_names, x))
            print(f"  loss={loss:.6f}  {pstr}")
        return loss

    res = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": maxiter, "xatol": 1e-4,
                            "fatol": 1e-6})
    t_fit, mod_fit = solve_for(res.x)
    return t_fit, mod_fit, res


# ==================================== legacy two-state API (unchanged)
EUR_AVG_WAGE = DEFAULT.eur_avg_wage
URATE_TARGET = DEFAULT.urate
KY_TARGET = DEFAULT.ky
ALPHA = DEFAULT.alpha
DELTA = DEFAULT.delta
ETA = DEFAULT.eta
PUE = DEFAULT.p_ue
PEU = URATE_TARGET / (1.0 - URATE_TARGET) * PUE
BETA = DEFAULT.beta

W_REF = DEFAULT.w_ref
EUR_PER_UNIT = DEFAULT.eur_per_unit


def eur(x_model):
    """Convert model units to euros."""
    return DEFAULT.eur(x_model)


def model_units(x_eur):
    """Convert euros to model units."""
    return DEFAULT.model_units(x_eur)


RHO_UI_PRE, RHO_UI_POST = DEFAULT.rr_ui_pre, DEFAULT.rr_ui_post
RHO_HA_PRE = DEFAULT.ha_pre_eur / EUR_AVG_WAGE
RHO_HA_POST = DEFAULT.ha_post_eur / EUR_AVG_WAGE

# Statutory Orpo-reform factors, to be applied to *estimated* pre-reform
# benefit levels so the post-reform counterparts follow the same data:
# - UI: staggering (100/80/75 % of the allowance over the spell, averaged
#   and blended with non-staggered basic-security recipients) plus the
#   abolition of child supplements  =>  ~ -16 %
# - HA: compensation rate 80 -> 70 % and basic deductible 42 -> 50 %
#   (>= 12.5 % cut for all recipients, ~ 17 % for unemployed households)
UI_REFORM_FACTOR = RHO_UI_POST / RHO_UI_PRE     # ~ 0.844
HA_REFORM_FACTOR = RHO_HA_POST / RHO_HA_PRE     # ~ 0.827

A_TEST_LO = model_units(DEFAULT.atest_lo_eur)
A_TEST_HI = model_units(DEFAULT.atest_hi_eur)

AMIN = -model_units(DEFAULT.borrow_limit_eur)
AMAX = 40.0


def _common(beta=BETA):
    return dict(
        beta=beta, eta=ETA, alpha=ALPHA, delta=DELTA,
        peu=PEU, pue=PUE,
        amin=AMIN, amax=AMAX,
    )


def pre_reform_params(beta=BETA):
    """Benefit system before the Orpo reforms (2023). No asset test."""
    p = _common(beta)
    p.update(
        b_ui=RHO_UI_PRE * W_REF,
        b_ha=RHO_HA_PRE * W_REF,
        phi_a=0.0,
        a_thresh=A_TEST_LO,
    )
    return p


def asset_test_only_params(beta=BETA):
    """Pre-reform benefit levels + the 2025 housing-allowance asset test."""
    p = _common(beta)
    b_ha = RHO_HA_PRE * W_REF
    p.update(
        b_ui=RHO_UI_PRE * W_REF,
        b_ha=b_ha,
        phi_a=b_ha / (A_TEST_HI - A_TEST_LO),   # linear phase-out 10k -> 50k
        a_thresh=A_TEST_LO,
    )
    return p


def orpo_reform_params(beta=BETA):
    """Full Orpo package: UI staggering cut + HA cut + HA asset test."""
    p = _common(beta)
    b_ha = RHO_HA_POST * W_REF
    p.update(
        b_ui=RHO_UI_POST * W_REF,
        b_ha=b_ha,
        phi_a=b_ha / (A_TEST_HI - A_TEST_LO),
        a_thresh=A_TEST_LO,
    )
    return p


def moments_table(mod):
    """Model moments next to their Finnish targets (approximate)."""
    return {
        "K/Y (annual)":            (mod.K / mod.output(), KY_TARGET),
        "unemployment rate":       (mod.labor.unemployment_rate(),
                                    URATE_TARGET),
        "interest rate r":         (mod.r, ALPHA / KY_TARGET - DELTA),
        "wage (model units)":      (mod.w, W_REF),
        "mean wealth (EUR)":       (eur(mod.aggregate_capital() /
                                        mod.dist.sum()), 230_000.0),
        "UI benefit (EUR/month)":  (eur(mod.b_ui) / 12.0
                                    if mod.b_ui is not None else None, None),
        "HA benefit (EUR/month)":  (eur(mod.b_ha) / 12.0, None),
        "tax rate tau":            (mod.tau, None),
    }
