"""
Finnish calibration for the Aiyagari asset-tested benefits model.

Model period: one year.  The euro-to-model-unit mapping is anchored on the
average gross wage: the reference model wage ``w_ref`` (implied by the
capital-output target) corresponds to EUR_AVG_WAGE per year.  All policy
quantities (benefit levels, asset-test thresholds, borrowing limit) are
fixed in model units through that mapping.

Macro targets (approximate; sources in the project notebook):
  - Unemployment rate 8.4 % in 2024 (Statistics Finland, Labour Force Survey).
  - Average gross earnings ~ EUR 4,000/month = EUR 48,000/year.
  - Capital-output ratio ~ 3.3 (annual); consumption of fixed capital
    ~ 20 % of GDP  =>  delta = 0.06.
  - Capital share alpha = 0.38 (Finnish labour share ~ 0.62, PWT).
  - Mean job-finding rate: mean unemployment duration ~ 17 months
    (registered unemployment, incl. long-term unemployed) => pue = 0.70/yr.
  - beta is calibrated internally so the equilibrium K/Y hits the target.

Policy calibration - the Orpo government reforms (2024-2025):
  1. Earnings-related allowance staggering (from 2 Sep 2024): 100 % for the
     first 40 benefit days, 80 % after 40 days, 75 % after 170 days.
     Averaged over a ~17-month spell and blended with (non-staggered) basic
     allowance recipients, plus the abolition of child supplements, this cuts
     the average unemployment-insurance transfer by ~15 %.
  2. General housing allowance cut (1 Apr 2024): compensation rate 80 -> 70 %,
     basic deductible 42 -> 50 % of qualifying income (>= 12.5 % cut for all
     recipients; ~17 % assumed here for unemployed households).
  3. General housing allowance asset test (1 Jan 2025): financial assets
     above EUR 10,000 (single household) reduce the allowance and no
     allowance is paid when assets exceed EUR 50,000.  Modelled as a linear
     phase-out of the housing-allowance component between the two thresholds
     (marginal rate ~ 8 %/yr, close to the statutory 20 % asset-to-income
     conversion x 50 % deductible x 70 % compensation ~ 7 %/yr).

Benefit levels (per unemployed household, fractions of the gross avg wage):
  - Unemployment insurance: pre-reform net replacement ~ 45 % of the gross
    average wage (blend of earnings-related ~ 56 % gross and basic allowance
    ~ 20 % gross, net of taxes on benefits, which the model does not levy).
    Post-reform: 45 % x 0.85 ~ 38 %.
  - General housing allowance: ~ EUR 325/month pre-reform for a typical
    unemployed recipient (~ 8.1 % of the gross avg wage), ~ EUR 270/month
    (~ 6.7 %) after the April 2024 cut.
"""

import numpy as np

# ---------------------------------------------------------------- targets
EUR_AVG_WAGE = 48_000.0          # gross average annual earnings, 2024

URATE_TARGET = 0.084             # unemployment rate, 2024
KY_TARGET    = 3.3               # annual capital-output ratio
ALPHA        = 0.38
DELTA        = 0.06
ETA          = 2.0
PUE          = 0.70              # annual job-finding probability
PEU          = URATE_TARGET / (1.0 - URATE_TARGET) * PUE   # => u* = 8.4 %

# beta calibrated with calibrate_beta() so equilibrium K/Y ~ KY_TARGET
# under the pre-reform benefit system; value hard-coded for reproducibility.
BETA = 0.94926

# ------------------------------------------------- euro <-> model units
# Reference wage implied by the K/Y target: w = (1-a) * (K/Y)^(a/(1-a))
W_REF = (1.0 - ALPHA) * KY_TARGET ** (ALPHA / (1.0 - ALPHA))
EUR_PER_UNIT = EUR_AVG_WAGE / W_REF          # euros per model unit


def eur(x_model):
    """Convert model units to euros."""
    return x_model * EUR_PER_UNIT


def model_units(x_eur):
    """Convert euros to model units."""
    return x_eur / EUR_PER_UNIT


# ------------------------------------------------------ benefit levels
RHO_UI_PRE,  RHO_UI_POST = 0.45, 0.38        # UI, fraction of gross avg wage
RHO_HA_PRE,  RHO_HA_POST = 0.081, 0.067      # housing allowance, same units

A_TEST_LO = model_units(10_000.0)            # assets start reducing HA
A_TEST_HI = model_units(50_000.0)            # HA fully gone

AMIN = -model_units(12_000.0)                # unsecured borrowing limit
AMAX = 40.0                                   # ~ EUR 1.5m, top of asset grid


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
    L = mod.pi_stat[0]
    Y = mod.K ** mod.alpha * L ** (1.0 - mod.alpha)
    return {
        "K/Y (annual)":            (mod.K / Y, KY_TARGET),
        "unemployment rate":       (mod.pi_stat[1], URATE_TARGET),
        "interest rate r":         (mod.r, ALPHA / KY_TARGET - DELTA),
        "wage (model units)":      (mod.w, W_REF),
        "mean wealth (EUR)":       (eur(mod.aggregate_capital() /
                                        mod.dist.sum()), 230_000.0),
        "UI benefit (EUR/month)":  (eur(mod.b_ui) / 12.0, None),
        "HA benefit (EUR/month)":  (eur(mod.b_ha) / 12.0, None),
        "tax rate tau":            (mod.tau, None),
    }
