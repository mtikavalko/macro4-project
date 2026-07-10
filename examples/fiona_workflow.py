"""
End-to-end calibration workflow as it would run inside FIONA (Statistics
Finland's remote-access environment): estimate moments from register
microdata, put them into CalibrationTargets, fit the model, and evaluate
the Orpo reform.

Run from the repository root:  python examples/fiona_workflow.py [--fit]

Everything below uses only numpy / scipy / pandas - no compilation and no
internet access required.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiyagari import (AiyagariModel, solve_general_equilibrium,
                      CalibrationTargets, params_from_targets,
                      compare_moments, smm_calibrate)
from aiyagari.welfare import cev_summary


# --------------------------------------------------------------------------
# Step 1 - data moments.
#
# Inside FIONA, replace the fields below with your register-based
# estimates, e.g.:
#   - FOLK employment spells      -> p_eu, p_ue, p_en, p_ne, p_un, p_nu,
#                                    urate, participation
#   - income register panel       -> rho_e, sigma_e (AR(1) of log earnings),
#                                    var_log_earnings
#   - wealth survey (HVT)         -> mean/median wealth, percentiles, Gini
#   - Kela benefit registers      -> replacement rates, HA levels,
#                                    recipiency
#
# Targets can also be kept in a JSON file next to the estimation scripts:
#   t = CalibrationTargets.from_json("targets_estimated.json")
# --------------------------------------------------------------------------
t = CalibrationTargets(
    # example: override with your own estimates here
    # p_eu=..., p_ue=..., rho_e=..., sigma_e=...,
    # mean_wealth_eur=..., wealth_gini=...,
)
t.to_json(Path(__file__).with_name("targets_example.json"))

# --------------------------------------------------------------------------
# Step 2 - solve the pre-reform economy with the rich labor process
# (employed x 5 productivity states, unemployed with earnings-related
# benefits, non-participation) and inspect the fit.
# --------------------------------------------------------------------------
t0 = time.time()
pre = AiyagariModel(**params_from_targets(t, regime="pre"))
solve_general_equilibrium(pre, verbose=False)
print(f"pre-reform equilibrium solved in {time.time()-t0:.1f}s "
      f"({pre.Nz} labor states)\n")
print(compare_moments(pre, t).to_string(index=False))

# --------------------------------------------------------------------------
# Step 3 (optional, --fit) - fit free parameters to the data moments by
# SMM.  beta is the obvious first candidate (it moves K/Y and the wealth
# level); add e.g. sigma_e against wealth_gini or var_log_earnings once
# you have register estimates.
# --------------------------------------------------------------------------
if "--fit" in sys.argv:
    print("\nfitting beta to K/Y ...")
    t, pre, res = smm_calibrate(
        t, param_names=["beta"], bounds=[(0.90, 0.955)],
        moment_targets={"K/Y": t.ky},
        regime="pre", maxiter=15,
    )
    print(f"fitted beta = {t.beta:.5f} "
          f"(K/Y = {pre.K/pre.output():.4f}, target {t.ky})")
    print(compare_moments(pre, t).to_string(index=False))

# --------------------------------------------------------------------------
# Step 4 - the Orpo reform under the fitted calibration.
# --------------------------------------------------------------------------
t0 = time.time()
orpo = AiyagariModel(**params_from_targets(t, regime="orpo"))
orpo.K = pre.K                       # warm start at the baseline equilibrium
solve_general_equilibrium(orpo, verbose=False)
print(f"\nreform equilibrium solved in {time.time()-t0:.1f}s")

print(f"\ntau: {pre.tau:.4f} -> {orpo.tau:.4f}")
cev = cev_summary(pre, orpo, "full Orpo reform")
for k, v in cev.items():
    if k != "regime":
        print(f"  {k:<22s} {v: .3f}")
