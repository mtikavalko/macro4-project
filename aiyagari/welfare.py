import numpy as np


def summarize_equilibrium(mod, label):
    grid = mod.a_grid_dist
    tr = mod.transfer_matrix(grid)                       # (M, Nz)
    ha_full = np.outer(np.full(len(grid), mod.b_ha),
                       mod.labor.ha_elig.astype(float))
    tr_full = mod.labor.b0[None, :] + ha_full            # no asset test
    ha = tr - mod.labor.b0[None, :]                      # housing component

    benefit_costs = float(np.sum(mod.dist * tr))
    elig_mass = float(mod.dist[:, mod.labor.ha_elig].sum())
    ha_recipient_mass = float(np.sum(mod.dist[ha > 1e-12]))
    share_elig_receiving_ha = (ha_recipient_mass / elig_mass
                               if elig_mass > 0 else float("nan"))

    bite = (tr_full - tr) > 1e-12
    asset_test_bite_mass = float(np.sum(mod.dist[bite]))
    average_asset_test_loss = float(np.sum(mod.dist[bite] *
                                           (tr_full - tr)[bite]))

    Y = mod.output()
    return {
        "regime": label,
        "K": mod.K,
        "K/Y": mod.K / Y,
        "r": mod.r,
        "w": mod.w,
        "tau": mod.tau,
        "N": float(mod.pi_stat[mod.labor.is_employed].sum()),
        "benefit_costs": benefit_costs,
        "benefit_costs/Y": benefit_costs / Y,
        "mass_borrowing": float(np.sum(mod.dist[grid < 0.0, :])),
        "ha_recipient_mass": ha_recipient_mass,
        "share_elig_receiving_ha": share_elig_receiving_ha,
        "ha_eligible_mass": elig_mass,
        "asset_test_bite_mass": asset_test_bite_mass,
        "average_asset_test_loss": average_asset_test_loss,
        "market_clearing_residual": mod.aggregate_capital() - mod.K,
    }


def _interp_v(mod, grid):
    out = np.empty((len(grid), mod.Nz))
    for z in range(mod.Nz):
        out[:, z] = np.interp(grid, mod.a_grid, mod.v[:, z])
    return out


def _v_to_consumption_scale(V, beta, eta):
    """Convert value function to consumption-equivalent scale."""
    return (1.0 - eta) * V + 1.0 / (1.0 - beta)


def compute_cev(baseline, policy_mod):
    """
    Consumption-equivalent variation of policy_mod relative to baseline,
    evaluated on baseline.a_grid_dist and weighted by baseline.dist.
    Requires both models to share beta, eta and the state space.
    """
    grid = baseline.a_grid_dist
    V_base = _interp_v(baseline, grid)
    V_policy = _interp_v(policy_mod, grid)

    A_base = _v_to_consumption_scale(V_base, baseline.beta, baseline.eta)
    A_policy = _v_to_consumption_scale(V_policy, baseline.beta, baseline.eta)

    ratio = A_policy / A_base
    cev = ratio ** (1.0 / (1.0 - baseline.eta)) - 1.0
    return cev


def cev_summary(baseline, policy_mod, label, a_thresh_split=None):
    cev = compute_cev(baseline, policy_mod)
    w = baseline.dist                            # (M, Nz)
    grid = baseline.a_grid_dist
    lab = baseline.labor
    w_flat = w.flatten()
    cev_flat = cev.flatten()
    total_mass = w_flat.sum()

    thresh = a_thresh_split if a_thresh_split is not None else baseline.a_thresh
    low_mask = grid <= thresh

    def wavg(wi, vi):
        s = wi.sum()
        return float(np.dot(wi.flatten(), vi.flatten()) / s) \
            if s > 0 else float("nan")

    sort_idx = np.argsort(cev_flat)
    cum_w = np.cumsum(w_flat[sort_idx]) / total_mass
    sorted_cev = cev_flat[sort_idx]

    def wquantile(q):
        idx = np.searchsorted(cum_w, q)
        return float(sorted_cev[min(idx, len(sorted_cev) - 1)])

    out = {
        "regime": label,
        "cev_mean_pct":       100.0 * wavg(w_flat, cev_flat),
        "cev_employed_pct":   100.0 * wavg(w[:, lab.is_employed],
                                           cev[:, lab.is_employed]),
        "cev_unemployed_pct": 100.0 * wavg(w[:, lab.is_unemp],
                                           cev[:, lab.is_unemp]),
        "cev_low_asset_pct":  100.0 * wavg(w[low_mask], cev[low_mask]),
        "cev_high_asset_pct": 100.0 * wavg(w[~low_mask], cev[~low_mask]),
        "cev_p10_pct": 100.0 * wquantile(0.10),
        "cev_p50_pct": 100.0 * wquantile(0.50),
        "cev_p90_pct": 100.0 * wquantile(0.90),
    }
    if lab.is_inact.any():
        out["cev_inactive_pct"] = 100.0 * wavg(w[:, lab.is_inact],
                                               cev[:, lab.is_inact])
    return out


def stationary_welfare(mod):
    V = _interp_v(mod, mod.a_grid_dist)
    return float(np.sum(mod.dist * V))


def policy_diagnostics(mod):
    mono = np.all(np.diff(mod.a_pol, axis=0) >= -1e-8, axis=0)
    return {
        "monotone_employed":   bool(mono[mod.labor.is_employed].all()),
        "monotone_unemployed": bool(mono[mod.labor.is_unemp].all()),
        "upper_grid_mass":     float(mod.dist[-1, :].sum()),
        "lower_grid_mass":     float(mod.dist[0,  :].sum()),
    }
