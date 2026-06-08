import numpy as np
from .model import transfer_vec


def summarize_equilibrium(mod, label):
    tr = transfer_vec(mod.a_grid_dist, mod.b, mod.phi_a, mod.a_thresh)
    tr_full = transfer_vec(mod.a_grid_dist, mod.b, 0.0, mod.a_thresh)

    mu_u = mod.dist[:, 1]
    benefit_costs = float(np.dot(mu_u, tr))
    share_receiving = float(np.sum(mu_u[tr > 1e-12]))

    bite_mask = (tr_full - tr) > 1e-12
    asset_test_bite_mass = float(np.sum(mu_u[bite_mask]))
    average_asset_test_loss = float(np.dot(mu_u[bite_mask], (tr_full - tr)[bite_mask]))

    return {
        "regime": label,
        "K": mod.K,
        "r": mod.r,
        "w": mod.w,
        "tau": mod.tau,
        "N": mod.pi_stat[0],
        "benefit_costs": benefit_costs,
        "mass_borrowing": float(mod.dist[0, :].sum()),
        "share_receiving_benefit": share_receiving,
        "asset_test_bite_mass": asset_test_bite_mass,
        "average_asset_test_loss": average_asset_test_loss,
        "mass_normalization": float(mod.dist.sum()),
        "market_clearing_residual": mod.aggregate_capital() - mod.K,
    }


def _interp_v(mod, grid):
    out = np.empty((len(grid), 2))
    out[:, 0] = np.interp(grid, mod.a_grid, mod.v[:, 0])
    out[:, 1] = np.interp(grid, mod.a_grid, mod.v[:, 1])
    return out


def _v_to_consumption_scale(V, beta, eta):
    """Convert value function to consumption-equivalent scale."""
    return (1.0 - eta) * V + 1.0 / (1.0 - beta)


def compute_cev(baseline, policy_mod):
    """
    Consumption-equivalent variation of policy_mod relative to baseline.

    Returns a (M, 2) array of CEV values on baseline.dist_grid,
    weighted by baseline.dist so that np.average(cev, weights=baseline.dist)
    gives the population mean.
    """
    grid = baseline.dist_grid
    V_base   = _interp_v(baseline,   grid)
    V_policy = _interp_v(policy_mod, grid)

    A_base   = _v_to_consumption_scale(V_base,   baseline.beta, baseline.eta)
    A_policy = _v_to_consumption_scale(V_policy, baseline.beta, baseline.eta)

    ratio = A_policy / A_base
    cev = ratio ** (1.0 / (1.0 - baseline.eta)) - 1.0
    return cev


def cev_summary(baseline, policy_mod, label, a_thresh_split=None):
    cev = compute_cev(baseline, policy_mod)
    w   = baseline.dist                          # shape (M, 2)
    w_flat = w.flatten()
    cev_flat = cev.flatten()
    total_mass = w_flat.sum()

    thresh = a_thresh_split if a_thresh_split is not None else baseline.a_thresh
    low_mask  = baseline.dist_grid <= thresh
    high_mask = ~low_mask

    def wavg(vals, mask=None, iz=None):
        if iz is not None:
            wi = w[:, iz]
            vi = vals[:, iz]
        else:
            wi = w_flat
            vi = cev_flat
        if mask is not None:
            if iz is not None:
                wi, vi = wi[mask], vi[mask]
            else:
                wi, vi = wi[mask.repeat(2)], vi[mask.repeat(2)]
        s = wi.sum()
        return float(np.dot(wi, vi) / s) if s > 0 else float("nan")

    # Distribution-weighted quantiles
    sort_idx = np.argsort(cev_flat)
    sorted_w = w_flat[sort_idx]
    sorted_cev = cev_flat[sort_idx]
    cum_w = np.cumsum(sorted_w) / total_mass

    def wquantile(q):
        idx = np.searchsorted(cum_w, q)
        return float(sorted_cev[min(idx, len(sorted_cev) - 1)])

    return {
        "regime": label,
        "cev_mean_pct":        100.0 * wavg(cev, iz=None),
        "cev_employed_pct":    100.0 * wavg(cev, iz=0),
        "cev_unemployed_pct":  100.0 * wavg(cev, iz=1),
        "cev_low_asset_pct":   100.0 * wavg(cev[low_mask],  iz=None),
        "cev_high_asset_pct":  100.0 * wavg(cev[high_mask], iz=None),
        "cev_p10_pct":  100.0 * wquantile(0.10),
        "cev_p50_pct":  100.0 * wquantile(0.50),
        "cev_p90_pct":  100.0 * wquantile(0.90),
    }


def stationary_welfare(mod):
    V = _interp_v(mod, mod.dist_grid)
    return float(np.sum(mod.dist * V))


def policy_diagnostics(mod):
    return {
        "monotone_employed":   bool(np.all(np.diff(mod.a_pol[:, 0]) >= -1e-8)),
        "monotone_unemployed": bool(np.all(np.diff(mod.a_pol[:, 1]) >= -1e-8)),
        "upper_grid_mass":     float(mod.dist[-1, :].sum()),
        "lower_grid_mass":     float(mod.dist[0,  :].sum()),
    }
