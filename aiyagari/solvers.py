import numpy as np
from .model import bellman_operator, build_transition_maps, markov_operator


def solve_value_function(mod, tol=1e-8, max_iter=5000, howard_every=20):
    v = mod.v if mod.v is not None else np.zeros((mod.num_a, 2))
    a_pol = np.zeros_like(v)
    Tv = np.empty_like(v)

    for it in range(max_iter):
        do_opt = (it % howard_every == 0)
        bellman_operator(
            Tv, v, a_pol,
            mod.tau, mod.r, mod.w, mod.b,
            mod.a_grid, mod.beta, mod.eta, mod.pi,
            mod.phi_a, mod.a_thresh,
            do_opt,
        )
        diff = np.max(np.abs(Tv - v))
        if diff < tol and do_opt:
            break
        v = Tv.copy()

    mod.v = v
    mod.a_pol = a_pol


def solve_distribution(mod, tol=1e-10, max_iter=10_000):
    M = len(mod.a_grid_dist)
    lo  = np.empty((M, 2), dtype=np.int32)
    hi  = np.empty((M, 2), dtype=np.int32)
    w_lo = np.empty((M, 2))
    w_hi = np.empty((M, 2))
    build_transition_maps(mod.a_grid, mod.a_grid_dist, mod.a_pol, lo, hi, w_lo, w_hi)

    dist = np.zeros((M, 2))
    i0 = np.argmin(np.abs(mod.a_grid_dist))
    dist[i0, 0] = mod.pi_stat[0]
    dist[i0, 1] = mod.pi_stat[1]
    dist /= dist.sum()

    Tdist = np.zeros_like(dist)
    for _ in range(max_iter):
        Tdist.fill(0.0)
        markov_operator(Tdist, dist, lo, hi, w_lo, w_hi, mod.pi)
        if np.max(np.abs(Tdist - dist)) < tol:
            break
        dist = Tdist.copy()

    mod.dist = dist


def solve_general_equilibrium(
    mod,
    tol=1e-4,
    max_iter=60,
    vf_tol=1e-8,
    dist_tol=1e-10,
    verbose=True,
):
    """
    Bisection on K, inner fixed-point on tau.

    Finds K* such that K_implied(K*) = K* while simultaneously
    solving for the balanced-budget tax rate tau(K*).
    """

    def excess_demand(K_try, tau_try, warm_v=None):
        mod.K = K_try
        mod.tau = tau_try
        mod.update_prices(K_try)
        if warm_v is not None:
            mod.v = warm_v.copy()
        solve_value_function(mod, tol=vf_tol)
        solve_distribution(mod, tol=dist_tol)
        K_imp = mod.aggregate_capital()
        tau_imp = mod.implied_tau()
        return K_imp - K_try, tau_imp

    # --- inner tau loop given K ------------------------------------------
    def solve_at_K(K_try, tau_init, warm_v=None, tau_iter=30, tau_tol=1e-6):
        tau = tau_init
        for _ in range(tau_iter):
            ed, tau_new = excess_demand(K_try, tau, warm_v=warm_v)
            warm_v = mod.v.copy()
            if abs(tau_new - tau) < tau_tol:
                tau = tau_new
                break
            tau = 0.5 * tau + 0.5 * tau_new
        K_imp = mod.aggregate_capital()
        return K_imp - K_try, tau, K_imp

    # --- bracket K -------------------------------------------------------
    K_lo = mod.amin * mod.pi_stat[1] + 1e-3   # lower bound: nearly no capital
    K_hi = mod.amax * 0.8                      # generous upper bound

    tau_cur = mod.tau

    if verbose:
        print("Bracketing K...")

    ed_lo, tau_lo, _ = solve_at_K(K_lo, tau_cur)
    ed_hi, tau_hi, _ = solve_at_K(K_hi, tau_cur)

    if ed_lo * ed_hi > 0:
        if verbose:
            print(f"Warning: could not bracket (ed_lo={ed_lo:.4f}, ed_hi={ed_hi:.4f}). "
                  f"Falling back to damped iteration.")
        _damped_ge(mod, tol=tol, max_iter=200, vf_tol=vf_tol, dist_tol=dist_tol,
                   verbose=verbose)
        return mod

    # --- bisection -------------------------------------------------------
    for it in range(max_iter):
        K_mid = 0.5 * (K_lo + K_hi)
        tau_mid = 0.5 * (tau_lo + tau_hi)
        ed_mid, tau_mid, K_imp = solve_at_K(K_mid, tau_mid, warm_v=mod.v)

        if verbose:
            print(f"[{it:3d}] K_lo={K_lo:.4f} K_hi={K_hi:.4f} "
                  f"K_mid={K_mid:.4f} K_imp={K_imp:.4f} "
                  f"ed={ed_mid:.6f} tau={tau_mid:.6f}")

        if abs(ed_mid) < tol:
            break

        if ed_lo * ed_mid < 0:
            K_hi, tau_hi = K_mid, tau_mid
            ed_hi = ed_mid
        else:
            K_lo, tau_lo = K_mid, tau_mid
            ed_lo = ed_mid

    if verbose:
        print(f"Converged. K={mod.K:.6f}, r={mod.r:.6f}, w={mod.w:.6f}, tau={mod.tau:.6f}")

    return mod


def _damped_ge(mod, tol, max_iter, vf_tol, dist_tol, verbose, damp=0.05):
    """Fallback damped-iteration GE solver (original approach)."""
    for it in range(max_iter):
        K_old, tau_old = mod.K, mod.tau
        mod.update_prices(K_old)
        solve_value_function(mod, tol=vf_tol)
        solve_distribution(mod, tol=dist_tol)
        K_imp = mod.aggregate_capital()
        tau_imp = mod.implied_tau()
        mod.K   = K_old   + damp * (K_imp - K_old)
        mod.tau = tau_old + damp * (tau_imp - tau_old)
        mod.update_prices(mod.K)
        if verbose:
            print(f"[{it:3d}] K={K_old:.4f} K_imp={K_imp:.4f} "
                  f"tau={tau_old:.6f} tau_imp={tau_imp:.6f}")
        if abs(K_imp - K_old) < tol and abs(tau_imp - tau_old) < tol:
            break
    return mod
