import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .model import egm_solve, build_transition_maps, markov_operator


def solve_household(mod, tol=1e-10, max_iter=8000):
    """EGM iteration on the consumption policy; also sets a_pol."""
    x, dx = mod.cash_on_hand(mod.a_grid)
    if mod.c_pol is None:
        # start from consuming a fixed fraction of cash on hand
        mod.c_pol = np.maximum(0.3 * x, 1e-6)
    egm_solve(mod.c_pol, mod.a_grid, x, dx, mod.pi, mod.beta, mod.eta,
              tol, max_iter)
    mod.a_pol = np.clip(x - mod.c_pol, mod.amin, mod.amax)


def solve_value_function(mod):
    """
    Value of the converged policy, by direct solution of the linear system
    V = u(c) + beta * Pi (x) P(a') V  (interpolation weights from a_pol).
    Exact policy evaluation - no iteration needed.
    """
    Na = mod.num_a
    n = 2 * Na
    u = (mod.c_pol ** (1.0 - mod.eta) - 1.0) / (1.0 - mod.eta)

    idx = np.searchsorted(mod.a_grid, mod.a_pol, side="right")
    idx = np.clip(idx, 1, Na - 1)
    lo = idx - 1
    span = mod.a_grid[idx] - mod.a_grid[lo]
    t = np.clip((mod.a_pol - mod.a_grid[lo]) / span, 0.0, 1.0)

    A = np.zeros((n, n))
    rows = np.arange(Na)
    for z in range(2):
        rn = z * Na + rows
        for z2 in range(2):
            coef = mod.beta * mod.pi[z, z2]
            np.add.at(A, (rn, z2 * Na + lo[:, z]), coef * (1.0 - t[:, z]))
            np.add.at(A, (rn, z2 * Na + idx[:, z]), coef * t[:, z])

    V = np.linalg.solve(np.eye(n) - A, u.T.reshape(n))
    mod.v = V.reshape(2, Na).T.copy()


def solve_distribution(mod, tol=1e-11, max_iter=20000):
    """
    Stationary distribution by direct sparse solve of (T - I) mu = 0 with
    sum(mu) = 1.  The transition matrix T has 4 nonzeros per column (two
    asset lottery points x two employment states), so the sparse LU is
    orders of magnitude faster than power iteration, whose mixing time
    explodes when the asset test creates slow-moving wealth regions.
    """
    M = len(mod.a_grid_dist)
    lo = np.empty((M, 2), dtype=np.int32)
    hi = np.empty((M, 2), dtype=np.int32)
    w_lo = np.empty((M, 2))
    w_hi = np.empty((M, 2))
    build_transition_maps(mod.a_grid, mod.a_grid_dist, mod.a_pol,
                          lo, hi, w_lo, w_hi)

    # source node n = iz*M + j sends pi[iz, iz2]*w to (lo/hi[j, iz], iz2)
    n = 2 * M
    src = np.empty(4 * n, dtype=np.int64)
    dst = np.empty(4 * n, dtype=np.int64)
    val = np.empty(4 * n)
    k = 0
    cols = np.arange(M)
    for iz in range(2):
        for iz2 in range(2):
            p = mod.pi[iz, iz2]
            for tgt, wgt in ((lo[:, iz], w_lo[:, iz]),
                             (hi[:, iz], w_hi[:, iz])):
                src[k:k + M] = iz * M + cols
                dst[k:k + M] = iz2 * M + tgt
                val[k:k + M] = p * wgt
                k += M

    # Solve the "grounded" system: (T - I) mu = 0 has one redundant equation
    # (columns of T sum to one), so pin mu[j0] = 1 at a reference node j0,
    # drop equation j0, and solve for the rest.  This keeps the matrix
    # fully sparse - a dense normalisation row would cause severe LU
    # fill-in.  The choice of j0 only works if the true stationary
    # distribution puts mass on it, so verify the stationarity residual and
    # walk through fallback candidates (distribution supports move around
    # drastically while the GE loop hunts for K).
    diag = np.arange(n, dtype=np.int64)
    rows = np.concatenate([dst, diag])
    colsA = np.concatenate([src, diag])
    data = np.concatenate([val, -np.ones(n)])

    def grounded_solve(j0):
        on_rhs = colsA == j0
        to_rhs = on_rhs & (rows != j0)
        rhs = np.zeros(n - 1)
        rr = rows[to_rhs]
        np.add.at(rhs, rr - (rr > j0), -data[to_rhs])
        keep = (rows != j0) & ~on_rhs
        r = rows[keep]
        c = colsA[keep]
        A = sp.csc_matrix((data[keep], (r - (r > j0), c - (c > j0))),
                          shape=(n - 1, n - 1))
        with np.errstate(all="ignore"):
            x = spla.spsolve(A, rhs)
        if not np.all(np.isfinite(x)):
            return None
        mu = np.insert(x, j0, 1.0)
        mu = np.clip(mu, 0.0, None)
        s = mu.sum()
        return mu / s if s > 0 else None

    candidates = []
    if mod.dist is not None:
        candidates.append(int(np.argmax(mod.dist.T.reshape(n))))
    candidates += [int(np.argmin(np.abs(mod.a_grid_dist - mod.K))),
                   M - 2, 1]

    Tdist = np.empty((M, 2))
    for j0 in candidates:
        mu = grounded_solve(j0)
        if mu is None:
            continue
        dist = np.ascontiguousarray(mu.reshape(2, M).T)
        Tdist.fill(0.0)
        markov_operator(Tdist, dist, lo, hi, w_lo, w_hi, mod.pi)
        if np.max(np.abs(Tdist - dist)) < 1e-8:
            mod.dist = dist
            return

    # last resort: power iteration (slow but unconditionally convergent)
    dist = np.zeros((M, 2))
    i0 = np.argmin(np.abs(mod.a_grid_dist))
    dist[i0, 0] = mod.pi_stat[0]
    dist[i0, 1] = mod.pi_stat[1]
    Tdist = np.zeros_like(dist)
    for _ in range(max_iter):
        Tdist.fill(0.0)
        markov_operator(Tdist, dist, lo, hi, w_lo, w_hi, mod.pi)
        converged = np.max(np.abs(Tdist - dist)) < tol
        dist, Tdist = Tdist, dist
        if converged:
            break
    mod.dist = dist / dist.sum()


def solve_general_equilibrium(
    mod,
    tol=1e-3,
    max_iter=60,
    hh_tol=1e-10,
    dist_tol=1e-11,
    verbose=True,
):
    """
    Root-find on K (Illinois false position) with an inner fixed point on
    the balanced-budget tau.  The bracket is grown outward from mod.K, so a
    good initial guess (e.g. the pre-reform equilibrium) keeps every
    evaluation in the well-behaved interior region.  Policies and
    distributions are warm-started across all evaluations.
    """

    def solve_at_K(K_try, tau_init, tau_iter=8, tau_tol=1e-6):
        tau = tau_init
        for _ in range(tau_iter):
            mod.K = K_try
            mod.tau = tau
            mod.update_prices(K_try)
            solve_household(mod, tol=hh_tol)
            solve_distribution(mod, tol=dist_tol)
            tau_new = mod.implied_tau()
            done = abs(tau_new - tau) < tau_tol
            tau = tau_new
            if done:
                break
        K_imp = mod.aggregate_capital()
        return K_imp - K_try, tau, K_imp

    # --- expanding bracket search around the initial guess ---------------
    K0 = min(max(mod.K, 0.5), 0.8 * mod.amax)
    tau_cur = mod.tau
    ed0, tau0, _ = solve_at_K(K0, tau_cur)
    if verbose:
        print(f"[bracket] K={K0:9.5f}  ed={ed0: .6f}  tau={tau0:.6f}")
    if abs(ed0) < tol:
        solve_value_function(mod)
        return mod

    step = 1.15
    if ed0 > 0:      # households want to save more -> equilibrium K higher
        K_lo, ed_lo, tau_lo = K0, ed0, tau0
        K_hi = K0
        for _ in range(40):
            K_hi = min(K_hi * step, 0.8 * mod.amax)
            ed_hi, tau_hi, _ = solve_at_K(K_hi, tau_lo)
            if verbose:
                print(f"[bracket] K={K_hi:9.5f}  ed={ed_hi: .6f}")
            if ed_hi < 0:
                break
            K_lo, ed_lo, tau_lo = K_hi, ed_hi, tau_hi
        else:
            raise RuntimeError("Could not bracket equilibrium from above.")
    else:
        K_hi, ed_hi, tau_hi = K0, ed0, tau0
        K_lo = K0
        for _ in range(40):
            K_lo = max(K_lo / step, 0.05)
            ed_lo, tau_lo, _ = solve_at_K(K_lo, tau_hi)
            if verbose:
                print(f"[bracket] K={K_lo:9.5f}  ed={ed_lo: .6f}")
            if ed_lo > 0:
                break
            K_hi, ed_hi, tau_hi = K_lo, ed_lo, tau_hi
        else:
            raise RuntimeError("Could not bracket equilibrium from below.")

    # --- Illinois false-position on the bracket --------------------------
    side = 0
    for it in range(max_iter):
        K_mid = (K_lo * ed_hi - K_hi * ed_lo) / (ed_hi - ed_lo)
        # keep strictly inside the bracket
        gap = K_hi - K_lo
        K_mid = min(max(K_mid, K_lo + 1e-3 * gap), K_hi - 1e-3 * gap)
        ed_mid, tau_mid, K_imp = solve_at_K(K_mid, 0.5 * (tau_lo + tau_hi))
        if verbose:
            print(f"[{it:3d}] K={K_mid:9.5f}  K_imp={K_imp:9.5f}  "
                  f"ed={ed_mid: .6f}  tau={tau_mid:.6f}")
        if abs(ed_mid) < tol:
            break
        if ed_lo * ed_mid < 0:
            K_hi, tau_hi, ed_hi = K_mid, tau_mid, ed_mid
            if side == -1:
                ed_lo *= 0.5
            side = -1
        else:
            K_lo, tau_lo, ed_lo = K_mid, tau_mid, ed_mid
            if side == 1:
                ed_hi *= 0.5
            side = 1

    solve_value_function(mod)
    if verbose:
        print(f"Converged. K={mod.K:.6f}, r={mod.r:.6f}, "
              f"w={mod.w:.6f}, tau={mod.tau:.6f}")
    return mod


def calibrate_beta(make_model, ky_target, beta_lo=0.92, beta_hi=0.955,
                   tol=0.02, max_iter=12, verbose=True):
    """
    Bisect beta so the equilibrium (annual) K/Y hits ky_target.
    make_model(beta) must return a fresh AiyagariModel.
    """
    def ky_gap(beta):
        mod = make_model(beta)
        solve_general_equilibrium(mod, verbose=False)
        ky = mod.K / mod.output()
        if verbose:
            print(f"  beta={beta:.5f}  K/Y={ky:.4f}")
        return ky - ky_target, mod

    g_lo, _ = ky_gap(beta_lo)
    g_hi, _ = ky_gap(beta_hi)
    if g_lo * g_hi > 0:
        raise RuntimeError("K/Y target not bracketed by beta range.")
    for _ in range(max_iter):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        g_mid, mod = ky_gap(beta_mid)
        if abs(g_mid) < tol:
            return beta_mid, mod
        if g_lo * g_mid < 0:
            beta_hi, g_hi = beta_mid, g_mid
        else:
            beta_lo, g_lo = beta_mid, g_mid
    return beta_mid, mod
