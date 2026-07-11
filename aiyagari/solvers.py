import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .model import egm_solve


def solve_household(mod, tol=1e-10, max_iter=8000):
    """EGM iteration on the consumption policy; also sets a_pol."""
    x, dx = mod.cash_on_hand(mod.a_grid)
    if mod.c_pol is None or mod.c_pol.shape != x.shape:
        # start from consuming a fixed fraction of cash on hand
        mod.c_pol = np.maximum(0.3 * x, 1e-6)
    egm_solve(mod.c_pol, mod.a_grid, x, dx, mod.pi, mod.beta, mod.eta,
              tol, max_iter)
    mod.a_pol = np.clip(x - mod.c_pol, mod.amin, mod.amax)
    # keep the budget identity exact where the grid bounds bind, so the
    # value function's u(c) is consistent with the a' transitions
    mod.c_pol = np.maximum(x - mod.a_pol, 1e-12)


def _lottery(grid, values):
    """Young-lottery indices and weights of `values` on `grid` (per column)."""
    g = np.clip(values, grid[0], grid[-1])
    idx = np.clip(np.searchsorted(grid, g, side="right"), 1, len(grid) - 1)
    lo = idx - 1
    span = grid[idx] - grid[lo]
    t = np.clip((g - grid[lo]) / span, 0.0, 1.0)
    return lo, idx, t


def _transition_matrix(mod, grid, policy):
    """
    Sparse column-stochastic transition T on `grid` x labor states, flat
    index n = z * len(grid) + j, induced by the savings policy and the
    labor Markov chain.
    """
    M = len(grid)
    Nz = mod.Nz
    lo, hi, t = _lottery(grid, policy)
    n = M * Nz
    js = np.arange(M)
    rows, cols, vals = [], [], []
    for z in range(Nz):
        src = z * M + js
        for z2 in range(Nz):
            p = mod.pi[z, z2]
            if p == 0.0:
                continue
            rows.append(z2 * M + lo[:, z]); cols.append(src)
            vals.append(p * (1.0 - t[:, z]))
            rows.append(z2 * M + hi[:, z]); cols.append(src)
            vals.append(p * t[:, z])
    return sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n))


def solve_value_function(mod):
    """
    Value of the converged policy by direct sparse solve of
    (I - beta * B) V = u(c), where B interpolates V at the savings choice.
    """
    Na, Nz = mod.num_a, mod.Nz
    n = Na * Nz
    if mod.eta == 1.0:
        u = np.log(mod.c_pol)
    else:
        u = (mod.c_pol ** (1.0 - mod.eta) - 1.0) / (1.0 - mod.eta)

    B = _transition_matrix(mod, mod.a_grid, mod.a_pol).T  # row = today
    A = sp.identity(n, format="csc") - mod.beta * B.tocsc()
    V = spla.spsolve(A, u.T.reshape(n))
    mod.v = V.reshape(Nz, Na).T.copy()


def solve_distribution(mod, tol=1e-11, max_iter=20000):
    """
    Stationary distribution of the policy-induced Markov chain.

    Two verified methods, tried in size-dependent order, with power
    iteration as the last resort:

    - "Grounded" sparse LU of (T - I) with mu[j0] = 1 pinned at a
      reference node and the redundant equation dropped.  Very fast for
      few labor states, but LU fill-in explodes as Nz grows.
    - ARPACK (scipy.sparse.linalg.eigs) for the Perron eigenvector,
      warm-started from the previous distribution.  Scales to many labor
      states, but stalls when slow-mixing wealth dynamics (e.g. under the
      asset test) push the second eigenvalue extremely close to one.

    Every candidate solution is accepted only if its stationarity
    residual max|T mu - mu| is tiny, so a failing method falls through.
    """
    M = len(mod.a_grid_dist)
    Nz = mod.Nz
    n = M * Nz
    g_fine = np.column_stack([
        np.interp(mod.a_grid_dist, mod.a_grid, mod.a_pol[:, z])
        for z in range(Nz)])
    T = _transition_matrix(mod, mod.a_grid_dist, g_fine)

    def accept(mu):
        if mu is None or not np.all(np.isfinite(mu)):
            return False
        if np.max(np.abs(T @ mu - mu)) >= 1e-8:
            return False
        mod.dist = np.ascontiguousarray(mu.reshape(Nz, M).T)
        return True

    def try_eigs():
        if mod.dist is not None and mod.dist.shape == (M, Nz):
            v0 = np.clip(mod.dist.T.reshape(n), 1e-16, None)
        else:
            v0 = np.full(n, 1.0 / n)
        try:
            w, V = spla.eigs(T, k=1, which="LM", v0=v0,
                             ncv=min(30, n - 1), tol=1e-12, maxiter=30)
            mu = V[:, 0].real
            if mu.sum() < 0:
                mu = -mu
            mu = np.clip(mu, 0.0, None)
            s = mu.sum()
            if s > 0 and abs(w[0].real - 1.0) < 1e-6:
                return accept(mu / s)
        except (spla.ArpackNoConvergence, spla.ArpackError):
            pass
        return False

    def try_grounded():
        A_full = (T - sp.identity(n, format="csc")).tocsr()

        def grounded_solve(j0):
            keep = np.ones(n, dtype=bool)
            keep[j0] = False
            B = A_full[keep][:, keep]
            rhs = -A_full[keep][:, j0].toarray().ravel()
            with np.errstate(all="ignore"):
                x = spla.spsolve(B.tocsc(), rhs)
            if not np.all(np.isfinite(x)):
                return None
            mu = np.insert(x, j0, 1.0)
            mu = np.clip(mu, 0.0, None)
            s = mu.sum()
            return mu / s if s > 0 else None

        candidates = []
        if mod.dist is not None and mod.dist.shape == (M, Nz):
            candidates.append(int(np.argmax(mod.dist.T.reshape(n))))
        iz_mode = int(np.argmax(mod.pi_stat))
        candidates += [
            iz_mode * M + int(np.argmin(np.abs(mod.a_grid_dist - mod.K))),
            iz_mode * M + M - 2,
            iz_mode * M + 1]
        return any(accept(grounded_solve(j0)) for j0 in candidates)

    # LU fill-in is cheap for few labor states; Krylov scales to many
    methods = (try_grounded, try_eigs) if n < 8000 else (try_eigs, try_grounded)
    for method in methods:
        if method():
            return

    # --- last resort: power iteration -------------------------------------
    mu = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        mu_new = T @ mu
        if np.max(np.abs(mu_new - mu)) < tol:
            mu = mu_new
            break
        mu = mu_new
    mu = np.clip(mu, 0.0, None)
    mod.dist = np.ascontiguousarray((mu / mu.sum()).reshape(Nz, M).T)


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
