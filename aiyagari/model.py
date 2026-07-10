import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _interp_1d(grid, vals, x):
    if x <= grid[0]:
        return vals[0]
    if x >= grid[-1]:
        return vals[-1]
    lo, hi = 0, len(grid) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] < x:
            lo = mid
        else:
            hi = mid
    t = (x - grid[lo]) / (grid[hi] - grid[lo])
    return vals[lo] * (1.0 - t) + vals[hi] * t


@njit(cache=True, fastmath=True)
def egm_solve(c_pol, a_grid, x, dx, pi, beta, eta, tol, max_iter):
    """
    Endogenous-grid-method iteration on the consumption policy.

    c_pol : (Na, 2) consumption on the exogenous asset grid, updated in place.
    x     : (Na, 2) cash on hand x_z(a) = R a + inc_z(a).
    dx    : (Na, 2) d x_z / d a (marginal return to assets, enters the Euler
            equation because the unemployment transfer depends on assets).
    """
    Na = a_grid.shape[0]
    c_endo = np.empty(Na)
    x_endo = np.empty(Na)
    c_new = np.empty_like(c_pol)
    it = 0
    for it in range(max_iter):
        for z in range(2):
            for j in range(Na):
                m = 0.0
                for z2 in range(2):
                    m += pi[z, z2] * dx[j, z2] * c_pol[j, z2] ** (-eta)
                c_endo[j] = (beta * m) ** (-1.0 / eta)
                x_endo[j] = c_endo[j] + a_grid[j]
            # guard against tiny non-monotonicities at the asset-test kinks
            for j in range(1, Na):
                if x_endo[j] <= x_endo[j - 1]:
                    x_endo[j] = x_endo[j - 1] + 1e-12
            for i in range(Na):
                xi = x[i, z]
                if xi <= x_endo[0]:
                    c = xi - a_grid[0]          # borrowing constraint binds
                elif xi >= x_endo[Na - 1]:      # extrapolate last segment
                    slope = (c_endo[Na - 1] - c_endo[Na - 2]) / \
                            (x_endo[Na - 1] - x_endo[Na - 2])
                    c = c_endo[Na - 1] + slope * (xi - x_endo[Na - 1])
                else:
                    c = _interp_1d(x_endo, c_endo, xi)
                c_new[i, z] = c if c > 1e-12 else 1e-12
        diff = 0.0
        for i in range(Na):
            for z in range(2):
                d = abs(c_new[i, z] - c_pol[i, z])
                if d > diff:
                    diff = d
                c_pol[i, z] = c_new[i, z]
        if diff < tol:
            break
    return it


@njit(cache=True, fastmath=True)
def _bisect_right(grid, x):
    lo, hi = 0, len(grid)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < grid[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True, fastmath=True)
def build_transition_maps(a_coarse, a_fine, g_coarse, lo, hi, w_lo, w_hi):
    M, N = len(a_fine), len(a_coarse)
    a_min_f, a_max_f = a_fine[0], a_fine[-1]
    for j in range(M):
        for iz in range(2):
            af = a_fine[j]
            idx = _bisect_right(a_coarse, af)
            if idx == 0:
                i, t = 0, 0.0
            elif idx >= N:
                i, t = N - 2, 1.0
            else:
                i = idx - 1
                span = a_coarse[i + 1] - a_coarse[i]
                t = (af - a_coarse[i]) / span if span > 0.0 else 0.0
            g = g_coarse[i, iz] * (1.0 - t) + g_coarse[i + 1, iz] * t
            if g <= a_min_f:
                k, u = 0, 0.0
            elif g >= a_max_f:
                k, u = M - 2, 1.0
            else:
                jdx = _bisect_right(a_fine, g)
                if jdx <= 0:
                    k, u = 0, 0.0
                elif jdx >= M:
                    k, u = M - 2, 1.0
                else:
                    k = jdx - 1
                    span = a_fine[k + 1] - a_fine[k]
                    u = (g - a_fine[k]) / span if span > 0.0 else 0.0
            lo[j, iz] = k
            hi[j, iz] = k + 1
            w_lo[j, iz] = 1.0 - u
            w_hi[j, iz] = u


@njit(cache=True, fastmath=True)
def markov_operator(Tdist, dist, lo, hi, w_lo, w_hi, pi):
    M = dist.shape[0]
    for j in range(M):
        for iz in range(2):
            mass = dist[j, iz]
            if mass == 0.0:
                continue
            k, kp1 = lo[j, iz], hi[j, iz]
            wl, wh = w_lo[j, iz], w_hi[j, iz]
            for iz2 in range(2):
                p = mass * pi[iz, iz2]
                Tdist[k,   iz2] += p * wl
                Tdist[kp1, iz2] += p * wh


def stationary_markov(pi):
    w, V = np.linalg.eig(pi.T)
    v = V[:, np.isclose(w, 1.0)].real
    if v.sum() < 0:
        v = -v
    return (v / v.sum())[:, 0]


def _power_grid(amin, amax, n, curv):
    t = np.linspace(0.0, 1.0, n)
    return amin + (amax - amin) * t ** curv


class AiyagariModel:
    """
    Aiyagari economy with two employment states.  The unemployed receive a
    universal unemployment-insurance transfer b_ui plus an asset-tested
    housing-allowance component:

        B(a) = b_ui + max(0, b_ha - phi_a * max(a - a_thresh, 0))

    A proportional tax tau on capital and labour income balances the budget.
    """

    def __init__(
        self,
        beta: float = 0.9384,
        eta: float = 2.0,
        delta: float = 0.06,
        alpha: float = 0.38,
        b_ui: float = 0.58,
        b_ha: float = 0.10,
        phi_a: float = 0.0,
        a_thresh: float = 0.27,
        tau0: float = 0.012,
        K0: float = 6.0,
        peu: float = 0.0642,
        pue: float = 0.70,
        amin: float = -0.32,
        amax: float = 40.0,
        num_a: int = 400,
        num_a_dist: int = 1600,
        grid_curv: float = 3.0,
    ):
        self.beta = beta
        self.eta = eta
        self.delta = delta
        self.alpha = alpha
        self.b_ui = b_ui
        self.b_ha = b_ha
        self.phi_a = phi_a
        self.a_thresh = a_thresh
        self.tau = tau0
        self.peu = peu
        self.pue = pue

        self.pi = np.array([[1.0 - peu, peu],
                            [pue,       1.0 - pue]])
        self.pi_stat = stationary_markov(self.pi)

        self.amin = amin
        self.amax = amax
        self.num_a = num_a
        self.a_grid = _power_grid(amin, amax, num_a, grid_curv)
        self.a_grid_dist = _power_grid(amin, amax, num_a_dist, grid_curv)

        self.c_pol = None
        self.a_pol = None
        self.v = None
        self.dist = None
        self.K = K0
        self.update_prices(K0)

    # welfare.py historically used .dist_grid
    @property
    def dist_grid(self):
        return self.a_grid_dist

    def update_prices(self, K):
        L = self.pi_stat[0]
        kl = K / L
        self.r = self.alpha * kl ** (self.alpha - 1.0) - self.delta
        self.w = (1.0 - self.alpha) * kl ** self.alpha

    def transfer_profile(self, a_grid=None, b_ha=None):
        """Total transfer B(a) received by an unemployed household."""
        a = self.a_grid_dist if a_grid is None else a_grid
        bh = self.b_ha if b_ha is None else b_ha
        ha = np.clip(bh - self.phi_a * np.maximum(a - self.a_thresh, 0.0),
                     0.0, None)
        return self.b_ui + ha

    def cash_on_hand(self, a_grid):
        """x_z(a) and d x_z / d a on a grid, for the current prices."""
        R = 1.0 + (1.0 - self.tau) * self.r
        n = len(a_grid)
        x = np.empty((n, 2))
        dx = np.empty((n, 2))
        x[:, 0] = R * a_grid + (1.0 - self.tau) * self.w
        dx[:, 0] = R
        x[:, 1] = R * a_grid + self.transfer_profile(a_grid)
        dx[:, 1] = R
        if self.phi_a > 0.0 and self.b_ha > 0.0:
            taper_end = self.a_thresh + self.b_ha / self.phi_a
            in_taper = (a_grid > self.a_thresh) & (a_grid < taper_end)
            dx[in_taper, 1] = R - self.phi_a
        return x, dx

    def aggregate_capital(self):
        return float(np.dot(self.dist[:, 0] + self.dist[:, 1],
                            self.a_grid_dist))

    def aggregate_transfer(self):
        tr = self.transfer_profile()
        return float(np.dot(self.dist[:, 1], tr))

    def implied_tau(self):
        costs = self.aggregate_transfer()
        denom = self.r * self.K + self.w * self.pi_stat[0]
        return costs / denom if denom > 0.0 else 0.0

    def output(self):
        L = self.pi_stat[0]
        return self.K ** self.alpha * L ** (1.0 - self.alpha)
