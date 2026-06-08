import numpy as np
from numba import njit, prange


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
def _transfer(a, b, phi_a, a_thresh):
    excess = a - a_thresh
    if excess < 0.0:
        excess = 0.0
    out = b - phi_a * excess
    return out if out > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def _action_value_cont(a_prime, cih, iz, v, a_grid, beta, eta, pi):
    c = cih - a_prime
    if c <= 0.0:
        return -1e30
    fv = pi[iz, 0] * _interp_1d(a_grid, v[:, 0], a_prime) + \
         pi[iz, 1] * _interp_1d(a_grid, v[:, 1], a_prime)
    return (c ** (1.0 - eta) - 1.0) / (1.0 - eta) + beta * fv


@njit(cache=True, fastmath=True)
def _gss(a, b, cih, iz, v, a_grid, beta, eta, pi, tol=1e-8, max_iter=500):
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _action_value_cont(c, cih, iz, v, a_grid, beta, eta, pi)
    fd = _action_value_cont(d, cih, iz, v, a_grid, beta, eta, pi)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _action_value_cont(d, cih, iz, v, a_grid, beta, eta, pi)
        else:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _action_value_cont(c, cih, iz, v, a_grid, beta, eta, pi)
    mid = (a + b) / 2.0
    return mid, _action_value_cont(mid, cih, iz, v, a_grid, beta, eta, pi)


@njit(cache=True, fastmath=True)
def _action_value_disc(ia_prime, cih, iz, v, a_grid, beta, eta, pi):
    c = cih - a_grid[ia_prime]
    if c <= 0.0:
        return -1e30
    fv = pi[iz, 0] * v[ia_prime, 0] + pi[iz, 1] * v[ia_prime, 1]
    return (c ** (1.0 - eta) - 1.0) / (1.0 - eta) + beta * fv


@njit(cache=True, fastmath=True, parallel=True)
def bellman_operator(Tv, v, a_pol, tau, r, w, b, a_grid, beta, eta, pi,
                     phi_a, a_thresh, do_opt):
    Na, Nz = v.shape
    for idx in prange(Na * Nz):
        ia = idx % Na
        iz = idx // Na
        a = a_grid[ia]
        if iz == 0:
            cih = (1.0 + (1.0 - tau) * r) * a + (1.0 - tau) * w
        else:
            cih = (1.0 + (1.0 - tau) * r) * a + _transfer(a, b, phi_a, a_thresh)

        if do_opt:
            lo, hi = 0, Na - 1
            while hi - lo > 2:
                mid = (lo + hi) // 2
                if _action_value_disc(mid + 1, cih, iz, v, a_grid, beta, eta, pi) > \
                   _action_value_disc(mid, cih, iz, v, a_grid, beta, eta, pi):
                    lo = mid
                else:
                    hi = mid + 1
            a_lo, a_hi = a_grid[lo], a_grid[hi]
            if a_hi <= a_lo:
                a_star = a_lo
                val = _action_value_cont(a_star, cih, iz, v, a_grid, beta, eta, pi)
            else:
                a_star, val = _gss(a_lo, a_hi, cih, iz, v, a_grid, beta, eta, pi)
        else:
            a_star = a_pol[ia, iz]
            val = _action_value_cont(a_star, cih, iz, v, a_grid, beta, eta, pi)

        Tv[ia, iz] = val
        a_pol[ia, iz] = a_star


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


@njit(cache=True, fastmath=True)
def transfer_vec(a_grid, b, phi_a, a_thresh):
    """Vectorized transfer for the full distribution grid."""
    out = np.empty(len(a_grid))
    for i in range(len(a_grid)):
        out[i] = _transfer(a_grid[i], b, phi_a, a_thresh)
    return out


def stationary_markov(pi):
    w, V = np.linalg.eig(pi.T)
    v = V[:, np.isclose(w, 1.0)].real
    if v.sum() < 0:
        v = -v
    return (v / v.sum())[:, 0]


class AiyagariModel:
    def __init__(
        self,
        beta: float = 0.95,
        eta: float = 2.0,
        delta: float = 0.04,
        alpha: float = 0.36,
        b: float = 0.1,
        tau0: float = 0.02,
        K0: float = 30.0,
        peu: float = 0.0435,
        pue: float = 0.5,
        amin: float = -2.0,
        amax: float = 30.0,
        num_a: int = 2000,
        num_a_dist_factor: int = 3,
        phi_a: float = 0.0,
        a_thresh: float = 1.0,
    ):
        self.beta = beta
        self.eta = eta
        self.delta = delta
        self.alpha = alpha
        self.b = b
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
        self.a_grid = np.linspace(amin, amax, num_a)
        self.a_grid_dist = np.linspace(amin, amax, num_a * num_a_dist_factor)

        self.v = None
        self.K = K0
        self.update_prices(K0)

    def update_prices(self, K):
        L = self.pi_stat[0]
        kl = K / L
        self.r = self.alpha * kl ** (self.alpha - 1.0) - self.delta
        self.w = (1.0 - self.alpha) * kl ** self.alpha

    def aggregate_capital(self):
        return float(np.dot(self.dist[:, 0] + self.dist[:, 1], self.a_grid_dist))

    def aggregate_transfer(self):
        tr = transfer_vec(self.a_grid_dist, self.b, self.phi_a, self.a_thresh)
        return float(np.dot(self.dist[:, 1], tr))

    def implied_tau(self):
        costs = self.aggregate_transfer()
        denom = self.r * self.K + self.w * self.pi_stat[0]
        return costs / denom if denom > 0.0 else 0.0
