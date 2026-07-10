"""
Aiyagari economy with an arbitrary finite labor-state process and an
asset-tested housing-allowance transfer.

Pure numpy/scipy implementation - no compilation, no internet access and
no packages beyond the standard scientific stack are required, so the
package can be copied as-is into restricted research environments such as
Statistics Finland's FIONA remote-access system.
"""

import numpy as np
from .labor import LaborProcess, two_state_process


def _power_grid(amin, amax, n, curv):
    t = np.linspace(0.0, 1.0, n)
    return amin + (amax - amin) * t ** curv


def egm_solve(c_pol, a_grid, x, dx, Pi, beta, eta, tol=1e-10, max_iter=8000):
    """
    Endogenous-grid-method iteration on the consumption policy (in place).

    c_pol : (Na, Nz) consumption on the exogenous asset grid.
    x     : (Na, Nz) cash on hand x_z(a) = R a + inc_z(a).
    dx    : (Na, Nz) marginal return d x_z / d a, which differs from R
            inside the housing-allowance phase-out band and enters the
            Euler equation through the envelope condition.
    """
    Na, Nz = c_pol.shape
    amin = a_grid[0]
    a_col = a_grid[:, None]
    it = 0
    for it in range(max_iter):
        emu = (dx * c_pol ** (-eta)) @ Pi.T          # E[u'(c') dx'] by state
        c_endo = (beta * emu) ** (-1.0 / eta)
        x_endo = c_endo + a_col
        # guard against tiny non-monotonicities at the asset-test kinks
        np.maximum.accumulate(x_endo, axis=0, out=x_endo)

        c_new = np.empty_like(c_pol)
        for z in range(Nz):
            c_new[:, z] = np.interp(x[:, z], x_endo[:, z], c_endo[:, z])
            # borrowing constraint binds below the first endogenous point
            lo = x[:, z] <= x_endo[0, z]
            c_new[lo, z] = x[lo, z] - amin
            # linear extrapolation above the last endogenous point
            hi = x[:, z] >= x_endo[-1, z]
            if hi.any():
                span = x_endo[-1, z] - x_endo[-2, z]
                slope = (c_endo[-1, z] - c_endo[-2, z]) / span if span > 0 else 1.0
                c_new[hi, z] = c_endo[-1, z] + slope * (x[hi, z] - x_endo[-1, z])
        np.clip(c_new, 1e-12, None, out=c_new)

        diff = np.max(np.abs(c_new - c_pol))
        c_pol[:] = c_new
        if diff < tol:
            break
    return it


class AiyagariModel:
    """
    Heterogeneous-agent economy.  Households face the labor-state Markov
    chain in `labor`; state income is

        inc_z(a) = (1 - tau) * w * e_z + b0_z
                   + ha_elig_z * max(0, b_ha - phi_a * max(a - a_thresh, 0))

    and a proportional tax tau on capital and labor income balances the
    transfer budget in equilibrium.

    For backward compatibility the original two-state economy can still be
    constructed directly with (b_ui, peu, pue) instead of `labor`.
    """

    def __init__(
        self,
        beta: float = 0.94926,
        eta: float = 2.0,
        delta: float = 0.06,
        alpha: float = 0.38,
        labor: LaborProcess = None,
        b_ui: float = None,
        b_ha: float = 0.0,
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
        self.b_ha = b_ha
        self.phi_a = phi_a
        self.a_thresh = a_thresh
        self.tau = tau0

        if labor is None:
            if b_ui is None:
                raise ValueError("Provide either `labor` or `b_ui`.")
            labor = two_state_process(peu, pue, b_ui)
        self.labor = labor
        self.b_ui = b_ui
        self.Nz = labor.n
        self.pi = labor.Pi                    # kept under the old names
        self.pi_stat = labor.pi_stat
        self.L = labor.aggregate_labor()

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
        kl = K / self.L
        self.r = self.alpha * kl ** (self.alpha - 1.0) - self.delta
        self.w = (1.0 - self.alpha) * kl ** self.alpha

    def housing_allowance(self, a_grid):
        """Asset-tested HA amount at each asset level (for eligible states)."""
        if self.b_ha == 0.0:
            return np.zeros(len(a_grid))
        return np.clip(self.b_ha - self.phi_a *
                       np.maximum(a_grid - self.a_thresh, 0.0), 0.0, None)

    def transfer_matrix(self, a_grid=None):
        """(n_grid, Nz) transfer received in each state at each asset level."""
        a = self.a_grid_dist if a_grid is None else a_grid
        ha = self.housing_allowance(a)
        return (self.labor.b0[None, :]
                + np.outer(ha, self.labor.ha_elig.astype(float)))

    def transfer_profile(self, a_grid=None, b_ha=None):
        """Total transfer of the first unemployed state (2-state legacy API)."""
        a = self.a_grid_dist if a_grid is None else a_grid
        iz = int(np.argmax(self.labor.is_unemp))
        if b_ha is not None:
            saved, self.b_ha = self.b_ha, b_ha
            out = self.transfer_matrix(a)[:, iz]
            self.b_ha = saved
            return out
        return self.transfer_matrix(a)[:, iz]

    def cash_on_hand(self, a_grid):
        """x_z(a) and d x_z / d a on a grid, at the current prices."""
        R = 1.0 + (1.0 - self.tau) * self.r
        labor_inc = (1.0 - self.tau) * self.w * self.labor.e
        x = R * a_grid[:, None] + labor_inc[None, :] + self.transfer_matrix(a_grid)
        dx = np.full_like(x, R)
        if self.phi_a > 0.0 and self.b_ha > 0.0:
            taper_end = self.a_thresh + self.b_ha / self.phi_a
            in_taper = (a_grid > self.a_thresh) & (a_grid < taper_end)
            dx[np.ix_(in_taper, self.labor.ha_elig)] = R - self.phi_a
        return x, dx

    # ------------------------------------------------------ aggregates
    def aggregate_capital(self):
        return float(self.dist.sum(axis=1) @ self.a_grid_dist)

    def aggregate_transfer(self):
        return float(np.sum(self.dist * self.transfer_matrix()))

    def implied_tau(self):
        costs = self.aggregate_transfer()
        denom = self.r * self.K + self.w * self.L
        return costs / denom if denom > 0.0 else 0.0

    def output(self):
        return self.K ** self.alpha * self.L ** (1.0 - self.alpha)
