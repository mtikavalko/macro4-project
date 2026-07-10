"""
Labor-market process for the Aiyagari model.

A LaborProcess is an arbitrary finite Markov chain over labor states.
Each state carries:
  - e        efficiency units (labor income = (1-tau) * w * e), 0 if not working
  - b0       base transfer in model units (earnings-related UI, basic
             allowance, social assistance, ...), 0 for workers
  - ha_elig  whether the state receives the asset-tested housing allowance
  - is_unemp / is_inact classification flags (for reporting)

The default builder combines a Rouwenhorst-discretized AR(1) for log
labor productivity with employment / unemployment / non-participation
flows, so that transition rates estimated from register data (e.g. FOLK
employment spells in FIONA) map directly into the model:

  states:  E_1..E_ne  (employed at productivity e_i)
           U_1..U_ne  (unemployed, benefits tied to own productivity)
           N          (out of the labor force)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LaborProcess:
    names: list
    Pi: np.ndarray          # (Nz, Nz) row-stochastic transition matrix
    e: np.ndarray           # efficiency units
    b0: np.ndarray          # base transfer, model units
    ha_elig: np.ndarray     # bool
    is_unemp: np.ndarray    # bool
    is_inact: np.ndarray    # bool
    pi_stat: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Pi = np.asarray(self.Pi, dtype=float)
        self.e = np.asarray(self.e, dtype=float)
        self.b0 = np.asarray(self.b0, dtype=float)
        self.ha_elig = np.asarray(self.ha_elig, dtype=bool)
        self.is_unemp = np.asarray(self.is_unemp, dtype=bool)
        self.is_inact = np.asarray(self.is_inact, dtype=bool)
        rowsum = self.Pi.sum(axis=1)
        if not np.allclose(rowsum, 1.0, atol=1e-10):
            raise ValueError(f"Pi rows must sum to 1, got {rowsum}")
        self.pi_stat = stationary_distribution(self.Pi)

    @property
    def n(self):
        return len(self.e)

    @property
    def is_employed(self):
        return self.e > 0.0

    # --- aggregate labor-market rates implied by the chain ---------------
    def participation_rate(self):
        return float(self.pi_stat[~self.is_inact].sum())

    def unemployment_rate(self):
        lf = self.pi_stat[~self.is_inact].sum()
        return float(self.pi_stat[self.is_unemp].sum() / lf) if lf > 0 else 0.0

    def employment_rate(self):
        return float(self.pi_stat[self.is_employed].sum())

    def aggregate_labor(self):
        return float(np.dot(self.pi_stat, self.e))


def stationary_distribution(Pi):
    w, V = np.linalg.eig(Pi.T)
    v = V[:, np.argmin(np.abs(w - 1.0))].real
    if v.sum() < 0:
        v = -v
    v = np.clip(v, 0.0, None)
    return v / v.sum()


def rouwenhorst(n, rho, sigma):
    """
    Rouwenhorst (1995) discretization of an AR(1) with persistence rho and
    *stationary* standard deviation sigma.  Returns (grid, P) where grid is
    in logs, symmetric around zero.
    """
    if n == 1:
        return np.zeros(1), np.ones((1, 1))
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1.0 - p], [1.0 - p, p]])
    for m in range(3, n + 1):
        Z = np.zeros((m, m))
        Z[:m - 1, :m - 1] += p * P
        Z[:m - 1, 1:] += (1.0 - p) * P
        Z[1:, :m - 1] += (1.0 - p) * P
        Z[1:, 1:] += p * P
        Z[1:-1, :] /= 2.0
        P = Z
    span = sigma * np.sqrt(n - 1.0)
    grid = np.linspace(-span, span, n)
    return grid, P


def two_state_process(peu, pue, b_ui, ha_on_unemployed=True):
    """The original 2-state economy: employed (e=1) and unemployed."""
    Pi = np.array([[1.0 - peu, peu],
                   [pue, 1.0 - pue]])
    return LaborProcess(
        names=["E", "U"],
        Pi=Pi,
        e=[1.0, 0.0],
        b0=[0.0, b_ui],
        ha_elig=[False, ha_on_unemployed],
        is_unemp=[False, True],
        is_inact=[False, False],
    )


def build_labor_process(
    n_e=5,
    rho_e=0.96,
    sigma_e=0.45,
    p_eu=0.055,          # E -> U (job separation)
    p_ue=0.70,           # U -> E (job finding)
    p_en=0.02,           # E -> N (labor-force exit)
    p_ne=0.10,           # N -> E (labor-force entry into employment)
    p_un=0.10,           # U -> N (discouragement / benefit exhaustion)
    p_nu=0.00,           # N -> U (entry via unemployment)
    rr_ui=0.45,          # UI replacement rate on own gross wage
    ui_floor=0.0,        # minimum UI transfer (basic allowance), model units
    ui_cap=np.inf,       # cap on the UI transfer, model units
    b_n=0.0,             # transfer in non-participation, model units
    w_anchor=1.0,        # wage used to convert replacement rates to levels
    ha_unemployed=True,
    ha_inactive=True,
):
    """
    Build the (2*n_e + 1)-state process E_1..E_ne, U_1..U_ne, N.

    Productivity follows the Rouwenhorst chain in E and U (frozen skills
    would set rho close to 1); the unemployed state remembers productivity
    so earnings-related benefits are tied to the own past wage, as in the
    Finnish system.  Entry from N draws productivity from the stationary
    distribution of the chain.

    All flow rates are per model period (one year) - exactly the moments
    one estimates from linked employer-employee register spells in FIONA.
    """
    log_e, P_e = rouwenhorst(n_e, rho_e, sigma_e)
    e_grid = np.exp(log_e)
    pi_e = stationary_distribution(P_e)
    e_grid /= np.dot(pi_e, e_grid)          # mean efficiency = 1

    if p_eu + p_en > 1.0 or p_ue + p_un > 1.0 or p_ne + p_nu > 1.0:
        raise ValueError("Flow rates out of a state exceed 1.")

    nz = 2 * n_e + 1
    Pi = np.zeros((nz, nz))
    E = slice(0, n_e)
    U = slice(n_e, 2 * n_e)
    N = 2 * n_e

    Pi[E, E] = (1.0 - p_eu - p_en) * P_e
    Pi[E, U] = p_eu * P_e
    Pi[U, E] = p_ue * P_e
    Pi[U, U] = (1.0 - p_ue - p_un) * P_e
    for i in range(n_e):
        Pi[i, N] = p_en
        Pi[n_e + i, N] = p_un
    Pi[N, E] = p_ne * pi_e
    Pi[N, U] = p_nu * pi_e
    Pi[N, N] = 1.0 - p_ne - p_nu

    b_ui = np.clip(rr_ui * e_grid * w_anchor, ui_floor, ui_cap)

    names = [f"E{i+1}" for i in range(n_e)] + \
            [f"U{i+1}" for i in range(n_e)] + ["N"]
    return LaborProcess(
        names=names,
        Pi=Pi,
        e=np.concatenate([e_grid, np.zeros(n_e + 1)]),
        b0=np.concatenate([np.zeros(n_e), b_ui, [b_n]]),
        ha_elig=[False] * n_e + [ha_unemployed] * n_e + [ha_inactive],
        is_unemp=[False] * n_e + [True] * n_e + [False],
        is_inact=[False] * (2 * n_e) + [True],
    )
