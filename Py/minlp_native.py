"""
Native (P-Native) EV charging network MINLP: enumerates station-opening
patterns x, then multistart-NLP-solves the nonconvex per-pattern queueing
problem in (y, lambda, mu) for each -- exact but exponential in |J|.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

# benign trust-constr warning on some restarts; doesn't affect correctness.
warnings.filterwarnings("ignore", message="delta_grad == 0.0")

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# 1. Problem instance
# --------------------------------------------------------------------------

@dataclass
class EVInstance:
    zones: list          # I
    sites: list          # J
    Lambda: dict         # Lambda_i, demand per zone
    d: dict              # d[(i, j)] generalized travel cost
    f: dict              # f_j fixed opening cost
    M: dict              # M_j max service capacity when open
    v: dict              # v_j congestion weight
    kappa: dict          # kappa_j quadratic capacity-cost coefficient
    p: dict              # p_j base price (marginal revenue at lambda_j = 0)
    b: dict              # b_j revenue curvature (diminishing marginal revenue)
    eps: float = 1.0     # stability margin

    def c(self, j, mu_j):
        """Quadratic capacity-cost function c_j(mu_j) = kappa_j * mu_j^2."""
        return self.kappa[j] * mu_j ** 2

    def r(self, j, lam_j):
        """Concave quadratic revenue r_j(lambda_j) = p_j*lambda_j - b_j*lambda_j^2."""
        return self.p[j] * lam_j - self.b[j] * lam_j ** 2


def example_instance() -> EVInstance:
    zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    sites = ["A", "B", "C", "D", "E"]

    Lambda = {"Z1": 80.0, "Z2": 60.0, "Z3": 50.0, "Z4": 40.0, "Z5": 120.0}

    d = {
        ("Z1", "A"): 2.0, ("Z1", "B"): 5.0, ("Z1", "C"): 9.0, ("Z1", "D"): 12.0, ("Z1", "E"): 15.0,
        ("Z2", "A"): 6.0, ("Z2", "B"): 2.0, ("Z2", "C"): 7.0, ("Z2", "D"): 10.0, ("Z2", "E"): 14.0,
        ("Z3", "A"): 8.0, ("Z3", "B"): 6.0, ("Z3", "C"): 3.0, ("Z3", "D"): 5.0, ("Z3", "E"): 9.0,
        ("Z4", "D"): 4.0, ("Z4", "E"): 8.0, ("Z4", "A"): 10.0, ("Z4", "B"): 12.0, ("Z4", "C"): 15.0,
        ("Z5", "D"): 7.0, ("Z5", "E"): 5.0, ("Z5", "A"): 14.0, ("Z5", "B"): 11.0, ("Z5", "C"): 9.0,
    }

    f = {"A": 400.0, "B": 350.0, "C": 500.0, "D": 450.0, "E": 550.0}
    M = {"A": 130.0, "B": 110.0, "C": 120.0, "D": 100.0, "E": 140.0}
    v = {"A": 15.0, "B": 15.0, "C": 15.0, "D": 15.0, "E": 15.0}
    kappa = {"A": 0.02, "B": 0.022, "C": 0.02, "D": 0.02, "E": 0.02}
    p = {"A": 18.0, "B": 18.5, "C": 16.5, "D": 17.5, "E": 19.5}
    b = {"A": 0.03, "B": 0.03, "C": 0.03, "D": 0.03, "E": 0.03}

    return EVInstance(zones=zones, sites=sites, Lambda=Lambda, d=d, f=f,
                       M=M, v=v, kappa=kappa, p=p, b=b, eps=1.0)


# --------------------------------------------------------------------------
# 2. Continuous NLP subproblem for a fixed station-opening pattern S subset J
# --------------------------------------------------------------------------

def solve_station_subproblem(inst: EVInstance, S: tuple, n_restarts: int = 8):
    """Solves (P-Native)'s continuous part for x_j=1 iff j in S; returns
    (total_cost, detail_dict), or (0.0, trivial detail) if S is empty."""
    I, J = inst.zones, inst.sites
    nI, nS = len(I), len(S)

    if nS == 0:
        # complete recourse (Prop 4.2): closing everything costs 0 exactly.
        detail = {"y": {}, "lam": {j: 0.0 for j in J}, "mu": {j: 0.0 for j in J},
                  "opened": [], "fixed_cost": 0.0, "travel_cost": 0.0,
                  "capacity_cost": 0.0, "queue_cost": 0.0, "revenue": 0.0}
        return 0.0, detail

    # Decision vector layout: [ y (nI*nS) | lambda (nS) | mu (nS) ]
    n_y = nI * nS

    def y_idx(i_pos, j_pos):
        return i_pos * nS + j_pos

    fixed_cost = sum(inst.f[j] for j in S)

    def unpack(z):
        y = z[:n_y].reshape(nI, nS)
        lam = z[n_y:n_y + nS]
        mu = z[n_y + nS:n_y + 2 * nS]
        return y, lam, mu

    def objective(z):
        y, lam, mu = unpack(z)
        travel = 0.0
        for i_pos, i in enumerate(I):
            for j_pos, j in enumerate(S):
                travel += inst.d[(i, j)] * inst.Lambda[i] * y[i_pos, j_pos]

        station_cost = 0.0
        for j_pos, j in enumerate(S):
            mu_j, lam_j = mu[j_pos], lam[j_pos]
            denom = mu_j * (mu_j - lam_j)
            if denom <= 1e-9:
                # guards the mu_j -> lambda_j singularity during the line search.
                return 1e12
            q_cost = inst.v[j] * (lam_j ** 2) / denom
            station_cost += inst.c(j, mu_j) + q_cost - inst.r(j, lam_j)

        return fixed_cost + travel + station_cost

    # all non-objective constraints are linear -> exact LinearConstraint/Bounds.
    n_vars = n_y + 2 * nS
    rows, lb, ub = [], [], []

    # sum_j y_ij <= 1  for each zone i
    for i_pos in range(nI):
        row = np.zeros(n_vars)
        for j_pos in range(nS):
            row[y_idx(i_pos, j_pos)] = 1.0
        rows.append(row); lb.append(-np.inf); ub.append(1.0)

    # lambda_j - sum_i Lambda_i y_ij <= 0
    for j_pos in range(nS):
        row = np.zeros(n_vars)
        for i_pos, i in enumerate(I):
            row[y_idx(i_pos, j_pos)] = -inst.Lambda[i]
        row[n_y + j_pos] = 1.0
        rows.append(row); lb.append(-np.inf); ub.append(0.0)

    # lambda_j - mu_j <= -eps   (i.e. lambda_j + eps <= mu_j)
    for j_pos in range(nS):
        row = np.zeros(n_vars)
        row[n_y + j_pos] = 1.0
        row[n_y + nS + j_pos] = -1.0
        rows.append(row); lb.append(-np.inf); ub.append(-inst.eps)

    lin_constraint = LinearConstraint(np.array(rows), np.array(lb), np.array(ub))

    lower = np.zeros(n_vars)
    upper = np.concatenate([
        np.ones(n_y),
        np.full(nS, np.inf),
        np.array([inst.M[j] for j in S]),
    ])
    var_bounds = Bounds(lower, upper)

    best_val, best_z = math.inf, None

    for trial in range(n_restarts):
        z0 = np.zeros(n_y + 2 * nS)
        y0 = np.zeros((nI, nS))
        for i_pos in range(nI):
            j_pos = RNG.integers(0, nS)
            y0[i_pos, j_pos] = RNG.uniform(0.3, 1.0)
        z0[:n_y] = y0.flatten()

        lam_bar0 = np.array([sum(inst.Lambda[i] * y0[i_pos, j_pos]
                                  for i_pos, i in enumerate(I))
                              for j_pos in range(nS)])
        M_arr = np.array([inst.M[j] for j in S])
        lam0 = np.minimum(0.6 * lam_bar0 + 1e-3, M_arr - inst.eps - 1e-2)
        lam0 = np.maximum(lam0, 0.0)
        mu0 = np.minimum(M_arr, lam0 + inst.eps + RNG.uniform(5, 20, size=nS))
        z0[n_y:n_y + nS] = lam0
        z0[n_y + nS:] = mu0

        res = minimize(objective, z0, method="trust-constr",
                        bounds=var_bounds, constraints=[lin_constraint],
                        options={"maxiter": 1000, "gtol": 1e-9, "xtol": 1e-12})

        # accept any numerically-feasible improving candidate over res.status alone.
        viol = np.maximum(0.0, np.array(rows) @ res.x - np.array(ub))
        bound_viol = np.maximum(0.0, np.maximum(lower - res.x, res.x - upper))
        feasible = viol.max(initial=0.0) < 1e-5 and bound_viol.max(initial=0.0) < 1e-5

        if feasible and res.fun < best_val:
            best_val, best_z = res.fun, res.x

    if best_z is None:
        return math.inf, None

    y, lam, mu = unpack(best_z)
    travel_cost = sum(inst.d[(i, j)] * inst.Lambda[i] * y[i_pos, j_pos]
                       for i_pos, i in enumerate(I)
                       for j_pos, j in enumerate(S))
    capacity_cost = sum(inst.c(j, mu[j_pos]) for j_pos, j in enumerate(S))
    queue_cost = sum(inst.v[j] * lam[j_pos] ** 2 / (mu[j_pos] * (mu[j_pos] - lam[j_pos]))
                      for j_pos, j in enumerate(S))
    revenue = sum(inst.r(j, lam[j_pos]) for j_pos, j in enumerate(S))

    y_full = {(i, j): 0.0 for i in I for j in J}
    for i_pos, i in enumerate(I):
        for j_pos, j in enumerate(S):
            y_full[(i, j)] = float(y[i_pos, j_pos])

    detail = {
        "y": y_full,
        "lam": {**{j: 0.0 for j in J}, **{j: float(lam[j_pos]) for j_pos, j in enumerate(S)}},
        "mu": {**{j: 0.0 for j in J}, **{j: float(mu[j_pos]) for j_pos, j in enumerate(S)}},
        "opened": list(S),
        "fixed_cost": fixed_cost,
        "travel_cost": travel_cost,
        "capacity_cost": capacity_cost,
        "queue_cost": queue_cost,
        "revenue": revenue,
    }
    return best_val, detail


# --------------------------------------------------------------------------
# 3. Outer loop: enumerate opening patterns (the integer part of the MINLP)
# --------------------------------------------------------------------------

def solve_p_native(inst: EVInstance, n_restarts: int = 8, verbose: bool = True):
    J = inst.sites
    best_val, best_detail, best_S = math.inf, None, None

    for r in range(0, len(J) + 1):
        for S in itertools.combinations(J, r):
            val, detail = solve_station_subproblem(inst, S, n_restarts=n_restarts)
            if verbose:
                tag = "{" + ",".join(S) + "}" if S else "{}"
                print(f"  x-pattern {tag:>12s}  ->  objective = {val:10.3f}")
            if val < best_val:
                best_val, best_detail, best_S = val, detail, S

    return best_val, best_detail, best_S


# --------------------------------------------------------------------------
# 4. Run the example
# --------------------------------------------------------------------------

if __name__ == "__main__":
    inst = example_instance()

    print("Enumerating station-opening patterns x in {0,1}^|J| "
          f"({2 ** len(inst.sites)} patterns) and solving the resulting "
          "continuous NLP for each...\n")

    best_val, best_detail, best_S = solve_p_native(inst)

    print("\n=================== Optimal design found ===================")
    print(f"Opened stations           : {best_S}")
    print(f"Total objective value     : {best_val:.3f}")
    print("--- cost / revenue breakdown ---")
    print(f"  fixed opening cost      : {best_detail['fixed_cost']:.3f}")
    print(f"  travel cost             : {best_detail['travel_cost']:.3f}")
    print(f"  capacity cost           : {best_detail['capacity_cost']:.3f}")
    print(f"  queueing cost           : {best_detail['queue_cost']:.3f}")
    print(f"  revenue (subtracted)    : -{best_detail['revenue']:.3f}")

    print("\n--- station-level detail ---")
    for j in inst.sites:
        if j in best_S:
            lam_j, mu_j = best_detail["lam"][j], best_detail["mu"][j]
            rho_j = lam_j / mu_j if mu_j > 0 else 0.0
            print(f"  station {j}: mu = {mu_j:7.2f}, lambda = {lam_j:7.2f}, "
                  f"utilization rho = {rho_j:5.3f}")
        else:
            print(f"  station {j}: closed")

    print("\n--- routing (fraction of zone demand y_ij, only nonzero) ---")
    for (i, j), val in best_detail["y"].items():
        if val > 1e-6:
            print(f"  y[{i},{j}] = {val:.3f}  "
                  f"(routes {val * inst.Lambda[i]:.2f} of zone {i}'s demand to {j})")
