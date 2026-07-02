"""
Algorithm 3 (first-order re-optimization for a fixed topology), from
Section 4.3 of "Hidden Convexity for Integrated EV Charging Network
Design".

    Input: x, lambda_bar; initial (lambda^0, s^0, tau^0); step-size rule,
           tolerance eps_stat.
    Repeat:
      1. Compute gradients of the smooth part of the objective.
      2. Take a gradient or accelerated-gradient step.
      3. Project onto the feasible set:
             0 <= lambda <= lambda_bar,  lambda + eps*x <= sqrt(s),
             0 <= s <= M^2 * x,          lambda^2 <= tau * s.
      4. t <- t + 1.
    Until: stationarity / objective-improvement < eps_stat.

Uses the SAME instance and the SAME fixed topology (stations A, B open, C
closed) found by Algorithm 1 in ev_algorithm1_oa_gbd.py, then demonstrates
fast re-optimization under a couple of updated demand forecasts.
Correctness is verified against ev_algorithm1_oa_gbd.py's own (already
P-Native-cross-validated) SP-j solver.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

warnings.filterwarnings("ignore", message="Solution may be inaccurate")

def phi_prime(tau: float) -> float:
    """phi'(tau), for phi(tau) = tau/(1 - sqrt(tau))"""
    tau = min(max(tau, 0.0), TAU_BAR)
    u = math.sqrt(tau)
    return (2.0 - u) / (2.0 * (1.0 - u) ** 2)


def r_prime(inst: EVInstance, j: str, lam: float) -> float:
    """r_j'(lambda_j) for the instance's revenue r_j(lambda) = p_j*lambda
    - b_j*lambda^2"""
    return inst.p[j] - 2.0 * inst.b[j] * lam


def _tau_of(lam: float, s: float) -> float:
    if s <= 1e-12:
        return 0.0
    return min(max((lam * lam) / s, 0.0), TAU_BAR)


def station_objective(inst: EVInstance, j: str, lam: float, s: float) -> float:
    """F(lambda, s) = kappa_j*s + v_j*phi(lambda^2/s) - r_j(lambda) -- tau
    eliminated analytically (see module docstring)."""
    return inst.kappa[j] * s + inst.v[j] * phi(_tau_of(lam, s)) - inst.r(j, lam)


def station_gradient(inst: EVInstance, j: str, lam: float, s: float):
    """Gradient of F(lambda, s), via the chain rule through
    tau(lambda, s) = lambda^2/s -- this is what correctly captures the
    lambda/s (capacity-vs-congestion) coupling; see module docstring."""
    tau = _tau_of(lam, s)
    dphi = inst.v[j] * phi_prime(tau)
    d_lam = dphi * (2.0 * lam / s) - r_prime(inst, j, lam) if s > 1e-12 else -r_prime(inst, j, lam)
    d_s = inst.kappa[j] - (dphi * (tau / s) if s > 1e-12 else 0.0)
    return np.array([d_lam, d_s])


def verify_reduced_objective_convex(n_samples: int = 3000, seed: int = 0) -> bool:
    """Numerically checks that F(lambda, s) is jointly convex (random
    Hessian sampling up to ~99% utilization) -- backs up the module
    docstring's "partial minimization preserves convexity" argument with a
    concrete check on this instance's cost coefficients."""
    inst = example_instance()
    j = inst.sites[0]
    rng = np.random.default_rng(seed)
    h = 1e-3
    worst = 0.0
    for _ in range(n_samples):
        lam = rng.uniform(0.5, 150.0)
        mu = lam * rng.uniform(1.01, 3.0)  # utilization up to ~99%
        s = mu * mu
        f = lambda a, b: station_objective(inst, j, a, b)
        fxx = (f(lam + h, s) - 2 * f(lam, s) + f(lam - h, s)) / h ** 2
        fyy = (f(lam, s + h) - 2 * f(lam, s) + f(lam, s - h)) / h ** 2
        fxy = (f(lam + h, s + h) - f(lam + h, s - h) - f(lam - h, s + h)
               + f(lam - h, s - h)) / (4 * h ** 2)
        eig_min = np.linalg.eigvalsh(np.array([[fxx, fxy], [fxy, fyy]])).min()
        worst = min(worst, eig_min)
    return worst > -1e-2


# --------------------------------------------------------------------------
# Step 3: exact projection onto the (lambda, s) feasible set
# --------------------------------------------------------------------------

def project_onto_feasible_set(lam_hat: float, s_hat: float, lambda_bar_j: float,
                               M_j: float, eps: float):
    """Orthogonal projection of (lam_hat, s_hat) onto
        {0 <= lam <= lambda_bar_j, (lam + eps)^2 <= s <= M_j^2}
    A tiny convex QCQP, solved with cvxpy/CLARABEL."""
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(nonneg=True)
    constraints = [
        lam <= lambda_bar_j,
        cp.square(lam + eps) <= s,
        s <= M_j ** 2,
    ]
    obj = cp.Minimize(cp.square(lam - lam_hat) + cp.square(s - s_hat))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if lam.value is None or s.value is None:
            raise cp.error.SolverError("no solution returned")
    except cp.error.SolverError:
        return None
    return float(lam.value), float(s.value)


# --------------------------------------------------------------------------
# Algorithm 3: (accelerated) projected gradient with backtracking
# --------------------------------------------------------------------------

@dataclass
class ReoptResult:
    lam: float
    s: float
    tau: float
    mu: float
    objective: float
    iterations: int
    wall_time: float
    history: list


def reoptimize_station(inst: EVInstance, j: str, xj: int, lambda_bar_j: float,
                        z0=None, tol: float = 1e-9, max_iter: int = 200,
                        t_max: float = 1.0e4) -> ReoptResult:
    """Algorithm 3 for a single station: fixed x_j, updated lambda_bar_j.

    Spectral Projected Gradient (SPG): a per-coordinate Barzilai-Borwein
    step size with a backtracking safeguard."""
    t0 = time.perf_counter()

    if xj == 0 or lambda_bar_j <= 1e-12:
        return ReoptResult(lam=0.0, s=0.0, tau=0.0, mu=0.0, objective=0.0,
                            iterations=0, wall_time=time.perf_counter() - t0,
                            history=[0.0])

    M_j, eps = inst.M[j], inst.eps

    if z0 is None:
        lam0 = min(0.5 * lambda_bar_j, M_j - eps)
        mu0 = min(M_j, lam0 + eps + 5.0)
        z0 = (lam0, mu0 ** 2)
    z = np.array(project_onto_feasible_set(z0[0], z0[1], lambda_bar_j, M_j, eps))

    history = [station_objective(inst, j, *z)]
    g = station_gradient(inst, j, *z)

    # step 1 (first iteration): no secant pair yet, so use a small,
    # deliberately conservative default step just to get one.
    t_diag = np.array([1e-4, 1e-4])
    k = 0

    for k in range(1, max_iter + 1):
        cur_obj = history[-1]

        # step 2: per-coordinate Barzilai-Borwein step size from the
        # secant pair (Delta z, Delta g) between this and the previous
        # accepted iterate.
        if k > 1:
            dz, dg = z - z_prev, g - g_prev
            with np.errstate(divide="ignore", invalid="ignore"):
                bb = np.where(np.abs(dz * dg) > 1e-14, (dz * dz) / (dz * dg), t_diag)
            t_diag = np.clip(np.abs(bb), 1e-8, t_max)

        # step 3: project the trial point; backtrack (halve t) on either
        # a failed/degenerate projection or a non-improving objective.
        t = t_diag.copy()
        for _ in range(60):
            cand = project_onto_feasible_set(*(z - t * g), lambda_bar_j, M_j, eps)
            if cand is not None:
                z_trial = np.array(cand)
                trial_obj = station_objective(inst, j, *z_trial)
                if trial_obj <= cur_obj + 1e-12:
                    break
            t *= 0.5
        else:
            z_trial, trial_obj = z, cur_obj

        z_prev, g_prev = z, g
        z = z_trial
        g = station_gradient(inst, j, *z)
        history.append(trial_obj)
        t_diag = t

        # step 4 / stopping: objective-improvement criterion.
        if abs(history[-2] - history[-1]) <= tol * max(1.0, abs(history[-1])):
            break

    lam, s = z
    tau = _tau_of(lam, s)
    return ReoptResult(lam=float(lam), s=float(s), tau=float(tau),
                        mu=math.sqrt(max(s, 0.0)), objective=float(history[-1]),
                        iterations=k, wall_time=time.perf_counter() - t0,
                        history=history)


def reoptimize_network(inst: EVInstance, x: dict, lambda_bar: dict, **kwargs):
    """Runs Algorithm 3 independently for every station (they are separable
    given fixed x and lambda_bar -- no coupling across j at this stage)."""
    results = {}
    for j in inst.sites:
        z0 = kwargs.pop("z0_map", {}).get(j) if "z0_map" in kwargs else None
        results[j] = reoptimize_station(inst, j, x[j], lambda_bar[j],
                                         z0=z0, **{k: v for k, v in kwargs.items() if k != "z0_map"})
    return results


def print_reopt_report(inst: EVInstance, x: dict, lambda_bar: dict, results: dict):
    total_obj = sum(results[j].objective for j in inst.sites)
    total_time = sum(results[j].wall_time for j in inst.sites)
    print(f"  total station recourse cost : {total_obj:.4f}   "
          f"(re-optimized in {total_time * 1000:.2f} ms total)")
    for j in inst.sites:
        if x[j] == 0:
            print(f"  station {j}: closed")
            continue
        r = results[j]
        rho = r.lam / r.mu if r.mu > 0 else 0.0
        print(f"  station {j}: lambda_bar={lambda_bar[j]:7.2f}  ->  "
              f"mu={r.mu:7.2f}  lambda={r.lam:7.2f}  rho={rho:5.3f}  "
              f"V_j={r.objective:9.3f}  [{r.iterations} iters, "
              f"{r.wall_time * 1000:6.2f} ms]")


if __name__ == "__main__":
    print("=================== Convexity sanity check ===================")
    ok = verify_reduced_objective_convex()
    print(f"F(lambda, s) jointly convex over random samples "
          f"(incl. near tau -> 1): {ok}\n")

    inst = example_instance()

    # Fixed topology from Algorithm 1's optimal solution (stations A, B
    # open, C closed), with its routed-demand (lambda_bar) values.
    x_fixed = {"A": 1, "B": 1, "C": 0}
    lambda_bar_baseline = {"A": 85.21, "B": 94.65, "C": 0.0}

    print("=================== Algorithm 3: baseline re-optimization "
          "===================")

    results = reoptimize_network(inst, x_fixed, lambda_bar_baseline)
    print_reopt_report(inst, x_fixed, lambda_bar_baseline, results)

    print("\n--- cross-check against Algorithm 1's own SP-j solver "
          "(scipy trust-constr) ---")
    for j in inst.sites:
        if x_fixed[j] == 0:
            continue
        ref = solve_sp_j(inst, j, 1, lambda_bar_baseline[j])
        got = results[j]
        diff = abs(ref.Vj - got.objective)
        tol = 1e-2 * max(1.0, abs(ref.Vj))
        print(f"  station {j}: Algorithm 1 V_j = {ref.Vj:9.4f}  (mu={ref.mu:.2f}, "
              f"lambda={ref.lam:.2f})   Algorithm 3 V_j = {got.objective:9.4f}  "
              f"(mu={got.mu:.2f}, lambda={got.lam:.2f})   diff = {diff:.5f}  "
              f"({'MATCH' if diff <= tol else 'MISMATCH'})")

    # --- demonstrate fast re-optimization under updated demand forecasts ---
    scenarios = {
        "baseline         ": lambda_bar_baseline,
        "+20% demand surge": {j: v * 1.2 for j, v in lambda_bar_baseline.items()},
        "-15% demand drop ": {j: v * 0.85 for j, v in lambda_bar_baseline.items()},
    }

    print("\n=================== Re-optimizing under updated demand "
          "forecasts ===================")

    prev = None
    for name, lb in scenarios.items():
        t0 = time.perf_counter()
        # warm-start from the previous scenario's operating point, exactly
        # as an operator re-solving under a newly updated forecast would.
        res = {}
        for j in inst.sites:
            z0 = None
            if prev is not None and x_fixed[j] == 1 and prev[j].s > 0:
                z0 = (prev[j].lam, prev[j].s)
            res[j] = reoptimize_station(inst, j, x_fixed[j], lb[j], z0=z0)
        elapsed = time.perf_counter() - t0

        print(f"scenario: {name}  (total re-optimization time: {elapsed * 1000:.2f} ms)")
        print_reopt_report(inst, x_fixed, lb, res)
        print()
        prev = res
