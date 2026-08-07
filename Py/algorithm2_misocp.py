"""
Algorithm 2 (monolithic MISOCP benchmark) for (P-MISOCP) / Reformulation I:
exact rotated-cone constraints in (lambda_hat=sqrt(lambda), mu, rho), valid
only under a revenue concave in lambda_hat (Prop 3.7)
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from ev_minlp_native import EVInstance, example_instance

RHO_BAR = 0.995  # rho-bar: max utilization allowed in (P-MISOCP)

# Regime-compliant revenue (concave in lambda_hat), DEFAULTS valid only for
# example_instance()'s site names "A".."E" -- use derive_regime_revenue()
# for any other instance, passed as P_HAT_override/B_HAT_override.
P_HAT = {"A": 170.0, "B": 175.0, "C": 155.0, "D": 10.0, "E": 10.0}
B_HAT = {"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 3.0}


def r_hat(j: str, lam_hat, P_HAT_override: dict | None = None,
          B_HAT_override: dict | None = None):
    """Regime-compliant revenue, concave in lambda_hat_j; uses module-level
    P_HAT/B_HAT unless overrides are given."""
    P = P_HAT_override if P_HAT_override is not None else P_HAT
    B = B_HAT_override if B_HAT_override is not None else B_HAT
    return P[j] * lam_hat - B[j] * lam_hat ** 2


def derive_regime_revenue(inst: EVInstance, lambda_ref_frac: float = 0.6):
    """Derives a Reformulation-I-compliant (P_HAT, B_HAT) for ANY instance,
    by matching r_hat_j(lambda_hat) to the native r_j(lambda) in value and
    slope at lambda_ref_j = lambda_ref_frac * M_j;"""
    P_HAT_new, B_HAT_new = {}, {}
    for j in inst.sites:
        lam_ref = max(lambda_ref_frac * inst.M[j], 1e-6)
        lam_hat_ref = math.sqrt(lam_ref)
        r_ref = inst.r(j, lam_ref)
        slope_ref = inst.p[j] - 2.0 * inst.b[j] * lam_ref  # r_j'(lambda_ref)

        b_hat = r_ref / lam_hat_ref ** 2 - 2.0 * slope_ref
        b_hat = max(b_hat, 1e-3)
        p_hat = 2.0 * lam_hat_ref * (b_hat + slope_ref)

        P_HAT_new[j], B_HAT_new[j] = p_hat, b_hat
    return P_HAT_new, B_HAT_new


def demonstrate_revenue_trap(inst: EVInstance):
    """Reproduces Proposition 3.7 numerically"""
    j = inst.sites[0]
    p_j, b_j = inst.p[j], inst.b[j]

    def r_hat_naive(lam_hat):
        return p_j * lam_hat ** 2 - b_j * lam_hat ** 4

    grid = np.linspace(1e-3, 12.0, 400)
    vals = r_hat_naive(grid)
    second_deriv = np.gradient(np.gradient(vals, grid), grid)
    convex_region = second_deriv > 1e-6

    print(f"Proposition 3.7 check (station {j}, native p_j={p_j}, b_j={b_j}):")
    print(f"  r_hat_naive(lambda_hat) := r_j(lambda_hat^2) = {p_j}*lambda_hat^2 "
          f"- {b_j}*lambda_hat^4")
    if convex_region.any():
        lo, hi = grid[convex_region].min(), grid[convex_region].max()
        print(f"  -> r_hat_naive is CONVEX (not concave) for lambda_hat in "
              f"[{lo:.2f}, {hi:.2f}] (d2/dlambda_hat^2 > 0 there).")
    else:
        print("  -> (unexpectedly) concave everywhere on this grid.")
    print(f"  Using the regime-compliant r_hat_j(lambda_hat) = P_HAT_j*lambda_hat "
          f"- B_HAT_j*lambda_hat^2 instead, which IS concave for all lambda_hat "
          f"(coefficient B_HAT_j = {B_HAT[j]} >= 0).\n")


# --------------------------------------------------------------------------
# (P-MISOCP) model builder
# --------------------------------------------------------------------------

@dataclass
class MISOCPVars:
    x: object
    y: object
    lam_bar: object
    lam_hat: object
    mu: object
    rho: object
    q: object
    tc: object
    tr: object


def build_p_misocp(inst: EVInstance, x_fixed: np.ndarray | None = None,
                    P_HAT_override: dict | None = None,
                    B_HAT_override: dict | None = None):
    """Builds (P-MISOCP): x is a Boolean Variable if x_fixed is None (the
    true MISOCP), else fixed data giving a pure continuous SOCP.
    Pass P_HAT_override/B_HAT_override (see derive_regime_revenue) for any
    instance other than example_instance()."""
    I, J = inst.zones, inst.sites
    nI, nJ = len(I), len(J)

    if x_fixed is None:
        x = cp.Variable(nJ, boolean=True)
    else:
        x = x_fixed

    y = cp.Variable((nI, nJ), nonneg=True)
    lam_bar = cp.Variable(nJ, nonneg=True)
    lam_hat = cp.Variable(nJ, nonneg=True)
    mu = cp.Variable(nJ, nonneg=True)
    rho = cp.Variable(nJ, nonneg=True)
    q = cp.Variable(nJ, nonneg=True)
    tc = cp.Variable(nJ)
    tr = cp.Variable(nJ)

    constraints = []

    # sum_j y_ij <= 1
    for i_pos in range(nI):
        constraints.append(cp.sum(y[i_pos, :]) <= 1)

    # y_ij <= x_j
    for i_pos in range(nI):
        for j_pos in range(nJ):
            constraints.append(y[i_pos, j_pos] <= x[j_pos])

    # lambda_bar_j = sum_i Lambda_i y_ij
    for j_pos, j in enumerate(J):
        constraints.append(
            lam_bar[j_pos] == cp.sum(cp.multiply(
                np.array([inst.Lambda[i] for i in I]), y[:, j_pos]))
        )

    for j_pos, j in enumerate(J):
        constraints.append(cp.square(lam_hat[j_pos]) <= lam_bar[j_pos])

        # rotated cones (Prop 3.6), via quad_over_lin (DCP-friendly form).
        constraints.append(rho[j_pos] <= RHO_BAR)                 # step 2: tighten
        constraints.append(cp.quad_over_lin(rho[j_pos], 1 - rho[j_pos]) <= q[j_pos])
        constraints.append(cp.quad_over_lin(lam_hat[j_pos], mu[j_pos]) <= rho[j_pos])

        constraints.append(tc[j_pos] >= inst.kappa[j] * cp.square(mu[j_pos]))
        constraints.append(tr[j_pos] >= -r_hat(j, lam_hat[j_pos],
                                                 P_HAT_override, B_HAT_override))

        constraints.append(mu[j_pos] <= inst.M[j] * x[j_pos])
        constraints.append(cp.square(lam_hat[j_pos]) + inst.eps * x[j_pos] <= mu[j_pos])

        # step 3: valid inequality tying lambda_hat_j to the station's own capacity cap.
        constraints.append(cp.square(lam_hat[j_pos]) <= inst.M[j] * x[j_pos])

    fixed_cost = cp.sum(cp.multiply(np.array([inst.f[j] for j in J]), x))
    travel_cost = cp.sum(cp.multiply(
        np.array([[inst.d[(i, j)] * inst.Lambda[i] for j in J] for i in I]), y))
    station_cost = cp.sum(cp.multiply(np.array([inst.v[j] for j in J]), q) + tc + tr)

    objective = cp.Minimize(fixed_cost + travel_cost + station_cost)
    prob = cp.Problem(objective, constraints)

    return prob, MISOCPVars(x=x, y=y, lam_bar=lam_bar, lam_hat=lam_hat,
                             mu=mu, rho=rho, q=q, tc=tc, tr=tr)


# --------------------------------------------------------------------------
# Algorithm 2: monolithic MISOCP benchmark
# --------------------------------------------------------------------------

def solve_p_misocp(inst: EVInstance, verbose: bool = True,
                    P_HAT_override: dict | None = None,
                    B_HAT_override: dict | None = None):
    """Algorithm 2 pseudocode, steps 1-6."""

    use_gurobi = "GUROBI" in cp.installed_solvers()

    if use_gurobi:
        if verbose:
            print("Step 1-3: building (P-MISOCP) (full conic model, tightened "
                  "bounds, valid inequality)...")
            print("Step 4: calling the MISOCP solver (GUROBI)...\n")

        prob, v = build_p_misocp(inst, P_HAT_override=P_HAT_override,
                                  B_HAT_override=B_HAT_override)  # step 1-3
        t0 = time.perf_counter()
        prob.solve(solver=cp.GUROBI, verbose=False)
        wall_time = time.perf_counter() - t0

        stats = prob.solver_stats
        gmodel = stats.extra_stats
        node_count = getattr(gmodel, "NodeCount", None)
        mip_gap = getattr(gmodel, "MIPGap", None)
        best_bound = getattr(gmodel, "ObjBound", None)   # best (dual) bound
        incumbent = getattr(gmodel, "ObjVal", None)       # best feasible solution found
        num_vars = getattr(gmodel, "NumVars", None)
        num_constrs = getattr(gmodel, "NumConstrs", None)

        if verbose:
            # step 5: record runtime, node count, bounds, incumbent, gap
            print("Step 5: solver diagnostics")
            print(f"  status              : {prob.status}")
            print(f"  wall-clock time     : {wall_time:.4f} s "
                  f"(solver-reported: {stats.solve_time:.4f} s)")
            print(f"  branch-and-bound nodes explored : {node_count}")
            print(f"  best bound (LB)     : {best_bound:.6f}")
            print(f"  incumbent (UB)      : {incumbent:.6f}")
            print(f"  final MIP gap       : {mip_gap}")
            print(f"  model size          : {num_vars} vars, {num_constrs} constrs\n")

        x_val = np.round(v.x.value).astype(int)
        return _extract_solution(inst, v, prob.value, x_val, wall_time,
                                  {"node_count": node_count, "mip_gap": mip_gap,
                                   "best_bound": best_bound, "incumbent": incumbent})

    # --- portable fallback: enumerate x, solve each continuous SOCP -----
    if verbose:
        print("GUROBI not available: falling back to enumerating station-"
              "opening patterns x in {0,1}^|J| and solving.\n")

    J = inst.sites
    best_val, best_result = np.inf, None
    t0 = time.perf_counter()
    for pattern in itertools.product([0, 1], repeat=len(J)):
        x_fixed = np.array(pattern, dtype=float)
        prob, v = build_p_misocp(inst, x_fixed=x_fixed, P_HAT_override=P_HAT_override,
                                  B_HAT_override=B_HAT_override)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except cp.error.SolverError:
            continue
        if prob.status not in ("optimal", "optimal_inaccurate"):
            continue
        if prob.value < best_val:
            best_val = prob.value
            best_result = (v, prob.value, x_fixed)
        if verbose:
            print(f"  x={pattern} -> objective = {prob.value:10.3f}")
    wall_time = time.perf_counter() - t0

    # every enumerated SOCP is convex, so the best incumbent IS the true optimum (gap 0).
    if verbose:
        print("Step 5: enumeration diagnostics")
        print(f"  wall-clock time     : {wall_time:.4f} s")
        print(f"  patterns enumerated (~ 'nodes') : {2 ** len(J)}")
        print(f"  best bound (LB)     : {best_val:.6f}")
        print(f"  incumbent (UB)      : {best_val:.6f}")
        print(f"  final gap           : 0.0 (exact, by convexity of each subproblem)\n")

    v, val, x_val = best_result
    return _extract_solution(inst, v, val, x_val.astype(int), wall_time,
                              {"node_count": 2 ** len(J), "mip_gap": 0.0,
                               "best_bound": best_val, "incumbent": best_val})


def _extract_solution(inst, v: MISOCPVars, objective, x_val, wall_time, diag):
    J = inst.sites
    y_val = v.y.value
    lam_hat_val = v.lam_hat.value
    mu_val = v.mu.value

    detail = {
        "objective": float(objective),
        "x": {j: int(x_val[j_pos]) for j_pos, j in enumerate(J)},
        "y": {(i, j): float(y_val[i_pos, j_pos])
              for i_pos, i in enumerate(inst.zones) for j_pos, j in enumerate(J)},
        "lam": {j: float(lam_hat_val[j_pos] ** 2) for j_pos, j in enumerate(J)},
        "mu": {j: float(mu_val[j_pos]) for j_pos, j in enumerate(J)},
        "wall_time": wall_time,
        "diag": diag,
    }
    return detail


def print_solution(inst: EVInstance, detail: dict):
    opened = [j for j in inst.sites if detail["x"][j] == 1]
    print(f"Opened stations           : {opened}")
    print(f"Total objective value     : {detail['objective']:.3f}")
    print("\n--- station-level detail ---")
    for j in inst.sites:
        if detail["x"][j] == 1:
            lam, mu = detail["lam"][j], detail["mu"][j]
            rho = lam / mu if mu > 0 else 0.0
            print(f"  station {j}: mu = {mu:7.2f}, lambda = {lam:7.2f}, "
                  f"utilization rho = {rho:5.3f}")
        else:
            print(f"  station {j}: closed")

    print("\n--- routing (fraction of zone demand y_ij, only nonzero) ---")
    for (i, j), val in detail["y"].items():
        if val > 1e-6:
            print(f"  y[{i},{j}] = {val:.3f}  "
                  f"(routes {val * inst.Lambda[i]:.2f} of zone {i}'s demand to {j})")


# --------------------------------------------------------------------------
# Independent cross-check: brute-force native MINLP with the SAME
# (regime-compliant) revenue, transported back to native coordinates.
# --------------------------------------------------------------------------

def r_native_from_r_hat(j: str, lam: float) -> float:
    """r_native(lambda_j) := r_hat_j(sqrt(lambda_j)), for the cross-check below only."""
    lam = max(lam, 0.0)
    return r_hat(j, np.sqrt(lam))


def solve_native_cross_check(inst: EVInstance, n_restarts: int = 6):
    """Re-solves the native MINLP with r_native
    to independently verify (P-MISOCP)'s answer."""
    import math
    from scipy.optimize import minimize

    I, J = inst.zones, inst.sites
    rng = np.random.default_rng(0)

    def solve_for_pattern(S):
        nI, nS = len(I), len(S)
        if nS == 0:
            return 0.0, {j: 0.0 for j in J}, {j: 0.0 for j in J}, {}

        n_y = nI * nS

        def unpack(z):
            y = z[:n_y].reshape(nI, nS)
            lam = z[n_y:n_y + nS]
            mu = z[n_y + nS:n_y + 2 * nS]
            return y, lam, mu

        fixed_cost = sum(inst.f[j] for j in S)

        def objective(z):
            y, lam, mu = unpack(z)
            travel = sum(inst.d[(i, j)] * inst.Lambda[i] * y[ip, jp]
                         for ip, i in enumerate(I) for jp, j in enumerate(S))
            cost = 0.0
            for jp, j in enumerate(S):
                mu_j, lam_j = mu[jp], lam[jp]
                denom = mu_j * (mu_j - lam_j)
                if denom <= 1e-9:
                    return 1e12
                q_cost = inst.v[j] * (lam_j ** 2) / denom
                cost += inst.kappa[j] * mu_j ** 2 + q_cost - r_native_from_r_hat(j, lam_j)
            return fixed_cost + travel + cost

        from scipy.optimize import LinearConstraint, Bounds
        n_vars = n_y + 2 * nS
        rows, lb, ub = [], [], []
        for ip in range(nI):
            row = np.zeros(n_vars)
            for jp in range(nS):
                row[ip * nS + jp] = 1.0
            rows.append(row); lb.append(-np.inf); ub.append(1.0)
        for jp in range(nS):
            row = np.zeros(n_vars)
            for ip, i in enumerate(I):
                row[ip * nS + jp] = -inst.Lambda[i]
            row[n_y + jp] = 1.0
            rows.append(row); lb.append(-np.inf); ub.append(0.0)
        for jp in range(nS):
            row = np.zeros(n_vars)
            row[n_y + jp] = 1.0
            row[n_y + nS + jp] = -1.0
            rows.append(row); lb.append(-np.inf); ub.append(-inst.eps)
        lin_con = LinearConstraint(np.array(rows), np.array(lb), np.array(ub))

        lower = np.zeros(n_vars)
        upper = np.concatenate([np.ones(n_y), np.full(nS, np.inf),
                                 np.array([inst.M[j] for j in S])])
        var_bounds = Bounds(lower, upper)

        best_val, best_z = math.inf, None
        for trial in range(n_restarts):
            y0 = np.zeros((nI, nS))
            for ip in range(nI):
                jp = rng.integers(0, nS)
                y0[ip, jp] = rng.uniform(0.3, 1.0)
            lam_bar0 = np.array([sum(inst.Lambda[i] * y0[ip, jp] for ip, i in enumerate(I))
                                  for jp in range(nS)])
            M_arr = np.array([inst.M[j] for j in S])
            lam0 = np.clip(0.5 * lam_bar0 + 1e-3, 0.0, M_arr - inst.eps - 1e-2)
            mu0 = np.minimum(M_arr, lam0 + inst.eps + rng.uniform(5, 20, size=nS))
            z0 = np.concatenate([y0.flatten(), lam0, mu0])

            res = minimize(objective, z0, method="trust-constr", bounds=var_bounds,
                            constraints=[lin_con],
                            options={"maxiter": 500, "gtol": 1e-9, "xtol": 1e-12})
            viol = np.maximum(0.0, np.array(rows) @ res.x - np.array(ub))
            bviol = np.maximum(0.0, np.maximum(lower - res.x, res.x - upper))
            if viol.max(initial=0.0) < 1e-5 and bviol.max(initial=0.0) < 1e-5:
                if res.fun < best_val:
                    best_val, best_z = res.fun, res.x

        if best_z is None:
            return math.inf, None, None, None
        y, lam, mu = unpack(best_z)
        return (best_val, {j: (float(lam[jp]) if j in S else 0.0) for jp, j in enumerate(S)}
                or {j: 0.0 for j in J},
                {j: (float(mu[jp]) if j in S else 0.0) for jp, j in enumerate(S)}
                or {j: 0.0 for j in J}, S)

    best_val, best_S = math.inf, None
    for r in range(len(J) + 1):
        for S in itertools.combinations(J, r):
            val, lam, mu, _ = solve_for_pattern(S)
            if val < best_val:
                best_val, best_S = val, S

    return best_val, best_S


if __name__ == "__main__":
    inst = example_instance()

    print("=================== Proposition 3.7 diagnostic ===================\n")
    demonstrate_revenue_trap(inst)

    print("=================== Algorithm 2: monolithic MISOCP (P-MISOCP) "
          "===================\n")
    detail = solve_p_misocp(inst)
    print_solution(inst, detail)

    print("\n=================== Independent cross-check ===================")
    print("Re-solving the NATIVE (non-convexified) queueing MINLP with the "
          "SAME r_hat-implied revenue...\n")
    native_val, native_S = solve_native_cross_check(inst)
    print(f"(P-MISOCP)          optimal objective : {detail['objective']:.3f}   "
          f"opened = {tuple(j for j in inst.sites if detail['x'][j] == 1)}")
    print(f"native cross-check  optimal objective : {native_val:.3f}   "
          f"opened = {native_S}")
    diff = abs(native_val - detail["objective"])
    tol = 1e-2 * max(1.0, abs(native_val))
    print(f"\nAbsolute difference: {diff:.6f} "
          f"({'MATCH' if diff <= tol else 'MISMATCH'} within tolerance {tol:.4f})")

    print("\nNote: (P-MISOCP) uses a different revenue functional form than "
          "(P-Native)/(P-HC), so its "
          "objective value is not directly comparable to Algorithm 1.")
