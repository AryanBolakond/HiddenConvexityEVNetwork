"""
Algorithm 1 (exact OA/GBD) for the hidden-convexity reformulation (P-HC),
from Section 4.1 of "Hidden Convexity for Integrated EV Charging Network
Design".

(P-HC) replaces the nonconvex native queueing term

    Q_inf,j(lambda_j, mu_j) = lambda_j^2 / (mu_j (mu_j - lambda_j))

with the lifted variables

    s_j   := mu_j^2                    (squared service rate)
    tau_j := (lambda_j / mu_j)^2       (squared utilization)

under which Q_inf,j = phi(tau_j) := tau_j / (1 - sqrt(tau_j)), a function
that IS convex on [0,1) (Proposition 3.2), and the coupling constraint
lambda_j^2 <= tau_j * s_j defines a convex (rotated-cone) set (Prop 3.3).
Station recourse (SP-j) is therefore a convex program for any fixed
(lambda_bar_j, x_j) (Theorem 3.4), and its value function V_j is convex in
lambda_bar_j (Corollary 3.5) -- which is exactly what licenses Benders/OA
cuts on it.

Algorithm 1 alternates between:

  * a MASTER problem (MP) -- a small MILP over (x, y, lambda_bar, theta)
    with routing/opening constraints plus accumulated OA cuts
        theta_j >= alpha_j^k x_j + pi_j^k lambda_bar_j,
  * per-station convex SUBPROBLEMS (SP-j) -- solved to global optimality
    (they are convex, so any KKT point found is the global optimum),
    which supply the recourse value V_j and the dual price pi_j of the
    lambda_j <= lambda_bar_j constraint, used to build the next cut
    (Proposition 4.1).

Only numpy/scipy are used:
  * SP-j (convex NLP, 3 variables) is solved with scipy.optimize.minimize
    (trust-constr) -- since it is provably convex, this returns the global
    optimum and reliable Lagrange multipliers.
  * (MP) (a small MILP) is solved with scipy.optimize.milp (HiGHS), which
    ships with scipy -- no external MILP solver needed.

The same instance data as the native-MINLP script (ev_minlp_native.py) is
reused, and at the end this script re-solves (P-Native) with that script's
brute-force+multistart routine to check that both approaches agree.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import (
    minimize, milp, LinearConstraint, NonlinearConstraint, Bounds,
)

from ev_minlp_native import EVInstance, example_instance, solve_p_native

warnings.filterwarnings("ignore", message="delta_grad == 0.0")
warnings.filterwarnings("ignore", message="Singular Jacobian matrix")

TAU_BAR = 0.995  # tau-bar: max squared utilization allowed in (P-HC)

# theta_j >= 0 is the textbook Benders default, and it IS valid whenever
# x_j = 0 -- complete recourse (Proposition 4.2) forces V_j(lambda_bar_j, 0)
# = 0 exactly. But V_j subtracts revenue r_j(lambda_j) and so CAN be
# negative once a station is actually open (a profitable station). A plain
# "theta_j >= 0" bound would then make the very first master solve blind to
# any upside from opening anything -- it would just set x_j = 0 everywhere,
# generate the trivial cut theta_j >= 0 again, and "converge" to that wrong
# fixed point immediately (this is exactly what happened before this fix).
# BIG_M_THETA gives the master a (loose, but valid) preview that opening a
# station MIGHT be very profitable, via the linking row
#     theta_j + BIG_M_THETA * x_j >= 0,
# i.e. theta_j >= 0 when x_j = 0, theta_j >= -BIG_M_THETA when x_j = 1.
# Subsequent OA cuts tighten theta_j to the true V_j quickly.
BIG_M_THETA = 1.0e5


def phi(tau: float) -> float:
    """phi(tau) = tau / (1 - sqrt(tau)), the convexified queueing term."""
    tau = min(max(tau, 0.0), 1.0 - 1e-9)
    return tau / (1.0 - math.sqrt(tau))


# --------------------------------------------------------------------------
# Station subproblem (SP-j): convex NLP in (lambda_j, s_j, tau_j)
# --------------------------------------------------------------------------

@dataclass
class StationSolution:
    Vj: float          # optimal recourse value
    lam: float
    s: float
    tau: float
    mu: float           # = sqrt(s), for reporting
    pi: float           # subgradient of Vj w.r.t. lambda_bar_j  (<= 0)


def _solve_sp_j_value(inst: EVInstance, j: str, xj: int, lambda_bar_j: float,
                       n_restarts: int = 4):
    """Solve (SP-j) for fixed (lambda_bar_j, x_j); returns (Vj, lam, s, tau, mu)
    only -- no dual price (see solve_sp_j for that)."""

    if xj == 0:
        # Complete recourse (Proposition 4.2): closing the station forces
        # s_j = 0 (0 <= s_j <= M_j^2 * 0) and hence lambda_j = 0, at zero
        # cost, for ANY lambda_bar_j.
        return 0.0, 0.0, 0.0, 0.0, 0.0

    kappa_j, v_j, eps = inst.kappa[j], inst.v[j], inst.eps
    M_j = inst.M[j]

    def objective(z):
        lam, s, tau = z
        return kappa_j * s + v_j * phi(tau) - inst.r(j, lam)

    # tau_j * s_j - lambda_j^2 >= 0   (rotated-cone constraint, Prop 3.3)
    def cone_fun(z):
        lam, s, tau = z
        return tau * s - lam ** 2

    cone_con = NonlinearConstraint(cone_fun, 0.0, np.inf)

    # sqrt(s_j) - lambda_j - eps * x_j >= 0
    def stab_fun(z):
        lam, s, tau = z
        return math.sqrt(max(s, 0.0)) - lam - eps * xj

    stab_con = NonlinearConstraint(stab_fun, 0.0, np.inf)

    bounds = Bounds([0.0, 0.0, 0.0], [lambda_bar_j, M_j ** 2, TAU_BAR])

    best = None
    for trial in range(n_restarts):
        frac = 0.3 + 0.5 * trial / max(n_restarts - 1, 1)
        lam0 = min(frac * lambda_bar_j, M_j - eps - 1e-2)
        lam0 = max(lam0, 0.0)
        mu0 = min(M_j, lam0 + eps + 5.0 + 3.0 * trial)
        s0 = mu0 ** 2
        tau0 = min((lam0 / mu0) ** 2 if mu0 > 0 else 0.0, TAU_BAR)
        z0 = np.array([lam0, s0, tau0])

        res = minimize(objective, z0, method="trust-constr",
                        bounds=bounds, constraints=[cone_con, stab_con],
                        options={"maxiter": 500, "gtol": 1e-10, "xtol": 1e-13})

        viol_cone = max(0.0, -cone_fun(res.x))
        viol_stab = max(0.0, -stab_fun(res.x))
        feasible = viol_cone < 1e-6 and viol_stab < 1e-6

        if feasible and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        raise RuntimeError(f"SP-{j} failed to converge from any restart")

    lam, s, tau = best.x
    mu = math.sqrt(max(s, 0.0))
    return float(best.fun), float(lam), float(s), float(tau), mu


def solve_sp_j(inst: EVInstance, j: str, xj: int, lambda_bar_j: float,
               h: float = 1.0, n_restarts: int = 4) -> StationSolution:
    """Solve (SP-j) and also estimate pi_j = d(Vj)/d(lambda_bar_j) <= 0.

    V_j is convex and (weakly) non-increasing in lambda_bar_j (Corollary
    3.5: more routed demand only ever relaxes the 0 <= lambda_j <=
    lambda_bar_j constraint). Rather than trust the NLP solver's raw
    Lagrange multiplier -- which we found to be numerically unreliable
    exactly at the degenerate boundary lambda_bar_j = 0 (singular Jacobian,
    since lambda_j <= lambda_bar_j and lambda_j >= 0 are both active at
    once there) -- we estimate the slope directly with a finite difference
    of V_j itself. This is simple, robust everywhere including at that
    boundary, and costs only 1-2 extra convex NLP solves per call.
    """
    Vj, lam, s, tau, mu = _solve_sp_j_value(inst, j, xj, lambda_bar_j, n_restarts)

    if xj == 0:
        pi_j = 0.0
    else:
        Vj_plus, *_ = _solve_sp_j_value(inst, j, xj, lambda_bar_j + h, n_restarts)
        if lambda_bar_j - h >= 0:
            Vj_minus, *_ = _solve_sp_j_value(inst, j, xj, lambda_bar_j - h, n_restarts)
            pi_j = (Vj_plus - Vj_minus) / (2 * h)
        else:
            pi_j = (Vj_plus - Vj) / h
        pi_j = min(pi_j, 0.0)  # guard: Vj is non-increasing, so pi_j <= 0

    return StationSolution(Vj=Vj, lam=lam, s=s, tau=tau, mu=mu, pi=pi_j)


# --------------------------------------------------------------------------
# Master problem (MP): MILP over (x, y, lambda_bar, theta)
# --------------------------------------------------------------------------

class MasterProblem:
    """Builds and re-solves (MP) as cuts accumulate."""

    def __init__(self, inst: EVInstance):
        self.inst = inst
        self.I, self.J = inst.zones, inst.sites
        self.nI, self.nJ = len(self.I), len(self.J)
        self.cuts = []  # list of (j, alpha_j^k, pi_j^k)

        # variable layout: [ x (nJ) | y (nI*nJ) | lambda_bar (nJ) | theta (nJ) ]
        self.n_x = self.nJ
        self.n_y = self.nI * self.nJ
        self.n_lb = self.nJ
        self.n_th = self.nJ
        self.n_vars = self.n_x + self.n_y + self.n_lb + self.n_th

    def _idx_x(self, j_pos):
        return j_pos

    def _idx_y(self, i_pos, j_pos):
        return self.n_x + i_pos * self.nJ + j_pos

    def _idx_lb(self, j_pos):
        return self.n_x + self.n_y + j_pos

    def _idx_th(self, j_pos):
        return self.n_x + self.n_y + self.n_lb + j_pos

    def add_cut(self, j_pos, alpha, pi):
        self.cuts.append((j_pos, alpha, pi))

    def solve(self):
        inst, I, J, nI, nJ = self.inst, self.I, self.J, self.nI, self.nJ

        c = np.zeros(self.n_vars)
        for j_pos, j in enumerate(J):
            c[self._idx_x(j_pos)] = inst.f[j]
        for i_pos, i in enumerate(I):
            for j_pos, j in enumerate(J):
                c[self._idx_y(i_pos, j_pos)] = inst.d[(i, j)] * inst.Lambda[i]
        for j_pos in range(nJ):
            c[self._idx_th(j_pos)] = 1.0

        rows, lb, ub = [], [], []

        # sum_j y_ij <= 1
        for i_pos in range(nI):
            row = np.zeros(self.n_vars)
            for j_pos in range(nJ):
                row[self._idx_y(i_pos, j_pos)] = 1.0
            rows.append(row); lb.append(-np.inf); ub.append(1.0)

        # y_ij - x_j <= 0
        for i_pos in range(nI):
            for j_pos in range(nJ):
                row = np.zeros(self.n_vars)
                row[self._idx_y(i_pos, j_pos)] = 1.0
                row[self._idx_x(j_pos)] = -1.0
                rows.append(row); lb.append(-np.inf); ub.append(0.0)

        # lambda_bar_j - sum_i Lambda_i y_ij = 0
        for j_pos, j in enumerate(J):
            row = np.zeros(self.n_vars)
            row[self._idx_lb(j_pos)] = 1.0
            for i_pos, i in enumerate(I):
                row[self._idx_y(i_pos, j_pos)] = -inst.Lambda[i]
            rows.append(row); lb.append(0.0); ub.append(0.0)

        # initial relaxation: theta_j + BIG_M_THETA * x_j >= 0 (see note by
        # BIG_M_THETA above) -- lets the master "see" that an as-yet-unpriced
        # open station could be very profitable, instead of only ever seeing
        # the true default theta_j >= 0 that holds for closed stations.
        for j_pos in range(nJ):
            row = np.zeros(self.n_vars)
            row[self._idx_th(j_pos)] = 1.0
            row[self._idx_x(j_pos)] = BIG_M_THETA
            rows.append(row); lb.append(0.0); ub.append(np.inf)

        # OA/Benders cuts: theta_j - alpha*x_j - pi*lambda_bar_j >= 0
        for (j_pos, alpha, pi) in self.cuts:
            row = np.zeros(self.n_vars)
            row[self._idx_th(j_pos)] = 1.0
            row[self._idx_x(j_pos)] = -alpha
            row[self._idx_lb(j_pos)] = -pi
            rows.append(row); lb.append(0.0); ub.append(np.inf)

        constraints = LinearConstraint(np.array(rows), np.array(lb), np.array(ub))

        lower = np.zeros(self.n_vars)
        # theta's real lower bound comes from the linking row / OA cuts
        # above, not from a flat variable bound -- leaving the default 0
        # here would silently re-impose theta_j >= 0 on top of those rows
        # and defeat the BIG_M_THETA relaxation (same pitfall as the
        # lambda_j <= lambda_bar_j duplication fixed in solve_sp_j).
        lower[self._idx_th(0):self._idx_th(0) + self.n_th] = -np.inf
        upper = np.concatenate([
            np.ones(self.n_x),
            np.ones(self.n_y),
            np.full(self.n_lb, sum(inst.Lambda.values())),
            np.full(self.n_th, np.inf),
        ])
        bounds = Bounds(lower, upper)

        integrality = np.zeros(self.n_vars)
        integrality[:self.n_x] = 1  # x_j binary

        res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
        if not res.success:
            raise RuntimeError(f"Master problem failed: {res.message}")

        z = res.x
        x = {j: round(z[self._idx_x(j_pos)]) for j_pos, j in enumerate(J)}
        y = {(i, j): float(z[self._idx_y(i_pos, j_pos)])
             for i_pos, i in enumerate(I) for j_pos, j in enumerate(J)}
        lambda_bar = {j: float(z[self._idx_lb(j_pos)]) for j_pos, j in enumerate(J)}
        theta = {j: float(z[self._idx_th(j_pos)]) for j_pos, j in enumerate(J)}

        return res.fun, x, y, lambda_bar, theta


# --------------------------------------------------------------------------
# Algorithm 1 main loop
# --------------------------------------------------------------------------

def solve_p_hc_via_oa_gbd(inst: EVInstance, eps_gap: float = 1e-4,
                           max_iter: int = 50, verbose: bool = True):
    I, J = inst.zones, inst.sites
    master = MasterProblem(inst)

    LB, UB = -math.inf, math.inf
    best_incumbent = None

    if verbose:
        print(f"{'iter':>4} | {'master obj (LB)':>16} | {'UB candidate':>13} | {'best UB':>10} | {'gap':>10}")

    for k in range(max_iter):
        mp_obj, x, y, lambda_bar, theta = master.solve()
        LB = max(LB, mp_obj)

        station_sols = {}
        for j_pos, j in enumerate(J):
            if x[j] == 1:
                station_sols[j] = solve_sp_j(inst, j, 1, lambda_bar[j])
            else:
                # Complete recourse (Prop 4.2): V_j(*, 0) = 0 exactly, no
                # NLP solve needed -- and per Proposition 4.1, cuts are only
                # ever derived from an iterate WITH x_j^k = 1 (see note by
                # the cut-adding loop below for why a "closed-station" cut
                # would actually be UNSOUND for future x_j = 1 iterates).
                station_sols[j] = StationSolution(Vj=0.0, lam=0.0, s=0.0,
                                                   tau=0.0, mu=0.0, pi=0.0)

        fixed_cost = sum(inst.f[j] * x[j] for j in J)
        travel_cost = sum(inst.d[(i, j)] * inst.Lambda[i] * y[(i, j)]
                           for i in I for j in J)
        recourse_cost = sum(station_sols[j].Vj for j in J)
        ub_candidate = fixed_cost + travel_cost + recourse_cost

        if ub_candidate < UB:
            UB = ub_candidate
            best_incumbent = {
                "x": dict(x), "y": dict(y), "lambda_bar": dict(lambda_bar),
                "stations": {j: station_sols[j] for j in J},
                "fixed_cost": fixed_cost, "travel_cost": travel_cost,
                "recourse_cost": recourse_cost, "objective": ub_candidate,
            }

        gap = (UB - LB) / max(1.0, abs(UB))
        if verbose:
            print(f"{k:>4} | {mp_obj:>16.4f} | {ub_candidate:>13.4f} | {UB:>10.4f} | {gap:>10.6f}")

        # Proposition 4.1 only licenses a cut from an iterate with x_j^k = 1:
        # the tangent theta_j >= V_j(lambda_bar_j^k,1) + pi_j^k (lambda_bar_j
        # - lambda_bar_j^k) is valid there by convexity of V_j(., 1). A
        # "cut" built the same way from an x_j^k = 0 iterate would collapse
        # to theta_j >= 0 -- true when x_j = 0 (complete recourse), but NOT
        # a valid bound once reused against a FUTURE iterate with x_j = 1,
        # since V_j(., 1) can be very negative (a profitable open station).
        # Skipping closed stations here is what keeps the BIG_M_THETA
        # relaxation (see MasterProblem) from being silently overwritten.
        for j_pos, j in enumerate(J):
            if x[j] == 0:
                continue
            sol = station_sols[j]
            alpha = sol.Vj - sol.pi * lambda_bar[j]
            master.add_cut(j_pos, alpha, sol.pi)

        if UB - LB <= eps_gap * max(1.0, abs(UB)):
            if verbose:
                print(f"\nConverged after {k + 1} iterations "
                      f"(UB - LB = {UB - LB:.6g} <= tol).")
            break
    else:
        if verbose:
            print("\nReached max_iter without closing the gap "
                  f"(UB - LB = {UB - LB:.6g}).")

    return UB, LB, best_incumbent


def print_incumbent(inst: EVInstance, incumbent: dict):
    x, y, stations = incumbent["x"], incumbent["y"], incumbent["stations"]
    opened = [j for j in inst.sites if x[j] == 1]

    print(f"Opened stations           : {opened}")
    print(f"Total objective value     : {incumbent['objective']:.3f}")
    print("--- cost / revenue breakdown ---")
    print(f"  fixed opening cost      : {incumbent['fixed_cost']:.3f}")
    print(f"  travel cost             : {incumbent['travel_cost']:.3f}")
    print(f"  recourse cost (Sum V_j) : {incumbent['recourse_cost']:.3f}")

    print("\n--- station-level detail ---")
    for j in inst.sites:
        if x[j] == 1:
            sol = stations[j]
            rho = sol.lam / sol.mu if sol.mu > 0 else 0.0
            print(f"  station {j}: mu = {sol.mu:7.2f}, lambda = {sol.lam:7.2f}, "
                  f"utilization rho = {rho:5.3f}")
        else:
            print(f"  station {j}: closed")

    print("\n--- routing (fraction of zone demand y_ij, only nonzero) ---")
    for (i, j), val in y.items():
        if val > 1e-6:
            print(f"  y[{i},{j}] = {val:.3f}  "
                  f"(routes {val * inst.Lambda[i]:.2f} of zone {i}'s demand to {j})")


if __name__ == "__main__":
    inst = example_instance()

    print("Running Algorithm 1 (exact OA/GBD) on the hidden-convexity "
          "reformulation (P-HC)...\n")
    ub, lb, incumbent = solve_p_hc_via_oa_gbd(inst)

    print("\n=================== Algorithm 1 result (P-HC) ===================")
    print_incumbent(inst, incumbent)

    print("\n=================== Cross-check against (P-Native) ===================")
    print("Re-solving the native MINLP (brute-force x + multistart NLP, from "
          "ev_minlp_native.py) on the SAME instance...\n")
    native_val, native_detail, native_S = solve_p_native(inst, verbose=False)

    print(f"(P-Native) optimal objective : {native_val:.3f}   opened = {native_S}")
    print(f"(P-HC)     optimal objective : {incumbent['objective']:.3f}   "
          f"opened = {tuple(j for j in inst.sites if incumbent['x'][j] == 1)}")
    diff = abs(native_val - incumbent["objective"])
    tol = 1e-2 * max(1.0, abs(native_val))
    print(f"\nAbsolute difference in objective value: {diff:.6f} "
          f"({'MATCH' if diff <= tol else 'MISMATCH'} within tolerance {tol:.4f})")
