"""
Algorithm 1 (exact OA/GBD) for the hidden-convexity reformulation (P-HC):
alternates a small MILP master (x, y, lambda_bar, theta) with per-station
convex NLP subproblems (SP-j) that supply Benders/OA cuts
"""

from __future__ import annotations

import concurrent.futures
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


def _gurobi_available() -> bool:
    """Probes for a working Gurobi to pick the master backend: scipy's milp
    (HiGHS) has no warm-start hook, so GurobiMasterProblem is used when
    possible, else MasterProblem."""
    try:
        import gurobipy as gp
        m = gp.Model()
        m.dispose()
        return True
    except Exception:
        return False

TAU_BAR = 0.995  # tau-bar: max squared utilization allowed in (P-HC)

# lets the master preview that opening a station might be profitable
# (theta_j >= -BIG_M_THETA when x_j=1) instead of only ever seeing the
# trivial theta_j >= 0 and never opening anything; OA cuts tighten it fast.
BIG_M_THETA = 1.0e5


def phi(tau: float) -> float:
    """phi(tau) = tau / (1 - sqrt(tau)), convexified queueing term."""
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
                       n_restarts: int = 4, z0_hint: tuple | None = None):
    """Solves (SP-j) for fixed (lambda_bar_j, x_j); returns (Vj, lam, s, tau,
    mu)"""

    if xj == 0:
        # complete recourse (Prop 4.2): closed station costs 0 for any lambda_bar_j.
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

    trial_z0s = []
    if z0_hint is not None:
        lam_h, s_h, tau_h = z0_hint
        # clip into this call's box in case lambda_bar_j shifted since the hint was recorded.
        lam_h = min(max(lam_h, 0.0), lambda_bar_j)
        s_h = min(max(s_h, 0.0), M_j ** 2)
        tau_h = min(max(tau_h, 0.0), TAU_BAR)
        trial_z0s.append(np.array([lam_h, s_h, tau_h]))

    for trial in range(n_restarts):
        frac = 0.3 + 0.5 * trial / max(n_restarts - 1, 1)
        lam0 = min(frac * lambda_bar_j, M_j - eps - 1e-2)
        lam0 = max(lam0, 0.0)
        mu0 = min(M_j, lam0 + eps + 5.0 + 3.0 * trial)
        s0 = mu0 ** 2
        tau0 = min((lam0 / mu0) ** 2 if mu0 > 0 else 0.0, TAU_BAR)
        trial_z0s.append(np.array([lam0, s0, tau0]))

    best = None
    for z0 in trial_z0s:
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
               h: float = 1.0, n_restarts: int = 4,
               z0_hint: tuple | None = None) -> StationSolution:
    """Solves (SP-j) and estimates pi_j = d(Vj)/d(lambda_bar_j) <= 0 via a
    finite difference of V_j."""
    Vj, lam, s, tau, mu = _solve_sp_j_value(inst, j, xj, lambda_bar_j, n_restarts,
                                             z0_hint=z0_hint)

    if xj == 0:
        pi_j = 0.0
    else:
        Vj_plus, *_ = _solve_sp_j_value(inst, j, xj, lambda_bar_j + h, n_restarts,
                                         z0_hint=z0_hint)
        if lambda_bar_j - h >= 0:
            Vj_minus, *_ = _solve_sp_j_value(inst, j, xj, lambda_bar_j - h, n_restarts,
                                              z0_hint=z0_hint)
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

    @property
    def num_cuts(self) -> int:
        """Total OA/Benders cuts accumulated so far."""
        return len(self.cuts)

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

        # initial relaxation: theta_j + BIG_M_THETA * x_j >= 0 (see BIG_M_THETA above).
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
        # theta's real lower bound comes from the linking row/cuts above, not a flat bound.
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


class GurobiMasterProblem:
    """Same as MasterProblem, but as a persistent gurobipy Model that
    adds cuts incrementally and MIP-warm-starts each re-solve from the
    previous iterate."""

    def __init__(self, inst: EVInstance):
        import gurobipy as gp
        from gurobipy import GRB
        self._gp, self._GRB = gp, GRB

        self.inst = inst
        self.I, self.J = inst.zones, inst.sites

        model = gp.Model()
        model.setParam("OutputFlag", 0)
        self.model = model

        x = {j: model.addVar(vtype=GRB.BINARY, name=f"x_{j}") for j in self.J}
        y = {(i, j): model.addVar(lb=0.0, ub=1.0, name=f"y_{i}_{j}")
             for i in self.I for j in self.J}
        lam_ub = sum(inst.Lambda.values())
        lambda_bar = {j: model.addVar(lb=0.0, ub=lam_ub, name=f"lb_{j}") for j in self.J}
        theta = {j: model.addVar(lb=-GRB.INFINITY, name=f"th_{j}") for j in self.J}
        model.update()
        self.x, self.y, self.lambda_bar, self.theta = x, y, lambda_bar, theta

        for i in self.I:
            model.addConstr(gp.quicksum(y[(i, j)] for j in self.J) <= 1.0)
        for i in self.I:
            for j in self.J:
                model.addConstr(y[(i, j)] <= x[j])
        for j in self.J:
            model.addConstr(
                lambda_bar[j] == gp.quicksum(inst.Lambda[i] * y[(i, j)] for i in self.I)
            )
        # initial BIG_M_THETA relaxation -- same rationale as MasterProblem.
        for j in self.J:
            model.addConstr(theta[j] + BIG_M_THETA * x[j] >= 0.0)

        fixed_cost = gp.quicksum(inst.f[j] * x[j] for j in self.J)
        travel_cost = gp.quicksum(inst.d[(i, j)] * inst.Lambda[i] * y[(i, j)]
                                   for i in self.I for j in self.J)
        theta_sum = gp.quicksum(theta[j] for j in self.J)
        model.setObjective(fixed_cost + travel_cost + theta_sum, GRB.MINIMIZE)
        model.update()

        self._have_prev_solution = False
        self._n_cuts = 0

    def add_cut(self, j_pos, alpha, pi):
        j = self.J[j_pos]
        self.model.addConstr(
            self.theta[j] - alpha * self.x[j] - pi * self.lambda_bar[j] >= 0.0
        )
        self._n_cuts += 1

    @property
    def num_cuts(self) -> int:
        """Total OA/Benders cuts accumulated so far."""
        return self._n_cuts

    def solve(self):
        if self._have_prev_solution:
            for j in self.J:
                self.x[j].Start = self._prev_x[j]
                self.theta[j].Start = self._prev_theta[j]
                self.lambda_bar[j].Start = self._prev_lambda_bar[j]
            for i in self.I:
                for j in self.J:
                    self.y[(i, j)].Start = self._prev_y[(i, j)]

        self.model.optimize()
        if self.model.Status != self._GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi master problem failed: status {self.model.Status}")

        x = {j: int(round(self.x[j].X)) for j in self.J}
        y = {(i, j): float(self.y[(i, j)].X) for i in self.I for j in self.J}
        lambda_bar = {j: float(self.lambda_bar[j].X) for j in self.J}
        theta = {j: float(self.theta[j].X) for j in self.J}

        self._prev_x, self._prev_y = x, y
        self._prev_lambda_bar, self._prev_theta = lambda_bar, theta
        self._have_prev_solution = True

        return self.model.ObjVal, x, y, lambda_bar, theta


# --------------------------------------------------------------------------
# Algorithm 1 main loop
# --------------------------------------------------------------------------

def solve_p_hc_via_oa_gbd(inst: EVInstance, eps_gap: float = 1e-4,
                           max_iter: int = 50, verbose: bool = True,
                           max_workers: int | None = None,
                           sp_n_restarts: int = 4):
    """sp_n_restarts: solve_sp_j's restart-grid size on top of its always-
    tried warm-start hint. Since (SP-j) is provably convex, one good solve
    already finds the global optimum."""
    I, J = inst.zones, inst.sites

    if _gurobi_available():
        master = GurobiMasterProblem(inst)
        if verbose:
            print("Master backend: GUROBI\n")
    else:
        master = MasterProblem(inst)
        if verbose:
            print("Master backend: scipy.optimize.milp/HiGHS\n")

    LB, UB = -math.inf, math.inf
    best_incumbent = None
    # j -> (lam, s, tau) from j's last open-station solve, used to warm-start
    # the next iteration's solve_sp_j; cleared when a station closes.
    prev_station_point: dict = {}

    if verbose:
        print(f"{'iter':>4} | {'master obj (LB)':>16} | {'UB candidate':>13} | {'best UB':>10} | {'gap':>10}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for k in range(max_iter):
            mp_obj, x, y, lambda_bar, theta = master.solve()
            LB = max(LB, mp_obj)

            # complete recourse (Prop 4.2): closed stations cost 0, no NLP solve needed.
            station_sols = {j: StationSolution(Vj=0.0, lam=0.0, s=0.0, tau=0.0,
                                                mu=0.0, pi=0.0)
                             for j in J if x[j] == 0}

            # (SP-j) is convex and independent per station, so solve open ones
            # concurrently, each warm-started from its own previous solution.
            open_stations = [j for j in J if x[j] == 1]
            futures = {
                pool.submit(solve_sp_j, inst, j, 1, lambda_bar[j],
                            n_restarts=sp_n_restarts,
                            z0_hint=prev_station_point.get(j)): j
                for j in open_stations
            }
            for fut in concurrent.futures.as_completed(futures):
                j = futures[fut]
                station_sols[j] = fut.result()

            for j in J:
                if x[j] == 1:
                    sol = station_sols[j]
                    prev_station_point[j] = (sol.lam, sol.s, sol.tau)
                else:
                    prev_station_point.pop(j, None)

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
                print(f"dual values (pi_j for open stations): {[station_sols[j].pi for j in J if x[j] == 1]}")

            # Prop 4.1 only licenses cuts from x_j^k=1 iterates; a cut from a
            # closed station would wrongly collapse to theta_j >= 0 for future opens.
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

    # stashed on the incumbent dict so existing 3-way unpacking still works.
    if best_incumbent is not None:
        best_incumbent["n_iterations"] = k + 1
        best_incumbent["n_cuts"] = master.num_cuts

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
    print("Re-solving the native MINLP on the SAME instance...\n")
    native_val, native_detail, native_S = solve_p_native(inst, verbose=False)

    print(f"(P-Native) optimal objective : {native_val:.3f}   opened = {native_S}")
    print(f"(P-HC)     optimal objective : {incumbent['objective']:.3f}   "
          f"opened = {tuple(j for j in inst.sites if incumbent['x'][j] == 1)}")
    diff = abs(native_val - incumbent["objective"])
    tol = 1e-2 * max(1.0, abs(native_val))
    print(f"\nAbsolute difference in objective value: {diff:.6f} "
          f"({'MATCH' if diff <= tol else 'MISMATCH'} within tolerance {tol:.4f})")
