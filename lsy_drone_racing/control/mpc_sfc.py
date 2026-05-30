"""SFC-planned, MPC-tracked level-2 controller with obstacle-avoidance constraints.

The gate/obstacle-aware SFC planner supplies the reference trajectory; an acados
NMPC tracks it over a short horizon. Unlike a pure tracking MPC (which cuts
corners off a curved reference into poles), this adds *soft* nonlinear
constraints keeping the drone clear of each obstacle pole, with pole positions as
runtime parameters updated every replan. A short vertical-climb hand-off and a
hover warm-start avoid the takeoff transient that crashed the unconstrained MPC.

Only one Controller subclass lives here (the loader requires that).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mpc import create_acados_model
from lsy_drone_racing.control.sfc_planner import SfcPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _build_constrained_solver(
    N: int, dt: float, params: dict, n_obs: int, r_safe: float, q_diag: "NDArray"
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """OCP with LINEAR_LS tracking cost + soft per-pole keep-out constraints.

    Pole xy positions are acados parameters ``p = [x0,y0,x1,y1,...]`` so they can
    be updated each tick. The constraint ``(px-xi)^2+(py-yi)^2 >= r_safe^2`` is
    softened (slack-penalised) to stay feasible under the so_rpy<->first_principles
    model mismatch.
    """
    ocp = AcadosOcp()
    ocp.model = create_acados_model(params)
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny, ny_e = nx + nu, nx
    ocp.solver_options.N_horizon = N

    # --- tracking cost (LINEAR_LS) ---
    Q = np.diag(np.asarray(q_diag, dtype=float))
    Rm = np.diag([1.0, 1.0, 1.0, 50.0])
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q, Rm)
    ocp.cost.W_e = Q.copy()
    Vx = np.zeros((ny, nx)); Vx[0:nx, 0:nx] = np.eye(nx); ocp.cost.Vx = Vx
    Vu = np.zeros((ny, nu)); Vu[nx:nx + nu, :] = np.eye(nu); ocp.cost.Vu = Vu
    Vx_e = np.zeros((ny_e, nx)); Vx_e[0:nx, 0:nx] = np.eye(nx); ocp.cost.Vx_e = Vx_e
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # --- obstacle keep-out as soft nonlinear constraints ---
    p = ca.MX.sym("obs_xy", 2 * n_obs)
    ocp.model.p = p
    ocp.parameter_values = np.zeros(2 * n_obs)
    px, py = ocp.model.x[0], ocp.model.x[1]
    h = ca.vertcat(*[(px - p[2 * i]) ** 2 + (py - p[2 * i + 1]) ** 2 for i in range(n_obs)])
    ocp.model.con_h_expr = h
    ocp.constraints.lh = np.full(n_obs, r_safe ** 2)
    ocp.constraints.uh = np.full(n_obs, 1e9)
    ocp.constraints.idxsh = np.arange(n_obs)
    ocp.cost.zl = 1e3 * np.ones(n_obs)
    ocp.cost.zu = 1e3 * np.ones(n_obs)
    ocp.cost.Zl = 1e2 * np.ones(n_obs)
    ocp.cost.Zu = 1e2 * np.ones(n_obs)

    # --- box constraints (tilt + thrust), like the example MPC ---
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, params["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, params["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx))

    # --- solver options: converged SQP + full condensing (stable with the
    # nonlinear keep-out constraints; SQP_RTI's single iteration threw QP
    # status-3 errors and produced erratic commands). ---
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 40
    ocp.solver_options.tf = N * dt

    return AcadosOcpSolver(
        ocp, json_file="c_generated_code/sfc_mpc.json", verbose=False, build=True, generate=True
    ), ocp


class SfcMpcController(Controller):
    """SFC reference + acados NMPC tracker with obstacle constraints."""

    def __init__(self, obs: dict, info: dict, config: dict) -> None:
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        self._N = 25

        self._params = load_params("so_rpy", config.sim.drone_model)
        self._hover_thrust = float(self._params["mass"] * -self._params["gravity_vec"][-1])

        obstacles = np.asarray(obs.get("obstacles_pos", np.zeros((0, 3))), dtype=np.float64)
        self._n_obs = int(obstacles.shape[0])
        # drone half-width + pole + margin for the keep-out radius.
        self._r_safe = 0.086 + 0.015 + 0.10
        q_diag = np.array([300.0, 300.0, 400.0, 1.0, 1.0, 1.0, 30.0, 30.0, 20.0, 5.0, 5.0, 5.0])
        self._solver, self._ocp = _build_constrained_solver(
            self._N, self._dt, self._params, self._n_obs, self._r_safe, q_diag
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu

        self.planner = SfcPlanner(obs, int(self._freq))
        self._start_z = float(np.asarray(obs["pos"])[2])
        self._airborne = False
        self._t = 0.0
        self._finished = False
        self._last_u = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float64)

    def _obstacle_params(self, obs: dict) -> "NDArray[np.floating]":
        obst = np.asarray(obs.get("obstacles_pos", np.zeros((self._n_obs, 3))), dtype=np.float64)
        return obst[:, :2].reshape(-1)

    def _progress_time(self, pos: "NDArray[np.floating]") -> float:
        tt = self.planner.t_total
        if tt <= 0.0:
            return 0.0
        ts = np.linspace(self._t, min(self._t + 0.6, tt), 13)
        best_t, best_d = self._t, np.inf
        for t in ts:
            p, _, _ = self.planner.evaluate(float(t))
            d = float(np.linalg.norm(pos - p))
            if d < best_d:
                best_d, best_t = d, float(t)
        return best_t

    def compute_control(self, obs: dict, info: dict | None = None) -> "NDArray[np.floating]":
        if self.planner.update(obs):
            self._t = 0.0

        pos = np.asarray(obs["pos"], dtype=np.float64)
        # Takeoff phase: climb straight up until airborne, then hand off to the
        # MPC. The cold-started SQP_RTI solve gives a poor first command that
        # tips the (model-mismatched) drone into the floor; a brief level climb
        # gets it clear before the MPC engages.
        if not self._airborne:
            if pos[2] > self._start_z + 0.25:
                self._airborne = True
            else:
                u = np.array([0.0, 0.0, 0.0, 1.3 * self._hover_thrust], dtype=np.float32)
                self._last_u = u.astype(np.float64)
                return u

        self._t = self._progress_time(pos)
        tt = self.planner.t_total
        p_obs = self._obstacle_params(obs)

        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((pos, rpy, np.asarray(obs["vel"], dtype=np.float64), drpy))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        for j in range(self._N):
            tj = min(self._t + j * self._dt, tt)
            p, v, _ = self.planner.evaluate(float(tj))
            yref = np.zeros(self._ny)
            yref[0:3] = p
            yref[6:9] = v
            yref[15] = self._hover_thrust
            self._solver.set(j, "yref", yref)
            self._solver.set(j, "p", p_obs)
        tN = min(self._t + self._N * self._dt, tt)
        pN, vN, _ = self.planner.evaluate(float(tN))
        yref_e = np.zeros(self._nx)
        yref_e[0:3] = pN
        yref_e[6:9] = vN
        self._solver.set(self._N, "yref", yref_e)
        self._solver.set(self._N, "p", p_obs)

        status = self._solver.solve()
        u0 = np.asarray(self._solver.get(0, "u"), dtype=np.float64)
        # Reuse the last good command on a failed/garbage solve.
        if status != 0 or not np.all(np.isfinite(u0)):
            u0 = self._last_u
        # Takeoff guard: never command below hover thrust while still near the
        # floor, so the (mismatched) initial solve cannot sink the drone.
        if pos[2] < self._start_z + 0.25:
            u0[3] = max(float(u0[3]), self._hover_thrust)
        self._last_u = u0

        if int(obs.get("target_gate", 0)) == -1:
            self._finished = True
        return u0.astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        return self._finished

    def episode_callback(self) -> None:
        self.episode_reset()

    def episode_reset(self) -> None:
        self.planner.episode_reset()
        self._airborne = False
        self._t = 0.0
        self._finished = False
        self._last_u = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float64)
