"""SFC-planned, MPC-tracked level-2 controller.

The gate/obstacle-aware SFC planner supplies the reference trajectory; an acados
NMPC tracks it over a short horizon, replacing the Mellinger-PID tracker used by
``sfc_attitude_controller.py``. The PID version's residual failures are gate-frame
clips (the drone deviates from an already-clear planned path); a dynamics-aware
MPC tracks that path more tightly, which is aimed at that floor.

Only one Controller subclass lives here (the loader requires that); the OCP setup
and the planner are reused from the existing modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mpc import create_ocp_solver
from lsy_drone_racing.control.sfc_planner import SfcPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SfcMpcController(Controller):
    """SFC reference + acados NMPC tracker, emitting [roll, pitch, yaw, thrust]."""

    def __init__(self, obs: dict, info: dict, config: dict) -> None:
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        self._N = 25  # horizon steps (~0.5 s at 50 Hz)

        self._params = load_params("so_rpy", config.sim.drone_model)
        self._hover_thrust = float(self._params["mass"] * -self._params["gravity_vec"][-1])
        # Stiff lateral (xy) position + velocity tracking so the MPC follows the
        # obstacle-clear SFC reference instead of cutting corners into a pole.
        q_diag = np.array(
            [300.0, 300.0, 400.0, 1.0, 1.0, 1.0, 30.0, 30.0, 20.0, 5.0, 5.0, 5.0]
        )
        self._solver, self._ocp = create_ocp_solver(
            self._N * self._dt, self._N, self._params, q_diag=q_diag
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu

        self.planner = SfcPlanner(obs, int(self._freq))
        self._t = 0.0
        self._finished = False
        self._last_u = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float64)

    def _progress_time(self, pos: "NDArray[np.floating]") -> float:
        """Monotonic path-time nearest the drone within a forward window.

        Anchors the MPC reference to actual progress (closed-loop) so the horizon
        never runs away from the drone if it lags.
        """
        tt = self.planner.t_total
        if tt <= 0.0:
            return 0.0
        hi = min(self._t + 0.6, tt)
        ts = np.linspace(self._t, hi, 13)
        best_t, best_d = self._t, np.inf
        for t in ts:
            p, _, _ = self.planner.evaluate(float(t))
            d = float(np.linalg.norm(pos - p))
            if d < best_d:
                best_d, best_t = d, float(t)
        return best_t

    def compute_control(
        self, obs: dict, info: dict | None = None
    ) -> "NDArray[np.floating]":
        if self.planner.update(obs):  # replan to revealed positions
            self._t = 0.0

        pos = np.asarray(obs["pos"], dtype=np.float64)
        self._t = self._progress_time(pos)
        tt = self.planner.t_total

        # Initial state x0 = [pos, rpy, vel, drpy]
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((pos, rpy, np.asarray(obs["vel"], dtype=np.float64), drpy))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # Stage references from the SFC plan over the horizon.
        for j in range(self._N):
            tj = min(self._t + j * self._dt, tt)
            p, v, _ = self.planner.evaluate(float(tj))
            yref = np.zeros(self._ny)
            yref[0:3] = p
            yref[6:9] = v
            yref[15] = self._hover_thrust
            self._solver.set(j, "yref", yref)

        # Terminal reference.
        tN = min(self._t + self._N * self._dt, tt)
        pN, vN, _ = self.planner.evaluate(float(tN))
        yref_e = np.zeros(self._nx)
        yref_e[0:3] = pN
        yref_e[6:9] = vN
        self._solver.set(self._N, "yref", yref_e)

        self._solver.solve()
        u0 = np.asarray(self._solver.get(0, "u"), dtype=np.float64)
        if not np.all(np.isfinite(u0)):
            u0 = self._last_u
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
        self._t = 0.0
        self._finished = False
        self._last_u = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float64)
