"""Cascaded-PID attitude controller (level-2 qualification)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_pid_v4.cascade_pid import PositionPid
from lsy_drone_racing.control.attitude_pid_v4.attitude import tracking_command
from lsy_drone_racing.control.attitude_pid_v4.tuning import (
    QualificationTuning,
    gate1_offset_tuning,
)
from lsy_drone_racing.control.attitude_pid_v4.trajectory import (
    build_reference_curve,
    load_route_overrides,
)
from lsy_drone_racing.control.attitude_pid_v4.geometry import (
    DEFAULT_GATE_POS,
    DEFAULT_GATE_RPY,
    DEFAULT_OBSTACLES,
    normalize_gate_index,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class _ObservationFrame:
    target_gate: int
    gate_pos: "NDArray[np.floating]"
    gate_quat: "NDArray[np.floating]"
    pos: "NDArray[np.floating]"
    vel: "NDArray[np.floating]"
    quat: "NDArray[np.floating]"


def _gates_rpy(gate_quat: "NDArray[np.floating]") -> "NDArray[np.floating]":
    rpy = np.zeros((gate_quat.shape[0], 3), dtype=np.float64)
    for i, q in enumerate(gate_quat):
        rpy[i] = R.from_quat(q).as_euler("xyz", degrees=False)
    return rpy


class QualificationController(Controller):
    """Stateful attitude controller for the level-2 qualification track."""

    def __init__(self, obs: dict, info: dict, config: dict) -> None:
        super().__init__(obs, info, config)
        self._env_freq = float(config.env.freq)
        self._dt = 1.0 / self._env_freq
        self._tuning: QualificationTuning = gate1_offset_tuning()
        self._gravity = 9.81

        # Drone mass via drone_models.
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])

        # Geometry + tuning state.
        self._overrides = load_route_overrides()
        self._obstacles = np.asarray(DEFAULT_OBSTACLES, dtype=np.float64)
        self._leg_times = np.asarray(self._tuning.leg_times, dtype=np.float64)
        self._leg_start_t = np.asarray(self._tuning.leg_start_times, dtype=np.float64)

        # Episode-local state.
        self._tick = 0
        self._finished = False
        self._active_leg: int | None = None
        self._reference = None
        self._reference_t_end = 0.0
        self._reference_t_start = 0.0
        self._planned_gate_pos = np.asarray(DEFAULT_GATE_POS, dtype=np.float64)
        self._planned_gate_rpy = np.asarray(DEFAULT_GATE_RPY, dtype=np.float64)
        self._last_action = np.array(
            [0.0, 0.0, 0.0, self._mass * self._gravity], dtype=np.float32
        )

        # PID with section-0 gains; will be swapped per sector.
        self._pid = PositionPid(self._tuning.section_gains[0])

        # Initial plan from constructor-time observation.
        self._frame_initial = self._parse_obs(obs)
        self._plan_sector(self._frame_initial)

    # ---- observation parsing ---------------------------------------------

    def _parse_obs(self, obs: dict) -> _ObservationFrame:
        target_gate = normalize_gate_index(obs["target_gate"])
        gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        return _ObservationFrame(
            target_gate=target_gate,
            gate_pos=gate_pos,
            gate_quat=gate_quat,
            pos=np.asarray(obs["pos"], dtype=np.float64).reshape(3),
            vel=np.asarray(obs["vel"], dtype=np.float64).reshape(3),
            quat=np.asarray(obs["quat"], dtype=np.float64).reshape(4),
        )

    # ---- replanning -------------------------------------------------------

    def _should_replan(self, frame: _ObservationFrame) -> bool:
        if self._reference is None:
            return True
        if self._active_leg != frame.target_gate:
            return True
        if frame.target_gate < 0 or frame.target_gate >= frame.gate_pos.shape[0]:
            return False
        tgt = frame.target_gate
        delta = float(np.linalg.norm(frame.gate_pos[tgt] - self._planned_gate_pos[tgt]))
        horiz_dist = float(np.linalg.norm(frame.gate_pos[tgt][:2] - frame.pos[:2]))
        return (
            delta > self._tuning.replan_gate_delta
            and horiz_dist < self._tuning.replan_horizontal_distance
        )

    def _plan_sector(self, frame: _ObservationFrame) -> None:
        tgt = frame.target_gate
        if tgt < 0 or tgt > 3:
            return
        # Section gain switch.
        gains = self._tuning.section_gains[tgt]
        if self._active_leg != tgt:
            self._pid.set_gains(gains)
        # Cache gate state used for replan-shift detection.
        self._planned_gate_pos = frame.gate_pos.copy()
        self._planned_gate_rpy = _gates_rpy(frame.gate_quat)
        t_start = float(self._leg_start_t[tgt])
        duration = float(self._leg_times[tgt])
        reference, t_end = build_reference_curve(
            tgt,
            self._planned_gate_pos,
            self._planned_gate_rpy,
            t_start=t_start,
            duration=duration,
            tuning=self._tuning.route,
            profile=self._tuning.speed_profiles[tgt],
            obstacles=self._obstacles,
            overrides=self._overrides,
        )
        self._reference = reference
        self._reference_t_start = t_start
        self._reference_t_end = float(t_end)
        self._active_leg = tgt

    # ---- main step --------------------------------------------------------

    def compute_control(self, obs: dict, info: dict | None = None) -> "NDArray[np.floating]":
        now = min(self._tick / self._env_freq, self._tuning.time_limit)
        frame = self._parse_obs(obs)
        if frame.target_gate == -1:
            self._finished = True
            return self._last_action
        if now >= self._tuning.time_limit:
            self._finished = True
            return self._last_action

        if self._should_replan(frame):
            self._plan_sector(frame)

        t_eval = float(np.clip(now, self._reference_t_start, self._reference_t_end))
        action = tracking_command(
            self._reference,
            pos=frame.pos,
            vel=frame.vel,
            quat=frame.quat,
            t_eval=t_eval,
            pid=self._pid,
            dt=self._dt,
            mass=self._mass,
            gravity=self._gravity,
            lateral_acc_limit=self._tuning.lateral_acc_limit,
            feedforward_scale=self._tuning.feedforward_scale,
        )
        self._last_action = action
        return action

    # ---- callbacks --------------------------------------------------------

    def step_callback(
        self,
        action: "NDArray[np.floating]",
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        tgt = normalize_gate_index(obs["target_gate"])
        if tgt == -1:
            self._finished = True
        return self._finished

    def episode_callback(self) -> None:
        self.reset()

    def episode_reset(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._tick = 0
        self._finished = False
        self._active_leg = None
        self._reference = None
        self._reference_t_end = 0.0
        self._reference_t_start = 0.0
        self._pid.reset()
        self._pid.set_gains(self._tuning.section_gains[0])
        self._last_action = np.array(
            [0.0, 0.0, 0.0, self._mass * self._gravity], dtype=np.float32
        )

    def render_callback(self, sim) -> None:
        if self._reference is None:
            return
        try:
            from lsy_drone_racing.utils.utils import draw_line
        except ImportError:
            return
        ts = np.linspace(self._reference_t_start, self._reference_t_end, 60)
        try:
            pts = np.asarray(self._reference(ts), dtype=np.float64)
            draw_line(sim, pts, rgba=np.array([0.2, 0.8, 0.2, 1.0]))
        except Exception:  # noqa: BLE001
            pass

    def diagnostic(self) -> dict:
        return {
            "tick": self._tick,
            "active_leg": self._active_leg,
            "reference_t_start": self._reference_t_start,
            "reference_t_end": self._reference_t_end,
            "finished": self._finished,
        }
