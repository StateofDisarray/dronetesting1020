"""Cascaded position/velocity PID controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec3(v: NDArray[np.floating] | list[float]) -> "NDArray[np.floating]":
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"expected length-3 vector, got size {a.size}")
    return a.copy()


_DEFAULT_OUTER_CLAMP = np.array([2.35, 2.35, 1.85], dtype=np.float64)
_DEFAULT_INNER_I_CLAMP = np.array([0.75, 0.75, 0.45], dtype=np.float64)
_DEFAULT_OUTPUT_CLAMP = np.array([3.2, 3.2, 4.2], dtype=np.float64)
_DEFAULT_DERIVATIVE_TAU = np.array([0.045, 0.045, 0.060], dtype=np.float64)


@dataclass
class PositionPidGains:
    """Gain set for the cascaded outer-position / inner-velocity PID loops."""

    outer_kp: "NDArray[np.floating]"
    outer_ki: "NDArray[np.floating]"
    outer_i_clamp: "NDArray[np.floating]"
    outer_clamp: "NDArray[np.floating]"
    inner_kp: "NDArray[np.floating]"
    inner_ki: "NDArray[np.floating]"
    inner_kd: "NDArray[np.floating]"
    inner_i_clamp: "NDArray[np.floating]"
    output_clamp: "NDArray[np.floating]"
    derivative_tau: "NDArray[np.floating]"

    @classmethod
    def from_xyz(
        cls,
        kp: NDArray[np.floating],
        ki: NDArray[np.floating],
        kd: NDArray[np.floating],
        i_clamp: NDArray[np.floating],
        outer_clamp: NDArray[np.floating] = _DEFAULT_OUTER_CLAMP,
        inner_i_clamp: NDArray[np.floating] = _DEFAULT_INNER_I_CLAMP,
        output_clamp: NDArray[np.floating] = _DEFAULT_OUTPUT_CLAMP,
        derivative_tau: NDArray[np.floating] = _DEFAULT_DERIVATIVE_TAU,
    ) -> "PositionPidGains":
        """Build a gain set from per-axis position gains and derived inner gains."""
        kp = _vec3(kp)
        ki = _vec3(ki)
        kd = _vec3(kd)
        inner_kp = np.maximum(1.04 * kd, 1e-9)
        outer_kp = 0.98 * kp / inner_kp
        outer_ki = 0.90 * ki / inner_kp
        inner_ki = 0.010 * inner_kp
        inner_kd = 0.012 * inner_kp
        return cls(
            outer_kp=outer_kp,
            outer_ki=outer_ki,
            outer_i_clamp=_vec3(i_clamp),
            outer_clamp=_vec3(outer_clamp),
            inner_kp=inner_kp,
            inner_ki=inner_ki,
            inner_kd=inner_kd,
            inner_i_clamp=_vec3(inner_i_clamp),
            output_clamp=_vec3(output_clamp),
            derivative_tau=_vec3(derivative_tau),
        )


@dataclass
class PositionPid:
    """Cascaded position/velocity PID controller with anti-windup and filtered derivative."""

    gains: PositionPidGains
    outer_integral: "NDArray[np.floating]" = field(default_factory=lambda: np.zeros(3))
    inner_integral: "NDArray[np.floating]" = field(default_factory=lambda: np.zeros(3))
    _filtered_derivative: "NDArray[np.floating]" = field(default_factory=lambda: np.zeros(3))
    _prev_vel_error: "NDArray[np.floating]" = field(default_factory=lambda: np.zeros(3))
    _first_sample: bool = True

    def reset(self) -> None:
        """Reset integrators, filtered derivative, and the first-sample flag."""
        self.outer_integral[:] = 0.0
        self.inner_integral[:] = 0.0
        self._filtered_derivative[:] = 0.0
        self._prev_vel_error[:] = 0.0
        self._first_sample = True

    def set_gains(self, gains: PositionPidGains) -> None:
        """Swap in new gains while clipping existing integrators to the new clamps."""
        # Preserve integrators but clip to new clamps.
        self.gains = gains
        self.outer_integral = np.clip(
            self.outer_integral, -gains.outer_i_clamp, gains.outer_i_clamp
        )
        self.inner_integral = np.clip(
            self.inner_integral, -gains.inner_i_clamp, gains.inner_i_clamp
        )

    def update(
        self, pos_error: "NDArray[np.floating]", vel_error: "NDArray[np.floating]", dt: float
    ) -> "NDArray[np.floating]":
        """Run one cascaded PID step and return the clamped acceleration command."""
        g = self.gains
        dt = max(float(dt), 1e-9)
        pe = _vec3(pos_error)
        ve = _vec3(vel_error)

        # ---- Outer loop ---------------------------------------------------
        proposed_outer_int = np.clip(
            self.outer_integral + pe * dt, -g.outer_i_clamp, g.outer_i_clamp
        )
        v_target_raw = g.outer_kp * pe + g.outer_ki * proposed_outer_int
        # Anti-windup: roll back per-axis if raw outer command saturates outer_clamp
        # in the same sign as pos_error.
        sat = np.abs(v_target_raw) > g.outer_clamp
        windup = sat & (np.sign(v_target_raw) == np.sign(pe))
        outer_int = np.where(windup, self.outer_integral, proposed_outer_int)
        # Recompute v_target with possibly-frozen integral, then clip output.
        v_target_raw = g.outer_kp * pe + g.outer_ki * outer_int
        v_target = np.clip(v_target_raw, -g.outer_clamp, g.outer_clamp)
        self.outer_integral = outer_int

        # ---- Inner loop ---------------------------------------------------
        velocity_error = ve + v_target

        # Filtered derivative on velocity_error
        alpha = dt / (np.maximum(g.derivative_tau, 0.0) + dt)
        if self._first_sample:
            raw_deriv = np.zeros(3, dtype=np.float64)
            self._first_sample = False
        else:
            raw_deriv = (velocity_error - self._prev_vel_error) / dt
        self._filtered_derivative = self._filtered_derivative + alpha * (
            raw_deriv - self._filtered_derivative
        )
        self._prev_vel_error = velocity_error.copy()

        proposed_inner_int = np.clip(
            self.inner_integral + velocity_error * dt, -g.inner_i_clamp, g.inner_i_clamp
        )
        cmd_raw = (
            g.inner_kp * velocity_error
            + g.inner_ki * proposed_inner_int
            + g.inner_kd * self._filtered_derivative
        )
        sat_in = np.abs(cmd_raw) > g.output_clamp
        windup_in = sat_in & (np.sign(cmd_raw) == np.sign(velocity_error))
        inner_int = np.where(windup_in, self.inner_integral, proposed_inner_int)
        cmd_raw = (
            g.inner_kp * velocity_error
            + g.inner_ki * inner_int
            + g.inner_kd * self._filtered_derivative
        )
        output = np.clip(cmd_raw, -g.output_clamp, g.output_clamp)
        self.inner_integral = inner_int
        return output
