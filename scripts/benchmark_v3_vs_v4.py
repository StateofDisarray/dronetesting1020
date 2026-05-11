"""Head-to-head benchmark between two controllers on level2.

Usage (inside the project's pixi env / docker sim container):

    pixi run python scripts/benchmark_v3_vs_v4.py
        [--n_runs 20] [--render False] [--config level2.toml]

Reports mean successful flight time, success rate, and per-run times for each
controller. Does NOT modify config files; works by passing the controller
name to ``scripts.sim.simulate`` directly, matching how ``scripts/evaluate.py``
selects a controller.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Iterable

import fire
import numpy as np
from sim import simulate

logger = logging.getLogger(__name__)

_DEFAULT_CONTROLLERS = ("gate_aware_fast_v3.py", "attitude_pid_v4.py")


def _summarize(name: str, ep_times: Iterable[float | None]) -> dict:
    times = list(ep_times)
    successes = [t for t in times if t is not None]
    n = len(times)
    n_ok = len(successes)
    out = {
        "controller": name,
        "n_runs": n,
        "n_success": n_ok,
        "success_rate": n_ok / n if n else 0.0,
        "mean_time_success": float(np.mean(successes)) if successes else float("nan"),
        "stdev_time_success": float(statistics.stdev(successes)) if len(successes) > 1 else 0.0,
        "min_time_success": float(np.min(successes)) if successes else float("nan"),
        "max_time_success": float(np.max(successes)) if successes else float("nan"),
        "times": times,
    }
    return out


def _print(summary: dict) -> None:
    print("---")
    print(f"controller     : {summary['controller']}")
    print(f"runs           : {summary['n_runs']}")
    print(f"success rate   : {summary['success_rate'] * 100:.1f}%")
    print(f"successes      : {summary['n_success']}/{summary['n_runs']}")
    print(
        "time (success) : "
        f"mean={summary['mean_time_success']:.3f}s "
        f"stdev={summary['stdev_time_success']:.3f}s "
        f"min={summary['min_time_success']:.3f}s "
        f"max={summary['max_time_success']:.3f}s"
    )
    formatted = ", ".join(
        f"{t:.2f}" if t is not None else "FAIL" for t in summary["times"]
    )
    print(f"per-run times  : [{formatted}]")


def main(
    config: str = "level2.toml",
    n_runs: int = 20,
    render: bool = False,
    controllers: tuple[str, ...] = _DEFAULT_CONTROLLERS,
) -> dict:
    """Run both controllers and print a side-by-side summary."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.WARNING)
    results = {}
    for ctrl in controllers:
        ctrl_path = Path(__file__).parents[1] / "lsy_drone_racing/control" / ctrl
        if not ctrl_path.exists():
            print(f"controller file missing: {ctrl_path}")
            continue
        print(f"\n=== running {ctrl} ({n_runs} runs, render={render}) ===")
        ep_times = simulate(config=config, controller=ctrl, n_runs=n_runs, render=render)
        results[ctrl] = _summarize(ctrl, ep_times)
        _print(results[ctrl])

    if len(results) == 2:
        a, b = controllers
        ra, rb = results[a], results[b]
        print("\n=== head-to-head ===")
        print(f"{a:30s} mean={ra['mean_time_success']:.3f}s  success={ra['success_rate']*100:.0f}%")
        print(f"{b:30s} mean={rb['mean_time_success']:.3f}s  success={rb['success_rate']*100:.0f}%")
        if np.isfinite(ra["mean_time_success"]) and np.isfinite(rb["mean_time_success"]):
            delta = rb["mean_time_success"] - ra["mean_time_success"]
            faster = a if delta > 0 else b
            print(f"faster (mean over successes): {faster} by {abs(delta):.3f}s")
    return results


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
