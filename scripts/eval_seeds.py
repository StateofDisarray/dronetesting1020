"""Deterministic seeded evaluation harness for iterating on the level-2 controller.

Unlike ``scripts/evaluate.py`` (random seeds, aborts at >50% failure), this runs a
fixed list of seeds so failures are reproducible across code changes and prints a
per-seed breakdown (gates passed, flight time, finished). Render is always off.

Run as:

    $ python scripts/eval_seeds.py --config level2.toml --n 20
    $ python scripts/eval_seeds.py --controller attitude_pid_v4.py --seeds 0,1,2,3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.control.controller import Controller

logger = logging.getLogger(__name__)


def _run_one(env, controller_cls, config, seed: int) -> tuple[int, float, bool]:
    """Run a single seeded episode. Returns (gates_passed, flight_time, finished)."""
    obs, info = env.reset(seed=seed)
    controller: Controller = controller_cls(obs, info, config)
    i = 0
    n_gates = len(config.env.track.gates)
    curr_time = 0.0
    while True:
        curr_time = i / config.env.freq
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        finished_flag = controller.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated or finished_flag:
            break
        i += 1
    controller.episode_callback()
    controller.episode_reset()
    gates_passed = int(obs["target_gate"])
    if gates_passed == -1:
        gates_passed = n_gates
    finished = gates_passed == n_gates
    return gates_passed, curr_time, finished


def evaluate_seeds(
    config: str = "level2.toml",
    controller: str | None = None,
    n: int = 20,
    seeds: str | None = None,
) -> dict:
    """Run a fixed set of seeds and report a per-seed breakdown.

    Args:
        config: Config file name in ``config/``.
        controller: Controller file name in ``lsy_drone_racing/control/``; defaults to config.
        n: Number of seeds (0..n-1) when ``seeds`` is not given.
        seeds: Comma-separated explicit seed list (overrides ``n``).

    Returns:
        Summary dict with success rate, mean successful time, and per-seed records.
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or cfg.controller.file)
    controller_cls = load_controller(controller_path)

    if seeds is not None:
        # ``fire`` turns "4,8,10" into a tuple; a single value arrives as int.
        if isinstance(seeds, (list, tuple)):
            seed_list = [int(s) for s in seeds]
        elif isinstance(seeds, int):
            seed_list = [seeds]
        else:
            seed_list = [int(s) for s in str(seeds).split(",") if str(s).strip() != ""]
    else:
        seed_list = list(range(int(n)))

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)

    records = []
    n_gates = len(cfg.env.track.gates)
    for s in seed_list:
        gates, t, finished = _run_one(env, controller_cls, cfg, s)
        records.append({"seed": s, "gates": gates, "time": t, "finished": finished})
        flag = "OK " if finished else "FAIL"
        logger.info(f"[{flag}] seed={s:>3}  gates={gates}/{n_gates}  time={t:5.2f}s")
    env.close()

    n_total = len(records)
    n_ok = sum(r["finished"] for r in records)
    ok_times = [r["time"] for r in records if r["finished"]]
    fail_seeds = [r["seed"] for r in records if not r["finished"]]
    # Distribution of gate index at which non-finished runs stopped.
    fail_at = {}
    for r in records:
        if not r["finished"]:
            fail_at[r["gates"]] = fail_at.get(r["gates"], 0) + 1

    summary = {
        "n": n_total,
        "success_rate": n_ok / n_total if n_total else 0.0,
        "mean_time_ok": float(np.mean(ok_times)) if ok_times else None,
        "fail_seeds": fail_seeds,
        "fail_at_gate": dict(sorted(fail_at.items())),
        "records": records,
    }
    logger.info(
        "SUMMARY: success=%d/%d (%.0f%%)  mean_ok_time=%s  fail_at_gate=%s  fail_seeds=%s",
        n_ok,
        n_total,
        100.0 * summary["success_rate"],
        f"{summary['mean_time_ok']:.2f}s" if summary["mean_time_ok"] is not None else "n/a",
        summary["fail_at_gate"],
        fail_seeds,
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(evaluate_seeds, serialize=lambda _: None)
