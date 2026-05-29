# How This Repository Works — A Guide Centered on `level2.toml`

This document explains what lives in this repository, how the pieces fit
together, and — in detail — how everything is driven by the
[`config/level2.toml`](../config/level2.toml) configuration file. It is meant
as a single, self-contained orientation for anyone picking up the project.

---

## 1. What this project is

This is the **LSY Autonomous Drone Racing** challenge codebase. The goal is to
fly a [Crazyflie](https://www.bitcraze.io/) quadrotor through a sequence of
gates as fast as possible while avoiding obstacles — first in a high-fidelity
simulator ([crazyflow](https://github.com/learnsyslab/crazyflow)) and
ultimately on real hardware.

The challenge is split into progressive **difficulty levels**, each described
by a TOML file in [`config/`](../config):

| Level | Randomized inertia | Randomized gates/obstacles | Random tracks | Theme |
| :---: | :----------------: | :------------------------: | :-----------: | :---- |
| `level0.toml` | No  | No  | No  | Perfect knowledge |
| `level1.toml` | Yes | No  | No  | Adaptive control |
| **`level2.toml`** | **Yes** | **Yes** | No | **Re-planning** |
| `level3.toml` | Yes | Yes | Yes | Online planning |

The **online competition always uses difficulty level 2**, which is why
`level2.toml` is the most important config file in the repo and the focus of
this guide.

---

## 2. Repository layout

```
dronetesting1020/
├── config/                       # TOML task definitions (one per difficulty level)
│   ├── level0.toml … level3.toml
│   ├── level2.toml               # ← the competition config
│   └── multi_level*.toml         # multi-drone variants
│
├── lsy_drone_racing/             # The Python package
│   ├── control/                  # Controllers (your code lives here)
│   │   ├── controller.py         # Abstract base class every controller subclasses
│   │   ├── attitude_pid_v4.py    # The active controller for level 2
│   │   ├── attitude_pid_v4/      # Supporting modules for that controller
│   │   │   ├── cascade_pid.py    # Cascaded position/velocity PID
│   │   │   ├── attitude.py       # Reference → attitude command conversion
│   │   │   ├── trajectory.py     # Per-sector waypoint & spline construction
│   │   │   ├── geometry.py       # Nominal track geometry + math helpers
│   │   │   ├── speed_profile.py  # Non-uniform knot scheduling
│   │   │   └── tuning.py         # All tunable constants in one place
│   │   └── …                     # Other example/experimental controllers
│   │
│   ├── envs/                     # Gymnasium environments (sim + real)
│   │   ├── race_core.py          # Core race logic, observation/action spaces
│   │   ├── drone_race.py         # Single-drone env "DroneRacing-v0"
│   │   ├── multi_drone_race.py   # Multi-drone env
│   │   └── real_race_env.py      # Real-hardware env (Vicon mocap)
│   │
│   └── utils/                    # Config loading, controller loading, helpers
│       └── utils.py              # load_config(), load_controller(), draw_line()
│
├── scripts/                      # Entry points
│   ├── sim.py                    # Run a controller in simulation
│   ├── deploy.py                 # Run on real hardware
│   └── …                         # benchmarking, track checks, etc.
│
├── tests/                        # Integration tests for controllers and envs
└── docs/                         # Sphinx documentation (this file lives here)
```

---

## 3. The big picture: how a race runs

The end-to-end flow when you run a simulation is:

```
scripts/sim.py
   │  1. load_config("config/level2.toml")  ──────────────┐
   │  2. load_controller(config.controller.file)          │  TOML drives
   │  3. gymnasium.make("DroneRacing-v0", **config.env…)   │  every step
   │                                                       │
   ▼                                                       ▼
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│  Environment (race_core.py) │◀──────▶│  Controller (attitude_pid_v4.py) │
│  - simulates physics        │  obs   │  - plans a trajectory             │
│  - randomizes gates/mass    │ ─────▶ │  - tracks it with cascaded PID    │
│  - reports observations     │        │  - returns an action              │
│  - checks gate passage      │ ◀───── │                                   │
└─────────────────────────────┘ action └──────────────────────────────────┘
```

The control loop in [`scripts/sim.py`](../scripts/sim.py) is:

```python
obs, info = env.reset()
controller = controller_cls(obs, info, config)     # config = level2.toml
while True:
    action = controller.compute_control(obs, info)  # decide what to do
    obs, reward, terminated, truncated, info = env.step(action)
    finished = controller.step_callback(action, obs, …)
    if terminated or truncated or finished:
        break
```

Each iteration runs at `env.freq` (50 Hz for level 2). The controller receives
an `obs` dictionary and must return an `action`.

### Observations (`obs`)

Defined by `build_observation_space` in
[`race_core.py`](../lsy_drone_racing/envs/race_core.py). The controller reads:

| Key | Meaning |
| :-- | :------ |
| `pos`, `vel`, `quat` | Drone position, velocity, orientation |
| `target_gate` | Index of the next gate to fly through (`-1` once finished) |
| `gates_pos`, `gates_quat` | Gate positions/orientations (nominal until in sensor range, then exact) |
| `obstacles_pos` | Obstacle positions (same nominal-vs-exact rule) |

### Actions

With `control_mode = "attitude"` (the level-2 setting), an action is
`[roll, pitch, yaw, collective_thrust]`. With `"state"` mode it would be a full
state command. The action space is built by `build_action_space`.

---

## 4. `level2.toml` section by section

Every value below is read by `load_config()`
([`utils.py`](../lsy_drone_racing/utils/utils.py)), which loads the TOML into an
`ml_collections.ConfigDict` so it is accessible as `config.env.freq`, etc.

### `[controller]`
```toml
file = "attitude_pid_v4.py"
```
Selects which file in `lsy_drone_racing/control/` to load as the controller.
`load_controller()` imports the module and finds the single class that
subclasses `Controller`. Passing `--controller <name>` to `scripts/sim.py`
overrides this.

### `[deploy]`
Only relevant on real hardware. Tells the deploy script whether to validate
gate/obstacle/start positions against the safety limits and whether physical
track objects are present.

### `[[deploy.drones]]`
The radio channel, ID, and model (`cf21B_500`) of the physical Crazyflie used
in the lab.

### `[sim]`
```toml
physics       = "first_principles"   # physics model fidelity
drone_model   = "cf21B_500"
freq          = 500                   # physics step rate (Hz)
attitude_freq = 500                   # simulated onboard attitude loop (Hz)
render        = true                  # show the GUI
camera        = -1                    # -1 → world view (see [[sim.cam_config]])
```
This block configures the crazyflow simulator. `physics` can range from a
full first-principles model to simplified identified models with rotor
dynamics and drag.

### `[env]`
```toml
id           = "DroneRacing-v0"   # which Gymnasium env to instantiate
seed         = -1                 # -1 → random; fixed int → reproducible
freq         = 50                 # control-loop frequency (Hz)
sensor_range = 0.7                # distance at which exact gate/obstacle poses appear
control_mode = "attitude"         # "attitude" or "state"
```
`sensor_range = 0.7` is central to **why level 2 needs re-planning**: gates and
obstacles report only their *nominal* positions until the drone gets within
0.7 m, at which point the *true* (randomized) position is revealed.

### `[env.track]`
Defines the nominal track — four gates and four obstacles — plus the drone's
start pose and the world safety limits.

```toml
randomize = false   # full-track randomization is OFF for level 2
```

- **Gates** — each has a `pos = [x, y, z]` and `rpy` (roll, pitch, yaw).
  Tall gates are 1.195 m, short gates 0.695 m; the 0.4 m opening sits inside a
  0.72 m frame.
- **Obstacles** — thin (0.03 m) cylinders 1.55 m tall.
- **Drone** — starts at `[-1.5, 0.75, 0.01]`.
- **Safety limits** — bounds that, if exceeded on real hardware, abort the run.

These nominal values are mirrored as the `DEFAULT_GATE_POS`,
`DEFAULT_GATE_RPY`, and `DEFAULT_OBSTACLES` constants in
[`geometry.py`](../lsy_drone_racing/control/attitude_pid_v4/geometry.py), so the
controller can plan an initial trajectory before it has seen anything.

### `[env.disturbances.*]` and `[env.randomizations.*]`
This is what makes level 2 *level 2*:

| Block | Effect |
| :---- | :----- |
| `disturbances.action` | Gaussian noise (σ = 0.001) added to every action |
| `disturbances.dynamics` | Uniform force disturbance ±0.1 |
| `randomizations.drone_pos` / `drone_rpy` | Start pose jitter |
| `randomizations.drone_mass` / `drone_inertia` | **Inertial randomization** (±5 g mass) |
| `randomizations.gate_pos` / `gate_rpy` | **Gates shift** up to ±0.15 m / ±0.2 rad |
| `randomizations.obstacle_pos` | **Obstacles shift** up to ±0.15 m |

Because gates and obstacles move each episode and are only revealed within
`sensor_range`, a controller that blindly follows a fixed precomputed path will
clip a gate or hit an obstacle. The level-2 controller must **re-plan** as it
discovers true positions.

---

## 5. How the active controller uses `level2.toml`

The controller named in the config —
[`attitude_pid_v4.py`](../lsy_drone_racing/control/attitude_pid_v4.py) — is the
`QualificationController`. Here is how it consumes the config and copes with the
randomization that `level2.toml` introduces.

### 5.1 Construction (reads the config)

```python
self._env_freq = float(config.env.freq)            # 50 Hz → dt = 0.02 s
params = load_params(config.sim.physics, config.sim.drone_model)
self._mass = float(params["mass"])                 # for thrust feedforward
```

It pulls the control frequency (to size its timestep) and the drone mass (for
gravity/thrust feedforward) straight out of the TOML. It then builds its tuning
bundle (`gate1_offset_tuning()`) and pre-plans the first sector from the initial
observation.

### 5.2 The track is divided into four "sectors"

The four gates define four legs. For each leg the controller:

1. **Builds waypoints** (`trajectory.build_route_waypoints`) — hand-tuned entry,
   exit, and shaping points expressed relative to each gate's axes
   (`geometry.gate_x_axis` / `gate_y_axis`).
2. **Schedules timing** (`speed_profile.schedule_knots`) — allocates the leg's
   time budget non-uniformly so the drone slows into tight turns and speeds up
   on straights. Leg durations come from `tuning.leg_times`.
3. **Fits a spline** (`CubicSpline` for early legs, `PchipInterpolator` for
   later ones) to produce a smooth time-parameterized reference curve.
4. **Checks obstacle clearance** (`trajectory._check_clearance`) — if the curve
   passes too close to that sector's obstacle, it inserts a "push" waypoint and
   re-fits, up to `MAX_AVOID_DEPTH` times. The clearance margins are tuned with
   the `±0.15 m` obstacle randomization from `level2.toml` in mind.

### 5.3 Re-planning (the level-2 essence)

On every step, `_should_replan()` decides whether to rebuild the current
sector's trajectory:

```python
delta = ‖observed_gate_pos − planned_gate_pos‖
horiz = ‖observed_gate_pos[:2] − drone_pos[:2]‖
replan if delta > replan_gate_delta and horiz < replan_horizontal_distance
```

In other words: once the drone is close enough to *see* the gate's true position
(within `sensor_range`) and that position differs from what was planned, it
rebuilds the curve through the real gate. It also swaps to that sector's PID
gains. This directly answers the randomization injected by
`[env.randomizations.gate_pos]` / `obstacle_pos`.

### 5.4 Tracking → attitude command

Each step, `attitude.tracking_command()`:

1. Evaluates the reference position/velocity/acceleration at the current time.
2. Computes position and velocity errors and feeds them to the **cascaded PID**
   (`cascade_pid.PositionPid`): an outer position loop produces a velocity
   target, an inner velocity loop (with filtered derivative and anti-windup)
   produces a force command.
3. Adds a mass-scaled acceleration feedforward and gravity compensation.
4. Converts the desired thrust vector into a `[roll, pitch, yaw, thrust]`
   attitude action — the format `level2.toml`'s `control_mode = "attitude"`
   expects.

### 5.5 Tuning lives in one place

[`tuning.py`](../lsy_drone_racing/control/attitude_pid_v4/tuning.py) centralizes
everything that is hand-calibrated for this specific level-2 track: per-sector
PID gains, leg time scales, speed profiles, gate entry/exit offsets, and
obstacle clearance triggers/margins/pushes. The comments there explain *why*
each value was chosen (e.g. stretching sector-3's time budget to give the tight
180° turn more headroom).

---

## 6. Running it

```bash
# Simulate the level-2 task with the controller named in the TOML:
python scripts/sim.py --config level2.toml

# Run several episodes (each re-randomizes gates/obstacles/mass):
python scripts/sim.py --config level2.toml --n_runs 10

# Override the controller without editing the TOML:
python scripts/sim.py --config level2.toml --controller attitude_mpc.py

# Headless (no GUI):
python scripts/sim.py --config level2.toml --render False
```

`sim.py` logs flight time, whether the run finished, and how many gates were
passed for each episode.

---

## 7. Mental model in one paragraph

`level2.toml` is the single source of truth for a race: it picks the controller,
configures the simulator and the Gymnasium environment, lays out the nominal
track, and — crucially — turns on the inertial, gate, and obstacle
**randomizations** that define difficulty level 2. The environment hides the
true (randomized) gate/obstacle positions until the drone is within
`sensor_range`. The `QualificationController` therefore plans an initial path
from the nominal geometry, then **re-plans each sector on the fly** as it sees
where the gates and obstacles really are, tracking each freshly built spline
with a cascaded PID that emits attitude commands. All the level-2-specific
numbers — gains, timings, clearances — are gathered in `tuning.py`.
