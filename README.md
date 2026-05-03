# asRoBallet MuJoCo

`asRoBallet_mujoco` is a MuJoCo-based reinforcement learning project for a ballbot-style robot with an omni-wheel drive mechanism. The repository contains the robot model, mesh assets, two Gymnasium environments, and a shared PPO training entry point.

The robot model is defined in `asRoBallet.xml`. It includes the main body, upper-body links, a ball, three omni-wheel actuators, onboard sensor sites, and STL mesh assets under `meshes/`.

## Project Structure

```text
.
├── asRoBallet.xml              # MuJoCo robot model
├── meshes/                     # STL mesh assets referenced by the XML
├── velocity_tracking_env.py    # Velocity-tracking Gymnasium environment
├── station_keeping_env.py      # Station-keeping Gymnasium environment
├── train.py                    # Shared PPO training script for both tasks
├── README.md
└── LICENSE
```

## Tasks

### Velocity Tracking

`velocity_tracking_env.py` trains the robot to follow commanded planar velocity and yaw-rate targets.

- Action space: 3 continuous wheel commands.
- Observation size: 16.
- Default episode length: 1000 environment steps.
- Default training horizon: 4,000,000 PPO timesteps.
- Reward terms include velocity tracking, angular-velocity penalty, action energy, and action-rate penalty.

### Station Keeping

`station_keeping_env.py` trains the robot to remain near its initial position and heading.

- Action space: 3 continuous wheel commands.
- Observation size: 17.
- Default episode length: 2000 environment steps.
- Default training horizon: 8,000,000 PPO timesteps.
- Reward terms include position/yaw retention, roll-pitch penalty, angular-velocity penalty, action energy, and action-rate penalty.

## Installation

Create and activate a virtual environment:

```bash
conda create -n asroballet python=3.11
conda activate asroballet
pip install gymnasium numpy mujoco glfw scipy stable-baselines3 tensorboard tqdm rich
```

## Training

Use the shared training script and pass the task name.

Train the velocity-tracking task:

```bash
python train.py velocity_tracking
```

Train the station-keeping task:

```bash
python train.py station_keeping
```

Useful options:

```bash
python train.py velocity_tracking --total-timesteps 100000
python train.py station_keeping --n-envs 4 --seed 3407
python train.py velocity_tracking --xml-file asRoBallet.xml --log-root logs
```

Show all options:

```bash
python train.py --help
```

Training logs and best models are written under:

```text
logs/<task_name>/
└── best_by_eprew/
    └── best_model.zip
```

TensorBoard can be launched with:

```bash
tensorboard --logdir logs
```

## Notes

- The MuJoCo timestep in `asRoBallet.xml` is `0.002` seconds.
- The environments use `frame_skip=5`, so one policy step advances `0.01` seconds of simulation time.
- Both tasks control only the first three actuators, corresponding to the three omni-wheel motors.
- Head and arm joints are position-controlled by the XML actuators and randomized during some resets.
