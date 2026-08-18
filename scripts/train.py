import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASKS = {
    "velocity_tracking": {
        "env_module": "envs.velocity_tracking",
        "env_class": "VelocityTrackingEnv",
        "default_timesteps": 4_000_000,
    },
    "station_keeping": {
        "env_module": "envs.station_keeping",
        "env_class": "StationKeepingEnv",
        "default_timesteps": 4_000_000,
    },
}


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    tensorboard_dir: Path
    monitor_dir: Path
    checkpoint_dir: Path


def write_run_config(run_dir, args, total_timesteps):
    config = {
        "task": args.task,
        "xml_file": args.xml_file,
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "activation": args.activation,
        "total_timesteps": total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "clip_range": 0.1,
        "target_kl": 0.02,
        "checkpoint_freq": args.checkpoint_freq,
        "device": args.device,
        "vec_env": args.vec_env,
        "render_mode": args.render_mode,
    }
    config_path = Path(run_dir) / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def callback_save_freq(checkpoint_freq, n_envs):
    return max((checkpoint_freq + n_envs - 1) // n_envs, 1)


def create_run_paths(log_root, task_name, timestamp=None):
    task_dir = Path(log_root) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    base_name = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = 1
    run_dir = task_dir / base_name
    while True:
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            run_dir = task_dir / f"{base_name}_{suffix:02d}"
            suffix += 1

    tensorboard_dir = run_dir / "tensorboard"
    monitor_dir = run_dir / "monitor"
    checkpoint_dir = run_dir / "checkpoints"
    for directory in (tensorboard_dir, monitor_dir, checkpoint_dir):
        directory.mkdir()
    return RunPaths(run_dir, tensorboard_dir, monitor_dir, checkpoint_dir)


def make_callback_classes(base_callback_cls):
    class SaveBestOnRolloutEpRewMean(base_callback_cls):
        """
        Save the model whenever the rolling mean episodic return improves.
        Mirrors TensorBoard's rollout/ep_rew_mean, averaged over recent episodes.
        """
        def __init__(
            self,
            save_dir: str,
            name_prefix: str = "best_model",
            check_every_steps: int = 1000,
            verbose: int = 1,
        ):
            super().__init__(verbose)
            self.save_dir = save_dir
            self.name_prefix = name_prefix
            self.check_every_steps = int(check_every_steps)
            self.best_mean = -np.inf
            self._last_check = 0

        def _init_callback(self) -> None:
            os.makedirs(self.save_dir, exist_ok=True)

        def _maybe_compute_mean(self):
            buf = getattr(self.model, "ep_info_buffer", None)
            if buf is None or len(buf) == 0:
                return None
            rewards = [episode["r"] for episode in buf if "r" in episode]
            return float(np.mean(rewards)) if rewards else None

        def _on_step(self) -> bool:
            if (self.num_timesteps - self._last_check) < self.check_every_steps:
                return True
            self._last_check = self.num_timesteps

            mean_reward = self._maybe_compute_mean()
            if mean_reward is None:
                return True

            self.logger.record("best/rollout_ep_rew_mean_current", mean_reward)
            self.logger.record("best/rollout_ep_rew_mean_best", self.best_mean)

            if mean_reward > self.best_mean:
                self.best_mean = mean_reward
                path = os.path.join(self.save_dir, self.name_prefix)
                self.model.save(path)
                if self.verbose:
                    print(
                        f"[BestByEpRew] New best {mean_reward:.3f} "
                        f"at {self.num_timesteps:,} steps -> {path}.zip"
                    )
            return True

    class RewardPartsLogger(base_callback_cls):
        """
        Log info["reward_parts"] from each environment into TensorBoard.
        Also accumulates per-episode sums for each reward component.
        """
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.n_envs = 1
            self._keys = set()
            self._ep_sums = None
            self._ep_lens = None

        def _on_training_start(self) -> None:
            self.n_envs = self.training_env.num_envs
            self._ep_sums = {}
            self._ep_lens = np.zeros(self.n_envs, dtype=np.int64)

        def _ensure_keys(self, keys):
            new_keys = [key for key in keys if key not in self._keys]
            for key in new_keys:
                self._ep_sums[key] = np.zeros(self.n_envs, dtype=np.float32)
            self._keys.update(new_keys)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [False] * self.n_envs)

            for i in range(self.n_envs):
                parts = None
                if i < len(infos) and isinstance(infos[i], dict):
                    parts = infos[i].get("reward_parts")

                if parts:
                    self._ensure_keys(parts.keys())
                    for key, value in parts.items():
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            continue
                        self.logger.record(f"reward_parts/{key}", value)
                        self._ep_sums[key][i] += value

                self._ep_lens[i] += 1

                if dones[i]:
                    for key in self._keys:
                        self.logger.record(
                            f"episode_parts/{key}",
                            float(self._ep_sums[key][i]),
                        )
                        self._ep_sums[key][i] = 0.0
                    self.logger.record("episode/length", int(self._ep_lens[i]))
                    self._ep_lens[i] = 0

            return True

    return RewardPartsLogger, SaveBestOnRolloutEpRewMean


def parse_args():
    parser = argparse.ArgumentParser(description="Train asRoBallet PPO policies.")
    parser.add_argument("task", choices=TASKS.keys(), help="Task to train.")
    parser.add_argument(
        "--xml-file",
        default="robots/mjcf/scene.xml",
        help="MuJoCo scene XML model path.",
    )
    parser.add_argument("--log-root", default="logs", help="Root directory for logs.")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override task default.")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO batch size.")
    parser.add_argument(
        "--checkpoint-freq",
        type=positive_int,
        default=100_000,
        help="Global environment transitions between periodic checkpoints.",
    )
    parser.add_argument("--device", default="cuda", help="PPO device.")
    parser.add_argument(
        "--activation",
        choices=("elu", "tanh"),
        default="tanh",
        help="Policy/value network activation function.",
    )
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="dummy",
        help="Vectorized environment backend.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("none", "human"),
        default="none",
        help="Set to 'human' to open the MuJoCo GLFW viewer during training.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Shortcut for --render-mode human. Requires --n-envs 1.",
    )
    args = parser.parse_args()
    if args.gui:
        args.render_mode = "human"
    if args.render_mode == "human" and args.n_envs != 1:
        parser.error("--render-mode human / --gui requires --n-envs 1 to avoid opening multiple GLFW windows.")
    return args


def load_env_class(task_name):
    task = TASKS[task_name]
    module = importlib.import_module(task["env_module"])
    return getattr(module, task["env_class"])


def learn_and_save(model, env, callbacks, total_timesteps, checkpoint_dir):
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=callbacks,
        )
        model.save(Path(checkpoint_dir) / "final_model")
    finally:
        env.close()


def main():
    args = parse_args()
    from torch import nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    RewardPartsLogger, SaveBestOnRolloutEpRewMean = make_callback_classes(BaseCallback)
    env_cls = load_env_class(args.task)
    total_timesteps = args.total_timesteps or TASKS[args.task]["default_timesteps"]
    paths = create_run_paths(args.log_root, args.task)

    env_kwargs = dict(xml_file=args.xml_file, render_mode=args.render_mode)
    env = make_vec_env(
        lambda: env_cls(**env_kwargs),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(paths.monitor_dir),
        vec_env_cls={"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[args.vec_env],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=callback_save_freq(args.checkpoint_freq, args.n_envs),
        save_path=str(paths.checkpoint_dir),
        name_prefix="model",
    )
    callbacks = CallbackList([
        RewardPartsLogger(),
        SaveBestOnRolloutEpRewMean(
            save_dir=str(paths.checkpoint_dir),
            name_prefix="best_model",
        ),
        checkpoint_callback,
    ])

    activation_fn = {"elu": nn.ELU, "tanh": nn.Tanh}[args.activation]
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"activation_fn": activation_fn},
        clip_range=0.1,
        target_kl=0.02,
        verbose=0,
        batch_size=args.batch_size,
        device=args.device,
        tensorboard_log=None,
        seed=args.seed,
    )
    model.set_logger(configure(str(paths.tensorboard_dir), ["tensorboard"]))
    config_path = write_run_config(paths.run_dir, args, total_timesteps)

    print(
        f"Training task={args.task} for {total_timesteps:,} timesteps "
        f"activation={args.activation} vec_env={args.vec_env}"
    )
    print(f"Run directory: {paths.run_dir}")
    print(f"Run configuration: {config_path}")
    start = time.time()
    learn_and_save(
        model,
        env,
        callbacks,
        total_timesteps,
        paths.checkpoint_dir,
    )
    print("Training time cost:", time.time() - start)


if __name__ == "__main__":
    main()
