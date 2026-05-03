import argparse
import importlib
import os
import time

import numpy as np


TASKS = {
    "velocity_tracking": {
        "env_module": "velocity_tracking_env",
        "default_timesteps": 4_000_000,
    },
    "station_keeping": {
        "env_module": "station_keeping_env",
        "default_timesteps": 8_000_000,
    },
}


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
    parser.add_argument("--xml-file", default="asRoBallet.xml", help="MuJoCo XML model path.")
    parser.add_argument("--log-root", default="logs", help="Root directory for logs.")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override task default.")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO batch size.")
    parser.add_argument("--device", default="cpu", help="PPO device.")
    return parser.parse_args()


def load_env_class(task_name):
    module = importlib.import_module(TASKS[task_name]["env_module"])
    return module.MagicBallEnv


def main():
    args = parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.env_util import make_vec_env

    RewardPartsLogger, SaveBestOnRolloutEpRewMean = make_callback_classes(BaseCallback)
    env_cls = load_env_class(args.task)
    total_timesteps = args.total_timesteps or TASKS[args.task]["default_timesteps"]
    log_dir = os.path.join(args.log_root, args.task)
    best_dir = os.path.join(log_dir, "best_by_eprew")

    env_kwargs = dict(xml_file=args.xml_file, render_mode="none")
    env = make_vec_env(
        lambda: env_cls(**env_kwargs),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=log_dir,
    )

    callbacks = CallbackList([
        RewardPartsLogger(),
        SaveBestOnRolloutEpRewMean(save_dir=best_dir, name_prefix="best_model"),
    ])

    model = PPO(
        "MlpPolicy",
        env,
        clip_range=0.1,
        target_kl=0.02,
        verbose=0,
        batch_size=args.batch_size,
        device=args.device,
        tensorboard_log=log_dir,
        seed=args.seed,
    )

    print(f"Training task={args.task} for {total_timesteps:,} timesteps")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
    print("Training time cost:", time.time() - start)
    env.close()


if __name__ == "__main__":
    main()
