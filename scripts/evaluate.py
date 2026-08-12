import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train


RUN_DIRECTORY_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?$"
)


def default_model_path(task_name, log_root="logs"):
    task_dir = Path(log_root) / task_name
    if task_dir.is_dir():
        run_dirs = sorted(
            (
                path
                for path in task_dir.iterdir()
                if path.is_dir() and RUN_DIRECTORY_PATTERN.fullmatch(path.name)
            ),
            reverse=True,
        )
        for run_dir in run_dirs:
            candidate = run_dir / "checkpoints" / "best_model.zip"
            if candidate.is_file():
                return str(candidate)
    return str(task_dir / "best_by_eprew" / "best_model.zip")


def add_rollout_arguments(parser, default_episodes, include_episode_limits=True):
    parser.add_argument("task", choices=train.TASKS.keys(), help="Task to run.")
    parser.add_argument(
        "--xml-file",
        default="robots/mjcf/scene.xml",
        help="MuJoCo scene XML model path.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a saved PPO .zip model.",
    )
    if include_episode_limits:
        parser.add_argument(
            "--episodes",
            type=int,
            default=default_episodes,
            help="Number of episodes to run.",
        )
    parser.add_argument("--seed", type=int, default=3407, help="Initial random seed.")
    parser.add_argument("--device", default="cpu", help="PPO device.")
    if include_episode_limits:
        parser.add_argument(
            "--max-steps",
            type=int,
            default=None,
            help="Optional per-episode step limit.",
        )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions. This is the default.",
    )
    action_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample stochastic policy actions.",
    )


def finalize_rollout_args(parser, args):
    if args.model_path is None:
        args.model_path = default_model_path(args.task)
    if hasattr(args, "episodes") and args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    if (
        hasattr(args, "max_steps")
        and args.max_steps is not None
        and args.max_steps < 1
    ):
        parser.error("--max-steps must be at least 1.")
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained asRoBallet PPO policies headlessly."
    )
    add_rollout_arguments(parser, default_episodes=5)
    return finalize_rollout_args(parser, parser.parse_args())


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def make_env(
    task_name,
    xml_file,
    render_mode,
    randomize_upper_body=True,
    randomize_commands=True,
    enforce_time_limit=True,
    follow_robot_camera=False,
):
    env_cls = train.load_env_class(task_name)
    return env_cls(
        xml_file=xml_file,
        render_mode=render_mode,
        randomize_upper_body=randomize_upper_body,
        randomize_commands=randomize_commands,
        enforce_time_limit=enforce_time_limit,
        follow_robot_camera=follow_robot_camera,
    )


def evaluate_model(
    model,
    env,
    episodes,
    seed,
    deterministic=True,
    max_steps=None,
    render_delay=0.0,
    should_reset=None,
    should_stop=None,
    collect_statistics=True,
):
    episode_results = []
    reward_part_sums = defaultdict(list)
    stop_requested = False
    episode_idx = 0

    try:
        while episodes is None or episode_idx < episodes:
            episode_seed = seed + episode_idx
            obs, _ = env.reset(seed=episode_seed)

            total_reward = 0.0
            length = 0
            terminated = False
            truncated = False
            episode_parts = defaultdict(float)

            while not (terminated or truncated):
                if should_stop is not None and should_stop():
                    stop_requested = True
                    truncated = True
                    break
                if should_reset is not None and should_reset():
                    obs, _ = env.reset()
                    total_reward = 0.0
                    length = 0
                    episode_parts.clear()
                    continue
                if max_steps is not None and length >= max_steps:
                    truncated = True
                    break

                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                if collect_statistics:
                    total_reward += float(reward)
                length += 1

                if collect_statistics:
                    parts = (
                        info.get("reward_parts", {})
                        if isinstance(info, dict)
                        else {}
                    )
                    for key, value in parts.items():
                        try:
                            episode_parts[key] += float(value)
                        except (TypeError, ValueError):
                            continue

                if render_delay > 0.0:
                    time.sleep(render_delay)

            if collect_statistics:
                episode_results.append(
                    {
                        "episode": episode_idx + 1,
                        "reward": total_reward,
                        "length": length,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                for key, value in episode_parts.items():
                    reward_part_sums[key].append(value)
            if stop_requested:
                break
            episode_idx += 1
    finally:
        env.close()

    return episode_results, reward_part_sums


def print_results(results, reward_part_sums):
    for result in results:
        status = "terminated" if result["terminated"] else "truncated"
        print(
            f"Episode {result['episode']:>3}: "
            f"reward={result['reward']:.3f} "
            f"length={result['length']} "
            f"status={status}"
        )

    reward_summary = summarize_values([result["reward"] for result in results])
    length_summary = summarize_values([result["length"] for result in results])
    terminated_count = sum(1 for result in results if result["terminated"])
    truncated_count = sum(1 for result in results if result["truncated"])

    print("\nSummary")
    print(
        "  reward: "
        f"mean={reward_summary['mean']:.3f} "
        f"std={reward_summary['std']:.3f} "
        f"min={reward_summary['min']:.3f} "
        f"max={reward_summary['max']:.3f}"
    )
    print(
        "  length: "
        f"mean={length_summary['mean']:.1f} "
        f"std={length_summary['std']:.1f} "
        f"min={length_summary['min']:.0f} "
        f"max={length_summary['max']:.0f}"
    )
    print(f"  terminated={terminated_count} truncated={truncated_count}")

    if reward_part_sums:
        print("\nReward parts per episode")
        for key in sorted(reward_part_sums):
            summary = summarize_values(reward_part_sums[key])
            print(
                f"  {key}: "
                f"mean={summary['mean']:.3f} "
                f"std={summary['std']:.3f} "
                f"min={summary['min']:.3f} "
                f"max={summary['max']:.3f}"
            )


def load_model(model_path, device):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from stable_baselines3 import PPO

    return PPO.load(model_path, device=device)


def main():
    args = parse_args()
    print(f"Evaluating task={args.task} model={args.model_path}")
    print(f"episodes={args.episodes} deterministic={args.deterministic}")

    model = load_model(args.model_path, args.device)
    env = make_env(args.task, args.xml_file, render_mode="none")
    results, reward_part_sums = evaluate_model(
        model=model,
        env=env,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
    )
    print_results(results, reward_part_sums)


if __name__ == "__main__":
    main()
