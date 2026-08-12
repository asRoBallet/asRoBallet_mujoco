import argparse
from pathlib import Path
import sys

import glfw
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate


class PlaybackKeyboard:
    """GLFW key callback for playback reset, quit, and velocity commands."""

    COMMAND_STEP = 0.1
    KEY_DELTAS = {
        glfw.KEY_W: np.array([COMMAND_STEP, 0.0, 0.0]),
        glfw.KEY_S: np.array([-COMMAND_STEP, 0.0, 0.0]),
        glfw.KEY_A: np.array([0.0, COMMAND_STEP, 0.0]),
        glfw.KEY_D: np.array([0.0, -COMMAND_STEP, 0.0]),
        glfw.KEY_Q: np.array([0.0, 0.0, COMMAND_STEP]),
        glfw.KEY_E: np.array([0.0, 0.0, -COMMAND_STEP]),
    }

    def __init__(self, env, enable_velocity_commands):
        self.env = env
        self.enable_velocity_commands = enable_velocity_commands
        self.quit_requested = False
        self.reset_requested = False

    def consume_reset_request(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested

    def __call__(self, window, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            self.quit_requested = True
            if window is not None:
                glfw.set_window_should_close(window, True)
            return
        if key == glfw.KEY_R:
            self.reset_requested = True
            return
        if not self.enable_velocity_commands:
            return
        if key == glfw.KEY_SPACE:
            self.env.commands = np.zeros(3, dtype=np.float64)
            return
        if key not in self.KEY_DELTAS:
            return

        commands = np.asarray(self.env.commands, dtype=np.float64)
        self.env.commands = np.clip(
            commands + self.KEY_DELTAS[key],
            self.env.speed_min,
            self.env.speed_max,
        )


def install_keyboard_controls(env, task_name):
    if env.window is None:
        raise RuntimeError("Playback keyboard control requires a GLFW window.")

    keyboard = PlaybackKeyboard(
        env,
        enable_velocity_commands=task_name == "velocity_tracking",
    )
    glfw.set_key_callback(env.window, keyboard)
    return keyboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play trained asRoBallet PPO policies in the MuJoCo viewer.",
        epilog=(
            "keys: R reset, Esc quit; velocity_tracking also uses W/S vx, "
            "A/D vy, Q/E yaw rate, Space zero"
        ),
    )
    evaluate.add_rollout_arguments(
        parser,
        default_episodes=None,
        include_episode_limits=False,
    )
    return evaluate.finalize_rollout_args(parser, parser.parse_args())


def main():
    args = parse_args()
    print(f"Playing task={args.task} model={args.model_path}")
    print(f"continuous=True deterministic={args.deterministic}")

    model = evaluate.load_model(args.model_path, args.device)
    env = evaluate.make_env(
        args.task,
        args.xml_file,
        render_mode="human",
        randomize_upper_body=False,
        randomize_commands=False,
        enforce_time_limit=False,
        follow_robot_camera=True,
    )
    keyboard = install_keyboard_controls(env, args.task)
    if args.task == "velocity_tracking":
        print(
            "Keyboard: R reset, W/S vx, A/D vy, Q/E yaw rate, "
            "Space zero, Esc quit"
        )
    else:
        print("Keyboard: R reset, Esc quit")
    evaluate.evaluate_model(
        model=model,
        env=env,
        episodes=None,
        seed=args.seed,
        deterministic=args.deterministic,
        max_steps=None,
        render_delay=env.dt,
        should_reset=keyboard.consume_reset_request,
        should_stop=lambda: env.render_mode != "human"
        or keyboard.quit_requested,
        collect_statistics=False,
    )


if __name__ == "__main__":
    main()
