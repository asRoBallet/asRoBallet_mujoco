import tempfile
import unittest
from pathlib import Path
from unittest import mock

import glfw
import numpy as np

from scripts import evaluate, play


class PlaybackTest(unittest.TestCase):
    def test_play_cli_has_no_episode_or_step_limit(self):
        with mock.patch("sys.argv", ["scripts/play.py", "station_keeping"]):
            args = play.parse_args()

        self.assertFalse(hasattr(args, "episodes"))
        self.assertFalse(hasattr(args, "max_steps"))

    def test_default_model_uses_newest_complete_run(self):
        with tempfile.TemporaryDirectory() as root:
            task_dir = Path(root) / "velocity_tracking"
            older = task_dir / "2026-08-12_10-00-00" / "checkpoints"
            newer = task_dir / "2026-08-12_11-00-00" / "checkpoints"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            expected = older / "best_model.zip"
            expected.touch()
            self.assertEqual(
                Path(evaluate.default_model_path("velocity_tracking", root)), expected
            )

            (newer / "best_model.zip").touch()
            self.assertEqual(
                Path(evaluate.default_model_path("velocity_tracking", root)),
                newer / "best_model.zip",
            )

    def test_velocity_keyboard_updates_clamps_resets_and_quits(self):
        class Env:
            commands = np.zeros(3, dtype=np.float64)
            speed_min = -0.5
            speed_max = 0.5

        env = Env()
        keyboard = play.PlaybackKeyboard(env, enable_velocity_commands=True)
        for key in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_Q):
            keyboard(None, key, None, glfw.PRESS, None)
        np.testing.assert_allclose(env.commands, [0.1, 0.1, 0.1])

        for _ in range(10):
            keyboard(None, glfw.KEY_W, None, glfw.REPEAT, None)
        np.testing.assert_allclose(env.commands, [0.5, 0.1, 0.1])

        keyboard(None, glfw.KEY_SPACE, None, glfw.PRESS, None)
        np.testing.assert_array_equal(env.commands, np.zeros(3))
        keyboard(None, glfw.KEY_W, None, glfw.RELEASE, None)
        np.testing.assert_array_equal(env.commands, np.zeros(3))
        keyboard(None, glfw.KEY_ESCAPE, None, glfw.PRESS, None)
        self.assertTrue(keyboard.quit_requested)

    def test_reset_key_is_consumed_once(self):
        class Env:
            commands = np.zeros(3, dtype=np.float64)
            speed_min = -0.5
            speed_max = 0.5

        env = Env()
        keyboard = play.PlaybackKeyboard(env, enable_velocity_commands=False)
        keyboard(None, glfw.KEY_W, None, glfw.PRESS, None)
        np.testing.assert_array_equal(env.commands, np.zeros(3))
        keyboard(None, glfw.KEY_R, None, glfw.PRESS, None)

        self.assertTrue(keyboard.consume_reset_request())
        self.assertFalse(keyboard.consume_reset_request())

        class Model:
            def predict(self, _obs, deterministic=True):
                return np.zeros(1), None

        class RolloutEnv:
            def __init__(self):
                self.reset_seeds = []
                self.closed = False

            def reset(self, seed=None):
                self.reset_seeds.append(seed)
                return np.zeros(1), {}

            def step(self, _action):
                return np.zeros(1), 1.0, True, False, {}

            def close(self):
                self.closed = True

        rollout_env = RolloutEnv()
        reset_checks = iter((True, False))
        results, _ = evaluate.evaluate_model(
            Model(),
            rollout_env,
            episodes=1,
            seed=3407,
            should_reset=lambda: next(reset_checks),
        )
        self.assertEqual(rollout_env.reset_seeds, [3407, None])
        self.assertTrue(rollout_env.closed)
        self.assertEqual(results[0]["length"], 1)

    def test_stop_request_closes_rollout_before_policy_step(self):
        class Model:
            def predict(self, *_args, **_kwargs):
                raise AssertionError("policy must not run after a stop request")

        class Env:
            closed = False

            def reset(self, seed=None):
                return np.zeros(1), {}

            def close(self):
                self.closed = True

        env = Env()
        results, _ = evaluate.evaluate_model(
            Model(), env, episodes=1, seed=3407, should_stop=lambda: True
        )
        self.assertTrue(env.closed)
        self.assertEqual(results[0]["length"], 0)
        self.assertTrue(results[0]["truncated"])

    def test_continuous_rollout_resets_after_termination(self):
        class Model:
            def predict(self, _obs, deterministic=True):
                return np.zeros(1), None

        class Env:
            def __init__(self):
                self.reset_seeds = []
                self.steps = 0
                self.closed = False

            def reset(self, seed=None):
                self.reset_seeds.append(seed)
                return np.zeros(1), {}

            def step(self, _action):
                self.steps += 1
                return np.zeros(1), 0.0, True, False, {}

            def close(self):
                self.closed = True

        env = Env()
        results, _ = evaluate.evaluate_model(
            Model(),
            env,
            episodes=None,
            seed=3407,
            should_stop=lambda: env.steps >= 1,
        )

        self.assertEqual(env.reset_seeds, [3407, 3408])
        self.assertTrue(env.closed)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["terminated"])
        self.assertTrue(results[1]["truncated"])


if __name__ == "__main__":
    unittest.main()
