import unittest

import glfw
import numpy as np

from scripts import evaluate, play


class PlaybackTest(unittest.TestCase):
    def test_keyboard_requests_are_applied(self):
        command_shape = next(iter(play.PlaybackKeyboard.KEY_DELTAS.values())).shape

        class Env:
            commands = np.zeros(command_shape, dtype=np.float64)
            speed_min = -np.ones(command_shape)
            speed_max = np.ones(command_shape)

        env = Env()
        keyboard = play.PlaybackKeyboard(env, enable_velocity_commands=True)

        command_key = next(iter(play.PlaybackKeyboard.KEY_DELTAS))
        keyboard(None, command_key, None, glfw.PRESS, None)
        self.assertTrue(np.any(env.commands))
        self.assertTrue(np.all(env.commands >= env.speed_min))
        self.assertTrue(np.all(env.commands <= env.speed_max))

        keyboard(None, glfw.KEY_SPACE, None, glfw.PRESS, None)
        self.assertFalse(np.any(env.commands))

        keyboard(None, glfw.KEY_R, None, glfw.PRESS, None)
        self.assertTrue(keyboard.consume_reset_request())
        self.assertFalse(keyboard.consume_reset_request())

        keyboard(None, glfw.KEY_ESCAPE, None, glfw.PRESS, None)
        self.assertTrue(keyboard.quit_requested)

    def test_evaluation_closes_environment(self):
        class Model:
            def predict(self, observation, deterministic=True):
                return np.zeros_like(observation), None

        class Env:
            def __init__(self):
                self.closed = False

            def reset(self, seed=None):
                return np.zeros(1), {}

            def step(self, action):
                return np.zeros_like(action), 0.0, True, False, {}

            def close(self):
                self.closed = True

        env = Env()
        seed = int(np.random.SeedSequence().generate_state(1).item())
        results, _ = evaluate.evaluate_model(Model(), env, episodes=1, seed=seed)

        self.assertTrue(env.closed)
        self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
