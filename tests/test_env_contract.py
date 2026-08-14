import importlib
import unittest

import numpy as np

from scripts import train


class EnvironmentContractTest(unittest.TestCase):
    def test_registered_environments_reset_and_step(self):
        for task_name, task in train.TASKS.items():
            with self.subTest(task=task_name):
                module = importlib.import_module(task["env_module"])
                env_class = getattr(module, task["env_class"])
                env = env_class(render_mode="none")
                try:
                    observation, info = env.reset()
                    self.assertTrue(env.observation_space.contains(observation))
                    self.assertIsInstance(info, dict)

                    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                    observation, reward, terminated, truncated, info = env.step(action)
                    self.assertTrue(env.observation_space.contains(observation))
                    self.assertTrue(np.isfinite(reward))
                    self.assertIsInstance(terminated, bool)
                    self.assertIsInstance(truncated, bool)
                    self.assertIsInstance(info, dict)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
