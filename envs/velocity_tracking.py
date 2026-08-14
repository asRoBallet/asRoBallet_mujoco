import numpy as np

from .base import BaseAsRoBalletEnv


class VelocityTrackingEnv(BaseAsRoBalletEnv):
    """Track commanded planar velocity and yaw rate."""

    OBSERVATION_SIZE = 16
    MAX_EPISODE_STEPS = 1000
    INITIAL_YAW_RANGE = (-np.pi, np.pi)
    WINDOW_TITLE = "Velocity Tracking"

    def _reset_task_state(self):
        if self.randomize_commands:
            self.commands = self.sample_command()
        else:
            self.commands = np.zeros(3, dtype=np.float64)

    def _get_obs(self):
        self._update_derived_state()
        return np.concatenate(
            [
                self.data.sensordata[self.ball_velocity_slice][0:2],
                self.commands,
                self.rpy[0:2],
                self.COM * 10.0,
                self.data.sensordata[self.imu_gyro_slice]
                + self.rng.normal(0.0, 0.009, size=3),
                self.last_action,
            ]
        ).ravel().astype(np.float32, copy=False)

    def _compute_reward(self, action):
        angular_vel_penalty = -0.1 * np.sum(
            np.square(self.data.sensordata[self.imu_gyro_slice])
        )
        energy_penalty = -0.1 * np.sum(np.square(action))
        action_rate_penalty = -0.1 * np.sum(np.square(self.last_action - action))
        tracking_reward = 0.5 * np.exp(
            -np.sum(
                (
                    self.data.sensordata[self.ball_velocity_slice][0:2]
                    - self.commands[0:2]
                )
                ** 2
            )
            / 0.07
        ) + 0.5 * np.exp(
            -(self.data.qvel[self.base_dof_adr + 5] - self.commands[2]) ** 2 / 0.07
        )
        reward = (
            1.0
            + angular_vel_penalty
            + energy_penalty
            + action_rate_penalty
            + tracking_reward
        )
        return reward, {
            "position_reward": tracking_reward,
            "angular_vel_pen": angular_vel_penalty,
            "energy_pen": energy_penalty,
            "action_rate_pen": action_rate_penalty,
            "total_reward": reward,
        }

    def sample_command(self):
        n = 3
        y_k = self.np_random.uniform(
            low=self.speed_min, high=self.speed_max, size=(n,)
        )
        z_k = self.np_random.random(size=(n,)) < 0.5
        w_k = self.np_random.random(size=(n,)) < 0.5
        return self.commands[:n] - w_k * (self.commands[:n] - y_k * z_k)
