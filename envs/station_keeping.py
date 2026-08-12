import numpy as np

from .base import BaseAsRoBalletEnv, _required_sensor_slice


class StationKeepingEnv(BaseAsRoBalletEnv):
    """Keep the robot near its initial planar position and heading."""

    OBSERVATION_SIZE = 17
    MAX_EPISODE_STEPS = 2000
    INITIAL_YAW_RANGE = (-np.deg2rad(30.0), np.deg2rad(30.0))
    WINDOW_TITLE = "Station Keeping"

    def _setup_task_model(self):
        self.ball_position_slice = _required_sensor_slice(
            self.model, "ball_pos", expected_dim=3
        )

    def _reset_task_state(self):
        self.data.qpos[
            self.base_qpos_adr : self.base_qpos_adr + 2
        ] = self.rng.uniform(-0.5, 0.5, size=2)

    def _get_obs(self):
        self._update_derived_state()
        return np.concatenate(
            [
                self.data.sensordata[self.ball_velocity_slice][0:2],
                self.data.sensordata[self.ball_position_slice][0:2],
                self.rpy[0:2],
                [np.sin(self.rpy[2]), np.cos(self.rpy[2])],
                self.COM * 10.0,
                self.data.sensordata[self.imu_gyro_slice]
                + self.rng.normal(0.0, 0.009, size=3),
                self.last_action,
            ]
        ).ravel().astype(np.float32, copy=False)

    def _compute_reward(self, action):
        angular_vel_penalty = -0.001 * np.sum(
            np.square(self.data.sensordata[self.imu_gyro_slice])
        )
        torso_RP_penalty = -0.001 * np.sum(np.square(self.rpy[0:2]))
        energy_penalty = -0.01 * np.sum(np.square(action))
        action_rate_penalty = -0.01 * np.sum(np.square(self.last_action - action))

        ball_position = self.data.sensordata[self.ball_position_slice]
        position_reward = 1.0 - np.tanh(
            np.linalg.norm([ball_position[0], ball_position[1], self.rpy[2]])
        )
        reward = (
            1.0
            + angular_vel_penalty
            + torso_RP_penalty
            + energy_penalty
            + action_rate_penalty
            + position_reward
        )
        return reward, {
            "position_reward": position_reward,
            "angular_vel_pen": angular_vel_penalty,
            "torso_RP_pen": torso_RP_penalty,
            "energy_pen": energy_penalty,
            "action_rate_pen": action_rate_penalty,
            "total_reward": reward,
        }
