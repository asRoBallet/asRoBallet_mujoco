import importlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ROOT / "robots"
MJCF = ROBOTS / "mjcf"
ENV_CLASSES = (
    ("envs.station_keeping", "StationKeepingEnv"),
    ("envs.velocity_tracking", "VelocityTrackingEnv"),
)
WHEELS = (
    "wheel_1_axle_joint",
    "wheel_2_axle_joint",
    "wheel_3_axle_joint",
)


def named_ids(model, object_type, names):
    return np.array(
        [mujoco.mj_name2id(model, object_type, name) for name in names],
        dtype=np.int32,
    )


class ZeroNoiseRng:
    def normal(self, _mean, _stddev, size):
        return np.zeros(size)


class EnvironmentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(dir=ROBOTS)
        temp_dir = Path(cls.temp_dir.name)

        robot = ET.parse(MJCF / "asRoBallet.xml")
        robot.getroot().find("compiler").set("meshdir", str(ROBOTS / "meshes"))
        actuator = robot.getroot().find("actuator")
        actuator[:] = list(actuator)[3:] + list(actuator)[:3]
        sensor = robot.getroot().find("sensor")
        sensor[:] = reversed(list(sensor))
        robot_path = temp_dir / "reordered.xml"
        robot.write(robot_path)

        scene = ET.parse(MJCF / "scene.xml")
        scene.getroot().find("include").set("file", robot_path.name)
        scene.getroot().find("contact").insert(
            0,
            ET.Element(
                "pair",
                name="base_link_floor",
                geom1="base_link_collision_geom",
                geom2="floor",
                condim="6",
                friction="0.2 0.2 0.01",
            ),
        )
        cls.scene_path = temp_dir / "scene.xml"
        scene.write(cls.scene_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_named_interfaces_survive_mjcf_reordering(self):
        action = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        for module_name, class_name in ENV_CLASSES:
            with self.subTest(module=module_name):
                env_class = getattr(importlib.import_module(module_name), class_name)
                env = env_class(
                    xml_file=str(self.scene_path)
                )
                try:
                    friction_before = env.model.dof_frictionloss.copy()
                    reset_obs, _ = env.reset(seed=3407)
                    step_obs, _, _, _, _ = env.step(action)

                    wheel_ids = named_ids(
                        env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEELS
                    )
                    np.testing.assert_allclose(env.data.ctrl[wheel_ids], action)
                    self.assertTrue(env.observation_space.contains(reset_obs))
                    self.assertTrue(env.observation_space.contains(step_obs))

                    env.data.sensordata[env.ball_velocity_slice] = [1.25, -2.5, 99.0]
                    env.data.sensordata[env.imu_gyro_slice] = [0.1, 0.2, 0.3]
                    if module_name == "envs.station_keeping":
                        env.data.sensordata[env.ball_position_slice] = [4.0, -5.0, 6.0]
                    env.rng = ZeroNoiseRng()
                    obs = env._get_obs()
                    np.testing.assert_allclose(obs[:2], [1.25, -2.5])
                    if module_name == "envs.station_keeping":
                        np.testing.assert_allclose(obs[2:4], [4.0, -5.0])
                        np.testing.assert_allclose(obs[11:14], [0.1, 0.2, 0.3])
                    else:
                        np.testing.assert_allclose(obs[2:5], env.commands)
                        np.testing.assert_allclose(obs[10:13], [0.1, 0.2, 0.3])

                    wheel_joint_ids = env.model.actuator_trnid[wheel_ids, 0]
                    expected_dofs = np.sort(env.model.jnt_dofadr[wheel_joint_ids])
                    changed_dofs = np.flatnonzero(
                        friction_before != env.model.dof_frictionloss
                    )
                    np.testing.assert_array_equal(changed_dofs, expected_dofs)

                    ball_pair = mujoco.mj_name2id(
                        env.model, mujoco.mjtObj.mjOBJ_PAIR, "ball_link_floor"
                    )
                    base_pair = mujoco.mj_name2id(
                        env.model, mujoco.mjtObj.mjOBJ_PAIR, "base_link_floor"
                    )
                    ball_friction = env.model.pair_friction[ball_pair, :2]
                    self.assertTrue(np.all((0.6 <= ball_friction) & (ball_friction <= 1.2)))
                    np.testing.assert_allclose(
                        env.model.pair_friction[base_pair, :2], [0.2, 0.2]
                    )
                finally:
                    env.close()

    def test_upper_body_pose_can_be_fixed_for_playback(self):
        from envs import StationKeepingEnv, VelocityTrackingEnv

        for env_class in (StationKeepingEnv, VelocityTrackingEnv):
            with self.subTest(env=env_class.__name__):
                env = env_class(render_mode="none", randomize_upper_body=False)
                try:
                    env.reset(seed=3407)
                    expected = np.zeros(env.upper_actuator_ids.size)
                    for side in ("right", "left"):
                        actuator_id = mujoco.mj_name2id(
                            env.model,
                            mujoco.mjtObj.mjOBJ_ACTUATOR,
                            f"{side}_elbow_joint",
                        )
                        actuator_position = np.flatnonzero(
                            env.upper_actuator_ids == actuator_id
                        ).item()
                        expected[actuator_position] = np.pi / 2.0
                    np.testing.assert_array_equal(
                        env.data.qpos[env.upper_qpos_indices],
                        expected,
                    )
                    np.testing.assert_array_equal(
                        env.data.ctrl[env.upper_actuator_ids],
                        expected,
                    )
                finally:
                    env.close()

    def test_velocity_playback_commands_reset_to_zero(self):
        from envs import VelocityTrackingEnv

        env = VelocityTrackingEnv(
            render_mode="none",
            randomize_commands=False,
        )
        try:
            for seed in (3407, None):
                env.reset(seed=seed)
                np.testing.assert_array_equal(env.commands, np.zeros(3))
        finally:
            env.close()

    def test_playback_camera_tracks_robot_front_for_both_tasks(self):
        from envs import StationKeepingEnv, VelocityTrackingEnv

        for env_class in (StationKeepingEnv, VelocityTrackingEnv):
            with self.subTest(env=env_class.__name__):
                env = env_class(render_mode="none", follow_robot_camera=True)
                try:
                    env.reset(seed=3407)
                    env._update_follow_camera()
                    scene = mujoco.MjvScene(env.model, maxgeom=1000)
                    mujoco.mjv_updateScene(
                        env.model,
                        env.data,
                        env.opt,
                        None,
                        env.camera,
                        mujoco.mjtCatBit.mjCAT_ALL,
                        scene,
                    )
                    camera_position = np.mean(
                        [scene.camera[0].pos, scene.camera[1].pos], axis=0
                    )
                    robot_position = env.data.xpos[env.robot_base_id]
                    robot_heading = np.array(
                        [np.cos(env.rpy[2]), np.sin(env.rpy[2])]
                    )
                    camera_offset = camera_position[:2] - robot_position[:2]

                    self.assertGreater(np.dot(camera_offset, robot_heading), 2.7)
                    self.assertEqual(
                        env.camera.type, mujoco.mjtCamera.mjCAMERA_TRACKING
                    )
                    self.assertEqual(env.camera.trackbodyid, env.robot_base_id)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
