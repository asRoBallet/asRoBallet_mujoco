import glfw
import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


WHEEL_ACTUATOR_NAMES = (
    "wheel_1_axle_joint",
    "wheel_2_axle_joint",
    "wheel_3_axle_joint",
)
INITIAL_TILT_LIMIT_RAD = np.deg2rad(5.0)
TERMINATION_TILT_LIMIT_RAD = np.deg2rad(20.0)
PLAYBACK_UPPER_BODY_TARGETS = {
    "right_elbow_joint": np.pi / 2.0,
    "left_elbow_joint": np.pi / 2.0,
}


def _required_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MJCF is missing required {object_type.name}: {name}")
    return object_id


def _required_sensor_slice(model, name, expected_dim):
    sensor_id = _required_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    sensor_dim = model.sensor_dim[sensor_id]
    if sensor_dim != expected_dim:
        raise ValueError(
            f"Sensor {name!r} must have dimension {expected_dim}, got {sensor_dim}."
        )
    sensor_adr = model.sensor_adr[sensor_id]
    return slice(sensor_adr, sensor_adr + sensor_dim)


class BaseAsRoBalletEnv(gym.Env):
    """Shared MuJoCo lifecycle for the asRoBallet training tasks."""

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}
    OBSERVATION_SIZE = None
    MAX_EPISODE_STEPS = None
    INITIAL_YAW_RANGE = None
    WINDOW_TITLE = "asRoBallet"

    def __init__(
        self,
        frame_skip=5,
        xml_file="robots/mjcf/scene.xml",
        speed_min=-0.5,
        speed_max=0.5,
        render_mode="none",
        randomize_upper_body=True,
        randomize_commands=True,
        enforce_time_limit=True,
        follow_robot_camera=False,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = frame_skip * self.model.opt.timestep

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self._resolve_model_interface()
        self._setup_task_model()

        high = np.array([np.finfo(np.float32).max] * self.OBSERVATION_SIZE)
        self.observation_space = gym.spaces.Box(
            -high, high, shape=(self.OBSERVATION_SIZE,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.window = None
        self.context = None
        self.scene = None
        self.camera = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0

        self.max_episode_steps = self.MAX_EPISODE_STEPS
        self.current_step = 0
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.randomize_upper_body = randomize_upper_body
        self.randomize_commands = randomize_commands
        self.enforce_time_limit = enforce_time_limit
        self.follow_robot_camera = follow_robot_camera
        self.commands = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(3, dtype=np.float64)
        self.force = [0.0, 0.0, 0.0]
        self.rng = np.random.default_rng()
        self._configure_camera()

        if self.render_mode == "human":
            self._init_rendering()

    def _resolve_model_interface(self):
        self.wheel_actuator_ids = np.array(
            [
                _required_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in WHEEL_ACTUATOR_NAMES
            ],
            dtype=np.int32,
        )
        if not np.all(
            self.model.actuator_trntype[self.wheel_actuator_ids]
            == mujoco.mjtTrn.mjTRN_JOINT
        ):
            raise ValueError("Wheel actuators must use joint transmissions.")
        wheel_joint_ids = self.model.actuator_trnid[self.wheel_actuator_ids, 0]
        if not np.all(
            self.model.jnt_type[wheel_joint_ids] == mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise ValueError("Wheel actuators must drive hinge joints.")
        self.controlled_dof_indices = self.model.jnt_dofadr[wheel_joint_ids]

        wheel_actuator_id_set = set(self.wheel_actuator_ids.tolist())
        self.upper_actuator_ids = np.array(
            [
                actuator_id
                for actuator_id in range(self.model.nu)
                if actuator_id not in wheel_actuator_id_set
            ],
            dtype=np.int32,
        )
        if self.upper_actuator_ids.size != 10:
            raise ValueError(
                "The upstream task requires exactly 10 non-wheel upper-body actuators."
            )
        if not np.all(
            self.model.actuator_trntype[self.upper_actuator_ids]
            == mujoco.mjtTrn.mjTRN_JOINT
        ):
            raise ValueError("Upper-body actuators must use joint transmissions.")
        upper_joint_ids = self.model.actuator_trnid[self.upper_actuator_ids, 0]
        if not np.all(
            self.model.jnt_type[upper_joint_ids] == mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise ValueError("Upper-body actuators must drive hinge joints.")
        if not np.all(self.model.actuator_ctrllimited[self.upper_actuator_ids]):
            raise ValueError("Upper-body actuators must have finite control ranges.")
        self.upper_qpos_indices = self.model.jnt_qposadr[upper_joint_ids]

        base_joint_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if self.model.jnt_type[base_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("floating_base_joint must be a free joint.")
        self.base_qpos_adr = self.model.jnt_qposadr[base_joint_id]
        self.base_dof_adr = self.model.jnt_dofadr[base_joint_id]
        self.robot_base_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.ball_body_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball_link"
        )
        self.ball_velocity_slice = _required_sensor_slice(
            self.model, "ball_vel", expected_dim=3
        )
        self.imu_gyro_slice = _required_sensor_slice(
            self.model, "imu_gyro", expected_dim=3
        )
        self.ball_floor_pair_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_PAIR, "ball_link_floor"
        )

    def _setup_task_model(self):
        """Resolve task-specific MJCF objects after the common interface."""

    def _init_rendering(self):
        if not glfw.init():
            raise Exception("Could not initialize GLFW")

        self.window = glfw.create_window(1200, 1000, self.WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Could not create GLFW window")
        glfw.make_context_current(self.window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )

    def _configure_camera(self):
        mujoco.mjv_defaultCamera(self.camera)
        if self.follow_robot_camera:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.robot_base_id
            self.camera.distance = 2.8
            self.camera.elevation = -5.0
            self.camera.azimuth = 180.0
        else:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = _required_id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "robot_cam",
            )

    def _update_follow_camera(self):
        if self.follow_robot_camera:
            self.camera.azimuth = np.rad2deg(self.rpy[2]) + 180.0

    def _update_derived_state(self):
        base_quat_slice = slice(self.base_qpos_adr + 3, self.base_qpos_adr + 7)
        rotation = R.from_quat(
            self.data.qpos[base_quat_slice], scalar_first=True
        )
        self.rpy = rotation.as_euler("xyz")
        self.COM = self.data.xmat[self.robot_base_id].reshape(3, 3).T @ (
            self.data.subtree_com[self.robot_base_id]
            - self.data.subtree_com[self.ball_body_id]
        )

    def _get_obs(self):
        raise NotImplementedError

    def _reset_task_state(self):
        raise NotImplementedError

    def _compute_reward(self, action):
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)

        self._reset_task_state()
        self.last_action = np.zeros(3, dtype=np.float64)

        roll, pitch = self.rng.uniform(
            -INITIAL_TILT_LIMIT_RAD,
            INITIAL_TILT_LIMIT_RAD,
            size=2,
        )
        yaw = self.rng.uniform(*self.INITIAL_YAW_RANGE)
        quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7] = np.roll(
            quat, 1
        )

        self.data.qvel[self.base_dof_adr : self.base_dof_adr + 2] = self.rng.uniform(
            low=-0.5, high=0.5, size=2
        )
        self.data.qvel[
            self.base_dof_adr + 3 : self.base_dof_adr + 6
        ] = self.rng.uniform(low=-0.1, high=0.1, size=3)

        if self.randomize_upper_body:
            upper_targets = np.array(
                [
                    self.rng.uniform(
                        low=self.model.actuator_ctrlrange[actuator_id, 0],
                        high=self.model.actuator_ctrlrange[actuator_id, 1],
                    )
                    for actuator_id in self.upper_actuator_ids
                ],
                dtype=np.float64,
            )
        else:
            upper_targets = np.zeros(self.upper_actuator_ids.size, dtype=np.float64)
            upper_actuator_positions = {
                actuator_id: position
                for position, actuator_id in enumerate(self.upper_actuator_ids)
            }
            for actuator_name, target in PLAYBACK_UPPER_BODY_TARGETS.items():
                actuator_id = _required_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_name,
                )
                upper_targets[upper_actuator_positions[actuator_id]] = target
        self.data.qpos[self.upper_qpos_indices] = upper_targets
        self.data.ctrl[self.upper_actuator_ids] = upper_targets

        self.rand_dynamics()
        mujoco.mj_forward(self.model, self.data)
        observation = self._get_obs()
        self.render()
        return observation, {}

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[self.wheel_actuator_ids] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, reward_parts = self._compute_reward(action)
        self.last_action = action
        terminated = bool(
            abs(self.rpy[0]) > TERMINATION_TILT_LIMIT_RAD
            or abs(self.rpy[1]) > TERMINATION_TILT_LIMIT_RAD
        )
        truncated = bool(
            self.enforce_time_limit
            and self.current_step >= self.max_episode_steps
        )

        self.render()
        return obs, reward, terminated, truncated, {"reward_parts": reward_parts}

    def render(self):
        if self.render_mode != "human" or self.window is None:
            return
        if glfw.window_should_close(self.window):
            self.close()
            self.render_mode = "none"
            return

        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        self._update_follow_camera()
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)

        ball_velocity = self.data.sensordata[self.ball_velocity_slice]
        text = (
            f"Velocity_x: {ball_velocity[0]:=+.2f} | {self.commands[0]:.2f} m/s\n"
            f"Velocity_y: {ball_velocity[1]:.2f} | {self.commands[1]:.2f} m/s\n"
            f"Yaw: {self.data.qvel[self.base_dof_adr + 5]:.2f} | "
            f"{self.commands[2]:.2f} rad/s\n"
            f"Roll: {np.rad2deg(self.rpy[0]):.1f} degree\n"
            f"Pitch: {np.rad2deg(self.rpy[1]):.1f} degree"
        )
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_200,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            text,
            "",
            self.context,
        )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None
            self.context = None
            self.scene = None

    def rand_dynamics(self):
        sliding_friction = self.rng.uniform(low=0.6, high=1.2)
        self.model.pair_friction[self.ball_floor_pair_id, 0:2] = sliding_friction
        self.model.dof_frictionloss[self.controlled_dof_indices] = self.rng.uniform(
            low=0.08,
            high=0.12,
            size=self.controlled_dof_indices.shape,
        )
