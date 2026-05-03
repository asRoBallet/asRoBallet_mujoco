import gymnasium as gym
import numpy as np
import mujoco
import glfw
from scipy.spatial.transform import Rotation as R

FLOOR_GEOM_ID = 0
TORSO_BASE_ID = 1
ACTUATOR_INDEX = np.array([14,25,36])

class MagicBallEnv(gym.Env):
    """Custom Gymnasium environment for the MagicBall MuJoCo model."""
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}
    
    def __init__(self, 
                frame_skip=5,
                xml_file="asRoBallet.xml", 
                speed_min=-0.5,
                speed_max=0.5,
                render_mode="none"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = frame_skip * 0.002

        # Action: 2D continuous (roll_motor, pitch_motor)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Obs: e.g. (roll_angle, pitch_angle, roll_vel, pitch_vel)
        high = np.array([np.finfo(np.float32).max]*16)
        self.observation_space = gym.spaces.Box(-high, high, shape=(16,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.context = None
        self.scene = None
        self.camera = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        # For stepping / episodes
        self.max_episode_steps = 1000  
        self.current_step = 0

        self.speed_min = speed_min
        self.speed_max = speed_max
        self.commands = np.array([0.0, 0.0, 0.0])
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.robot_base_id = 1
        self.force = [0,0,0]
        self.rng = np.random.default_rng()

        # If user requested human rendering, initialize a GLFW window
        if self.render_mode == "human":
            self._init_rendering()

    def _init_rendering(self):
        if not glfw.init():
            raise Exception("Could not initialize GLFW")

        self.window = glfw.create_window(1200, 1000, "MagicBallEnv", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Could not create GLFW window")
        glfw.make_context_current(self.window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Reasonable default camera setup
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'robot_cam')

    def _get_obs(self):
        r = R.from_quat(self.data.qpos[3:7], scalar_first=True)
        self.rpy = r.as_euler('xyz')  # Get Euler angles (roll, pitch, yaw) in radians
        self.COM = ( (self.data.xmat[self.robot_base_id].reshape(3, 3)).T @ (self.data.subtree_com[1]-self.data.subtree_com[-1]) )

        return np.concatenate(
            [
                self.data.sensordata[12:14],     # 2 linear velocity in robot's local frame
                self.commands,                   # 3
                self.rpy[0:2],                   # 2 body angle
                self.COM*10,                     # 3 center of mass, default value is [0.001, 0, 0.4471]
                self.data.sensordata[9:12] + self.rng.normal(0.0, 0.009, size=3),      # 3 body angular velocity in local frame
                self.last_action                 # 3 last action
            ]
        ).ravel()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)

        # Randomize the command speed each episode
        self.commands = self.sample_command()
        
        self.last_action = np.array([0.0, 0.0, 0.0])

        # Randomize initial roll pitch yaw
        roll, pitch = self.rng.uniform(-5/180*3.1415, 5/180*3.1415, size=2)
        yaw = self.rng.uniform(-np.pi, np.pi)
        quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        quat_mujoco = np.roll(quat, 1)  # reorder [x,y,z,w] → [w,x,y,z]
        self.data.qpos[3:7] = quat_mujoco

        # Randonmize initial velocity d(xyzrpy)=U(-0.5, 0.5)
        self.data.qvel[:2] = self.rng.uniform(low=-0.5, high=0.5, size=2)
        self.data.qvel[3:6] = self.rng.uniform(low=-0.1, high=0.1, size=3)

        # Randonmize initial 10 joint angles of arms and head
        for i in range(10):
            self.data.qpos[i+7] = self.rng.uniform(low=self.model.actuator_ctrlrange[i+3,0], high=self.model.actuator_ctrlrange[i+3,1])
            self.data.ctrl[i+3] = self.data.qpos[i+7]            
        
        self.rand_dynamics()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1
        
        action = np.clip(action, -1.0, 1.0)
        # Apply to the actuators
        self.data.ctrl[0] = float(action[0]) 
        self.data.ctrl[1] = float(action[1]) 
        self.data.ctrl[2] = float(action[2])

        # Step the simulation frame_skip times
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()

        angular_vel_penalty = - 0.1 * np.sum(np.square(self.data.sensordata[9:12]))
        
        # Energy related rewards
        energy_penalty = - 0.1 * np.sum(np.square(action))
        action_rate_penalty = - 0.1 * np.sum(np.square(self.last_action-action))

        tracking_reward = (
                  0.5 * np.exp( - np.sum( (self.data.sensordata[12:14]-self.commands[0:2])**2 ) / 0.07 )
                + 0.5* np.exp(-(self.data.qvel[5]-self.commands[2])**2/0.07)
        )
        reward = (
                # Base-related rewards
                1.0 

                + angular_vel_penalty
                + energy_penalty
                + action_rate_penalty
                + tracking_reward
        )

        info = {"reward_parts": {
            "position_reward": tracking_reward,
            "angular_vel_pen": angular_vel_penalty,
            "energy_pen": energy_penalty,
            "action_rate_pen": action_rate_penalty,
            "total_reward": reward
        }}

        self.last_action = action
        terminated = bool(abs(self.rpy[0]) > 20/180*3.1415 or abs(self.rpy[1]) > 20/180*3.1415)  # ~ 20 degrees
        truncated = (self.current_step >= self.max_episode_steps)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            return

        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        # Update scene
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None,
                               self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)

        # -------- Overlay info -------- #
        obs = self._get_obs()
        text = f"Velocity_x: {self.data.sensordata[12]:=+.2f} | {self.commands[0]:.2f} m/s\nVelocity_y: {self.data.sensordata[13]:.2f} | {self.commands[1]:.2f} m/s\nYaw: {self.data.qvel[5]:.2f} | {self.commands[2]:.2f} rad/s\nRoll: {self.rpy[0]/3.1415*180:.1f} degree\nPitch: {self.rpy[1]/3.1415*180:.1f} degree"
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_200,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            text, "",  # left-aligned text, no right column
            self.context
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

    def sample_command(self):
        N=3
        y_k = self.np_random.uniform(low=self.speed_min, high=self.speed_max, size=(N,))

        # Determine active dimensions based on probabilities cmd_b
        z_k = self.np_random.random(size=(N,)) < 0.5  # Boolean array

        # Randomly decide whether to update each dimension (50% chance each)
        w_k = self.np_random.random(size=(N,)) < 0.5  # Boolean array

        # Update the command based on the combination of y_k, z_k, and w_k
        x_kp1 = self.commands[:N] - w_k * (self.commands[:N] - y_k * z_k)

        return x_kp1

    def rand_dynamics(self):
        self.model.pair_friction[0][0] = self.rng.uniform(low=0.6, high=1.2)
        self.model.pair_friction[0][1] = self.rng.uniform(low=0.01, high=0.5)
        
        # Scale static friction.
        self.model.dof_frictionloss[ACTUATOR_INDEX] = self.rng.uniform(low=0.08, high=0.12, size=(3,))