import gymnasium as gym
import robosuite
import numpy as np

from robosuite import load_controller_config
from robosuite.robots.robot import Robot
from gymnasium.envs.registration import register

# from scripts.utils.wrapper import MDP

class CompLiftEnv(gym.Env):
    def __init__(self, robot="Panda", baseline_mode=False, training_mode=False, verbose=True, normalize_reward=False):
        self._controller_config = load_controller_config(default_controller='OSC_POSITION')
        self._robot = robot
        self._verbose = verbose
        self._normalize_reward = normalize_reward
        # Subtasks to complete the lift task (in order)
        self.tasks = ['reach', 'grasp', 'lift']
        
        self.current_task = self.tasks[0]

        # Create the default robosuite lift environment
        self._env = robosuite.make(
            env_name="Lift",
            robots=self._robot,
            controller_configs=self._controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=100,
            # reward_scale=1 if self._normalize_reward else None,
        )
        
        # The initial joint configuration of the robot depends on the subtask
        self._robot_init_qpos = {self.current_task: self._env.robots[0].init_qpos}

        # hard-coded (for IIWA)
        # self._robot_init_qpos['grasp'] = [0.01231588, 0.99710506, -0.02401533, -1.51316928, 0.03385481, 0.6337215, -0.02739373]
        # self._robot_init_qpos['lift'] = [0.01231588, 0.99710506, -0.02401533, -1.51316928, 0.03385481, 0.6337215, 0.89631194]

        # init joint positions for Panda (hard-coded)
        # self._robot_init_qpos['grasp'] = [-0.02296645, 0.87523372, -0.00537831, -2.07363196, 0.02362597, 2.94811447, 0.73588951]
        # self._robot_init_qpos['lift'] = [-0.023814369, 0.937344171, 0.000128018001, -1.96319784, 0.000138185644, 2.89760903, 0.764912129]

        if self._verbose:
            print(f'{self.current_task} initial joint configuration: {self._robot_init_qpos[self.current_task]}')
        
        # Action and observation spaces for compositional training
        # For baseline training, see action_space() and observation_space()
        self.action_spaces = {
            'reach': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32), # (x,y,z)
            'grasp': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32), # (z,g)
            'lift': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32), # (z,g)
        }
        self.observation_spaces = {
            'reach': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
            'grasp': gym.spaces.Box(low=-5, high=5, shape=(3,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
        }

        self.reward_criteria = None
        # Reset params
        self.setup_skip_reset_once = False
        self.fresh_reset = False
        # In baseline mode, we train all tasks at once, so training_mode=False
        self.baseline_mode = baseline_mode
        # In training mode, we train one subtask at a time.
        self.training_mode = False if self.baseline_mode else training_mode 

    @property
    def action_space(self):
        if self.baseline_mode:
            # np.shape=(4,) since OSC_POSE takes 3 inputs + 1 input for gripper
            return gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            return self.action_spaces[self.current_task]
    
    @property
    def observation_space(self):
        if self.baseline_mode:
            # np.shape=(7,) since the robot arm has 7 actuated DOFs
            return gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
        else:
            return self.observation_spaces[self.current_task]

    def _get_obs(self):
        obs = self._env._get_observations() 
        
        hand_pos = obs['robot0_eef_pos']
        cube_pos = obs['cube_pos']
        gripper = obs['robot0_gripper_qpos'][0]

        if self.baseline_mode:
            # (hand_position, cube_position, gripper); np.shape(obs) == (7,)
            obs = np.concatenate([hand_pos, cube_pos, [gripper]]).astype(np.float32)
            return obs

        if self.current_task == 'reach':
            # (hand_position, cube_position); np.shape(obs) == (6,)
            obs = np.concatenate([hand_pos, cube_pos]).astype(np.float32)
        
        elif self.current_task == 'grasp':
            # (hand_z_position, cube_z_position, gripper); np.shape(obs) == (3,)
            obs = np.concatenate([hand_pos[2:3], cube_pos[2:3], [gripper]]).astype(np.float32)

        elif self.current_task == 'lift':
            # (hand_z_position, cube_z_position, gripper); np.shape(obs) == (3,)
            obs = np.concatenate([hand_pos[2:3], cube_pos[2:3], [gripper]]).astype(np.float32)

        else:
            raise RuntimeError('Invalid task')
        return obs

    def _process_action(self, action):
        # These are deltas (see robosuite/controllers/osc.py)
        if self.baseline_mode:
            # Control eef x,y,z pos and gripper
            return action
        if self.current_task == 'reach':
            # Control eef x,y,z pos only
            action = np.concatenate([action[:3], [-1]])
        elif self.current_task == 'grasp':
            # Control eef z pos and gripper only
            action = np.concatenate([[0, 0], action])

        elif self.current_task == 'lift':
            # Control eef z pos and gripper only
            action = np.concatenate([[0, 0], action])

        else:
            raise RuntimeError('Invalid task')

        assert len(action) == 4, f'action for task {self.current_task} is shape {np.shape(action)}. Expected (4,). The action is {action}'

        return action
     
    def _compute_reward_criteria(self, observation):
        hand_pos = observation['robot0_eef_pos'].copy()
        cube_pos = observation['cube_pos'].copy()
        table_height = self._env.model.mujoco_arena.table_offset[2]

        grasp_height = table_height + 0.01
        lift_height = table_height + 0.05

        reach_dist = np.linalg.norm(cube_pos - hand_pos)
        grasp_dist = np.linalg.norm(np.concatenate([cube_pos[:2],[grasp_height]]) - cube_pos)
        lift_dist = np.linalg.norm(np.concatenate([cube_pos[:2],[lift_height]])-cube_pos)
        check_grasp = self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=self._env.cube)
       
        criteria = {
            'reach_dist': reach_dist,
            'cube_height': cube_pos[2],
            'grasp_dist': grasp_dist,
            'lift_dist': lift_dist,
            'lift_height': lift_height,
            'check_grasp': check_grasp,
        }
        return criteria

    def _evaluate_task(self, observation):
        task_completed = False
        task_failed = False
        new_reward_criteria = self._compute_reward_criteria(observation)
        if self.baseline_mode:
            """
                Using robosuite lift environment reward function:
                    
                Sparse reward of 2.25 if cube is lifted

                Summed reward shaping ([] for continuous, {} for discrete):
                - Reaching in [0, 1], to encourage the arm to reach the cube
                - Grasping in {0, 0.25}, non-zero if arm is grasping the cube
                - Lifting in {0, 1}, non-zero if arm has lifted the cube          
            """
            if self._env._check_success():
                # check if cube is lifted
                task_completed = True
                task_reward = 2.25
            else:
                task_reward = 1 - np.tanh(10.0 * new_reward_criteria['reach_dist'])
                if self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=self._env.cube): 
                    # check if cube is grasped
                    task_reward += 0.25
        else:
            """
                Compositional mode: separate reward functions for each subtask
                - Reach: {[0,1], 2}, to encourage arm to reach the cube with sparse completion reward = 2
                - Grasp: {0, 0.25}, non-zero if arm is grasping the cube
                - Lift: {0, 2}, non-zero if arm has lifted the cube
            """
            if self.current_task == 'reach':
                task_reward = 1 - np.tanh(10.0 * new_reward_criteria['reach_dist'])
                if new_reward_criteria['reach_dist'] < 0.01:
                    task_completed = True
                    task_reward = 2
            
            elif self.current_task == 'grasp':
                # task_reward = 1 - np.tanh(10.0 * new_reward_criteria['grasp_dist'])
                # task_reward = 1 - np.tanh(10.0 * new_reward_criteria['reach_dist'])
                task_reward = 0
                if new_reward_criteria['check_grasp']:
                    task_completed = True
                    task_reward += 0.25
            
            elif self.current_task == 'lift':
                # task_reward = 1 - np.tanh(10.0 * new_reward_criteria['lift_dist'])
                task_reward = 0
                if new_reward_criteria['cube_height'] > new_reward_criteria['lift_height']: 
                    # task is completed when cube is lifted
                    task_completed = True
                    task_reward = 2.25

            else:
                raise RuntimeError('Invalid subtask')
        
        if self._normalize_reward: 
            reward /= 2.25
        
        self.reward_criteria = new_reward_criteria
        return task_reward, task_completed, task_failed

    def step(self, action):
        self.fresh_reset = False
        task_action = self._process_action(action)
        observation, reward, done, info = self._env.step(task_action)
        task_reward, task_completed, task_failed = self._evaluate_task(observation)

        terminated = done
        truncated = task_failed or terminated

        info['task_success'] = False                            # entire task success

        # If baseline_mode, always reset current_task to first task (reach)
        if self.baseline_mode:
            self.fresh_reset = True

        # Note: in training mode, we only train one subtask at a time.
        if task_completed and (not self.training_mode):
            if self.current_task == self.tasks[len(self.tasks) - 1]:
                # Completed final subtask (lift), so completely reset environment 
                # (set current_task to first subtask)
                self.fresh_reset = True
                info['task_success'] = True
            else:
                if self._verbose:
                    print(f"Completed subtask {self.current_task}, moving on to {self.tasks[self.tasks.index(self.current_task) + 1]}")
                # Compositional training mode:
                # completed current subtask so moving on to the next subtask
                self.current_task = self.tasks[self.tasks.index(self.current_task) + 1]


        # If completed current subtask, save the final q_pos to later set as init q_pos for next subtask
        # Skip when current task is last subtask
        if self.current_task != self.tasks[-1] and not self.baseline_mode:
            next_task = self.tasks[self.tasks.index(self.current_task) + 1]
            if task_completed:
                # Check if robot_init_qpos for next_task has been set. If not, set it to final qpos of previous task
                if self._robot_init_qpos.get(next_task) is None:
                    self._robot_init_qpos[next_task] = np.arctan2(
                        observation['robot0_joint_pos_sin'],
                        observation['robot0_joint_pos_cos'],
                    )
                # if self._verbose:
                #     print(f"Setting initial joint configuration for subtask {self.current_task} to:", self._robot_init_qpos[self.current_task])
        
        # Get obs (this depends on self.current_task, which may be updated above)
        obs = self._get_obs()

        # Log step statistics
        info['current_task'] = self.current_task                # current task
        info['task_truncated'] = truncated                      # step terminated
        info['task_reward'] = task_reward                       # task reward
        info['current_task_obs'] = obs                          # task observation
        info['current_task_action'] = task_action               # task action
        info['reward_criteria'] = self.reward_criteria          # task reward criteria (used for rwd fn)
        info['observation'] = observation                       # robosuite observation
        info['action'] = action                                 # robosuite action
        info['is_success'] = True if task_completed else False  # subtask success 

        return obs, task_reward, terminated, task_failed, info

    def _setup_skip_reset(self):
        self.setup_skip_reset_once = True
    
    def _reset_robot_init_qpos(self):
        if self.baseline_mode:
            init_qpos = self._robot_init_qpos[self.tasks[0]]
        else:
            init_qpos = self._robot_init_qpos[self.current_task]
        return init_qpos

    def reset(self, seed=None, options=None):
        if self.setup_skip_reset_once:
            self.setup_skip_reset_once = False
        else:
            if self.fresh_reset:
                # Reset current task to first task (reach)
                if self._verbose:
                    print(f'Called reset right after task switch to {self.current_task}.')
                self.current_task = self.tasks[0]            
            # Reset the robosuite environment
            self._env.reset()
            if self.training_mode:
                # Reset initial joint configuration to correct init_qpos based on subtask
                self._env.robots[0].init_qpos = self._reset_robot_init_qpos()
                # Reset robot to reset to new init_qpos
                self._env.robots[0].reset(deterministic=True)

        obs = self._get_obs()
        self.reward_criteria = self._compute_reward_criteria(self._env._get_observations())

        return obs, {'current_task': self.current_task,
                     'current_task_obs': self._get_obs()}

    def render(self, *args, **kwargs):
        self._env.render()

    def close(self) -> None:
        self._env.close()
        return super().close()


def register_envs():
    """
        Register custom gymnasium environments
    """
    register(
        id="CompLift-Panda",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'Panda', 'verbose': False}
    )
    register(
        id="BaselineLift-Panda",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'Panda', 'baseline_mode': True, 'verbose': False}
    )
    register(
        id="CompLift-IIWA",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'IIWA', 'verbose': False}
    )
    register(
        id="BaselineLift-IIWA",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'IIWA', 'baseline_mode': True, 'verbose': False}
    )
    register(
        id="CompLift-Kinova3",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'Kinova3', 'verbose': False}
    )
    register(
        id="BaselineLift-Kinova3",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'Kinova3', 'baseline_mode': True, 'verbose': False}
    )