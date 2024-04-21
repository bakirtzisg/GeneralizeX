import gymnasium as gym
import robosuite
import numpy as np

from robosuite import load_controller_config
from robosuite.robots.robot import Robot
from gymnasium.envs.registration import register

class CompPickPlaceCanEnv(gym.Env):
    def __init__(self, robot="Panda", baseline_mode=False, training_mode=False, verbose=True, normalize_reward=False):
        # Why not OSC_POSE?
        self._controller_config = load_controller_config(default_controller='OSC_POSITION')
        self._robot = robot
        self._verbose = verbose
        self._normalize_reward = normalize_reward
        # Subtasks to complete the pick and place task (in order)
        self.tasks = ['reach', 'grasp', 'lift', 'move', 'place']
        
        self.current_task = self.tasks[0]

        # Create separate environments for each subtask
        self._env = robosuite.make(
            env_name="PickPlaceCan",
            robots=self._robot,
            controller_configs=self._controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=100,
        )
        
        # The initial joint configuration of the robot depends on the subtask
        self._robot_init_qpos = {self.current_task: self._env.robots[0].init_qpos}

        if self._verbose:
            print(f'{self.current_task} initial joint configuration: {self._robot_init_qpos[self.current_task]}')
        
        # Action and observation spaces for compositional training
        # For baseline training, see action_space() and observation_space()
        self.action_spaces = {
            'reach': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32), # (x,y,z)
            'grasp': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32), # (z,g)
            'lift': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32), # (z,g)
            'move': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32), # (x,y,z)
            'place': gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32), # (z,g)
        }
        self.observation_spaces = {
            'reach': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
            'grasp': gym.spaces.Box(low=-5, high=5, shape=(3,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'move': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
            'place': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
        }

        self.reward_criteria = None
        # Reset params
        self.setup_skip_reset_once = False
        self.fresh_reset = False
        # In baseline mode, we train all tasks at once, so training_mode=False
        self.baseline_mode = baseline_mode
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
        can_pos = obs['Can_pos']
        goal_pos = self._env.target_bin_placements[self._env.object_to_id['can']]
        gripper = obs['robot0_gripper_qpos'][0] * 2

        if self.baseline_mode:
            # (hand_position, can_position, gripper); np.shape(obs) == (7,)
            obs = np.concatenate([hand_pos, can_pos, [gripper]]).astype(np.float32)
            return obs

        if self.current_task == 'reach':
            # (hand_position, can_position); np.shape(obs) == (6,)
            obs = np.concatenate([hand_pos, can_pos]).astype(np.float32)
        elif self.current_task == 'grasp':
            # (hand_z_position, cube_z_position, gripper); np.shape(obs) == (3,)
            obs = np.concatenate([hand_pos[2:3], can_pos[2:3], [gripper]]).astype(np.float32)
        elif self.current_task == 'lift':
            # (hand_z_position, cube_z_position, gripper); np.shape(obs) == (3,)
            obs = np.concatenate([hand_pos[2:3], can_pos[2:3], [gripper]]).astype(np.float32)
        elif self.current_task == 'move':
            obs = np.concatenate([hand_pos, can_pos]).astype(np.float32)
        elif self.current_task == 'place':
            obs = np.concatenate([hand_pos, can_pos, goal_pos, [gripper]]).astype(np.float32)
        else:
            raise RuntimeError('Invalid task')
        return obs

    def _process_action(self, action):
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
        elif self.current_task == 'move':
            # Control eef x,y,z pos only
            action = np.concatenate([action[:3], [1]])
        elif self.current_task == 'place':
            # Control eef z pos and gripper only
            action = np.concatenate([[0, 0], action])
        else:
            raise RuntimeError('Invalid task')
        
        assert len(action) == 4, f'action for task {self.current_task} is shape {np.shape(action)}. Expected (4,). The action is {action}'

        return action
     
    def _compute_reward_criteria(self, observation):
        hand_pos = observation['robot0_eef_pos'].copy()
        can_pos = observation['Can_pos'].copy()
        table_height = self._env.target_bin_placements[self._env.object_to_id['can']][2]
        lift_height = table_height + 0.25
        goal_pos = self._env.target_bin_placements[self._env.object_to_id['can']]

        reach_dist = np.linalg.norm(can_pos + np.array([0, 0, 0.005]) - hand_pos)
        grasp_dist = np.linalg.norm([0, 0, 0.02] - can_pos)
        lift_dist = np.linalg.norm([0, 0, lift_height] - can_pos)
        move_dist = np.linalg.norm([goal_pos[0], goal_pos[1], lift_height] - can_pos)
        goal_dist = np.linalg.norm(goal_pos - can_pos)

        check_grasp = self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=self._env.objects[-1].contact_geoms)
        
        criteria = {
            'reach_dist': reach_dist,
            'can_pos': can_pos,
            'grasp_dist': grasp_dist,
            'lift_dist': lift_dist,
            'lift_height': lift_height,
            'move_dist': move_dist,
            'goal_dist': goal_dist,
            'check_grasp': check_grasp,
        }
        return criteria

    def _evaluate_task(self, observation):
        task_completed = False
        task_failed = False
        new_reward_criteria = self._compute_reward_criteria(observation)
        if self.baseline_mode:
            """
                Using robosuite pick and place environment reward function:

                - Reaching: in {0, [0, 0.1]}, proportional to the distance between the gripper and the closest object
                - Grasping: in {0, 0.35}, nonzero if the gripper is grasping an object
                - Lifting: in {0, [0, 1]}, nonzero only if object is grasped; proportional to lifting height
                - Placing: in {0, [0.5, 0.7]}, nonzero only if object is lifted; proportional to distance from object to bin
         
            """
            task_reward = self._env.reward()
        else:
            """
                Compositional mode: separate reward functions for each subtask
            """
            if self.current_task == 'reach':
                task_reward = 1 - np.tanh(10.0 * new_reward_criteria['reach_dist'])
                if new_reward_criteria['reach_dist'] < 0.01:
                    task_completed = True
                    task_reward = 2
            
            elif self.current_task == 'grasp':
                # task_reward =  1 - np.tanh(10.0 * new_reward_criteria['grasp_dist'])
                task_reward = 0
                if new_reward_criteria['check_grasp']:
                    task_completed = True
                    task_reward += 0.25
            
            elif self.current_task == 'lift':
                task_reward = 1 - np.tanh(10.0 * new_reward_criteria['lift_dist'])
                # can_height = new_reward_criteria['can_pos'][2]
                if new_reward_criteria['lift_dist'] < 0.01: # if can is lifted
                    task_completed = True
                    task_reward = 2

            elif self.current_task == 'move':
                task_reward = 1 - np.tanh(10.0 * new_reward_criteria['move_dist'])
                if new_reward_criteria['move_dist'] < 0.01: # if can is lifted
                        task_completed = True
                        task_reward = 2
            
            elif self.current_task == 'place':
                task_reward = 1 - np.tanh(10.0 * new_reward_criteria['goal_dist'])
                if new_reward_criteria['goal_dist'] < 0.01:
                    task_completed = True
                    task_reward = 2

            else:
                raise NotImplementedError(f'Invalid task {self.current_task}')
        
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
                # Completed final subtask (place), so completely reset environment 
                # (set current_task to first subtask)
                self.fresh_reset = True
                info['task_success'] = True
            else:
                # if self._verbose:
                print(f"Completed subtask {self.current_task}, moving on to {self.tasks[self.tasks.index(self.current_task) + 1]}")
                # Compositional training mode: completed current subtask so moving on to the next subtask
                self.current_task = self.tasks[self.tasks.index(self.current_task) + 1]


        # If completed current subtask, save the final q_pos to later set as init q_pos for next subtask
        # Skip when current task is last subtask
        if self.current_task != self.tasks[-1] and not self.baseline_mode:
            # next_task = self.tasks[self.tasks.index(self.current_task) + 1]
            if task_completed:
                # Check if robot_init_qpos for next_task has been set. If not, set it to final qpos of previous task
                if self._robot_init_qpos.get(self.current_task) is None:
                    self._robot_init_qpos[self.current_task] = np.arctan2(
                        observation['robot0_joint_pos_sin'],
                        observation['robot0_joint_pos_cos'],
                    )
                    # if self._verbose:
                    print(f"Setting initial joint configuration for subtask {self.current_task} to:", self._robot_init_qpos[self.current_task])
        
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
        info['init_qpos'] = self._robot_init_qpos               # Initial qpos

        return obs, task_reward, terminated, task_failed, info

    def _setup_skip_reset(self):
        self.setup_skip_reset_once = True

    def reset(self, seed=None, options=None):
        if self.setup_skip_reset_once:
            self.setup_skip_reset_once = False
        else:
            if self.fresh_reset:
                # Reset current task to first task (reach)
                if self._verbose:
                    print(f'Called reset right after task switch to {self.current_task}.')
                self.current_task = self.tasks[0]
            # Reset initial joint configuration to correct init_qpos based on subtask
            self._env.reset()
            self._env.robots[0].init_qpos = self._robot_init_qpos[self.current_task]

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
        id="CompPickPlaceCan-Panda",
        entry_point=CompPickPlaceCanEnv,
        max_episode_steps=100
    )
    register(
        id="BaselinePickPlaceCan-Panda",
        entry_point=CompPickPlaceCanEnv,
        max_episode_steps=100,
        kwargs={'baseline_mode': True}
    )
    register(
        id="CompPickPlaceCan-IIWA",
        entry_point=CompPickPlaceCanEnv,
        max_episode_steps=100,
        kwargs={'robot': 'IIWA'}
    )
    register(
        id="BaselinePickPlaceCan-IIWA",
        entry_point=CompPickPlaceCanEnv,
        max_episode_steps=100,
        kwargs={'robot': 'IIWA', 'baseline_mode': True}
    )
    register(
        id="CompPickPlaceCan-Kinova3",
        entry_point=CompPickPlaceCanEnv,
        max_episode_steps=100,
        kwargs={'robot': 'Kinova3'}
    )
    register(
        id="BaselinePickPlaceCan-Kinova3",
        entry_point=CompPickPlaceCanEnv,
        max_episode_steps=100,
        kwargs={'robot': 'Kinova3', 'baseline_mode': True}
    )