import gymnasium as gym
import robosuite
import numpy as np
from robosuite import load_controller_config
from gymnasium.envs.registration import register

class CompLiftEnv(gym.Env):
    def __init__(self, robot="Panda", baseline_mode=False, training_mode=False):
        config = load_controller_config(default_controller='OSC_POSITION')
        self._robot = robot
        self._env = robosuite.make(
            env_name="Lift",
            robots=self._robot,
            controller_configs=config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=100
        )
        self.tasks = ['reach', 'lift']
        self.subprocesses = ['grip'] # TODO
        
        assert len(self.subprocesses) == len(self.tasks) - 1, 'Invalid zig-zag diagram'

        self.action_spaces = {
            'reach': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32), # (x,y,z)
            'grip': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32), # (z,g)
            'lift': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32), # TODO: (z,g)
        }
        self.observation_spaces = {
            'reach': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
            'grip': gym.spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
        }

        self.current_task = 'reach'
        self.reward_criteria = None
        self.setup_skip_reset_once = False
        self.fresh_reset = False
        self.baseline_mode = baseline_mode
        # In baseline mode, we train all tasks at once, so training_mode=False
        self.training_mode = False if self.baseline_mode else training_mode 

    @property
    def action_space(self):
        if self.baseline_mode:
            return gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            return self.action_spaces[self.current_task]
    
    @property
    def observation_space(self):
        if self.baseline_mode:
            return gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
        else:
            return self.observation_spaces[self.current_task]

    def _get_obs(self):
        obs = self._env._get_observations() 
        
        hand_pos = obs['robot0_eef_pos']
        cube_pos = obs['cube_pos']
        gripper = obs['robot0_gripper_qpos'][0] * 2

        if self.baseline_mode:
            # obs = (hand_position, cube_position, gripper); np.shape(obs) == (7,)
            obs = np.concatenate([hand_pos, cube_pos, [gripper]]).astype(np.float32)
            return obs

        if self.current_task == 'reach':
            # obs = (hand_position, cube_position); np.shape(obs) == (6,)
            obs = np.concatenate([hand_pos, cube_pos]).astype(np.float32)
        elif self.current_task == 'grip':
            # obs = (hand_z_position, gripper); np.shape(obs) == (2,)
            obs = np.concatenate([hand_pos[2:3], [gripper]]).astype(np.float32)
        elif self.current_task == 'lift':
            # obs = (hand_z_position, cube_z_position, gripper); np.shape(obs) == (3,)
            obs = np.concatenate([hand_pos[2:3], cube_pos[2:3], [gripper]]).astype(np.float32)
        else:
            raise RuntimeError('Invalid task')
        return obs

    def _process_action(self, action):
        if self.baseline_mode:
            return action
        if self.current_task == 'reach':
            action = np.concatenate([action[:3], [-1]])
        if self.current_task == 'grip':
            # TODO
            raise NotImplementedError
        if self.current_task == 'lift':
            action = np.concatenate([[0, 0], action])
        return action
     
    def _compute_reward_criteria(self, observation):
        hand_pos = observation['robot0_eef_pos'].copy()
        cube_pos = observation['cube_pos'].copy()
        reach_dist = np.linalg.norm(cube_pos - hand_pos)
        criteria = {
            'reach_dist': reach_dist,
            'cube_height': cube_pos[2]
        }
        return criteria

    def _evaluate_task(self, observation):
        task_completed = False
        task_failed = False

        new_reward_criteria = self._compute_reward_criteria(observation)
        if self.baseline_mode:
            task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
            task_reward += new_reward_criteria['cube_height'] - self.reward_criteria['cube_height']
        else:
            if self.current_task == 'reach':
                task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
                if new_reward_criteria['reach_dist'] < 0.01:
                    task_completed = True
            elif self.current_task == 'lift':
                task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
                task_reward += new_reward_criteria['cube_height'] - self.reward_criteria['cube_height']
                if self._env._check_success(): # if cube is lifted
                    task_completed = True
            if task_completed:
                task_reward = 10

        self.reward_criteria = new_reward_criteria
        return task_reward, task_completed, task_failed

    def step(self, action):
        self.fresh_reset = False
        action = self._process_action(action)
        observation, reward, done, info = self._env.step(action)
        task_reward, task_completed, task_failed = self._evaluate_task(observation)
        terminated = done
        truncated = task_failed or terminated
        # The entire task is completed if the can is placed in the bin.
        if task_completed:
            task_reward = 10

        info['task_reward'] = task_reward
        info['task_truncated'] = truncated

        # Note: in training mode, we only train one subtask at a time.
        if task_completed and (not self.training_mode):
            if self.current_task == self.tasks[len(self.tasks) - 1]:
                # Completed final subtask (lift), so completely reset environment
                self.fresh_reset = True
                info['task_completed'] = True
            else:
                # Completed current subtask. Moving on the next one
                self.current_task = self.tasks[self.tasks.index(self.current_task) + 1]
        else:
            info['task_completed'] = False
        
        # Log current task, observation, and action
        obs = self._get_obs()
        info['current_task'] = self.current_task
        info['current_task_obs'] = obs
        info['current_action'] = action
        return obs, task_reward, terminated, task_failed, info

    def _setup_skip_reset(self):
        self.setup_skip_reset_once = True

    def reset(self, seed=None, options=None):
        if self.setup_skip_reset_once:
            self.setup_skip_reset_once = False
        else:
            if self.fresh_reset:
                # Reset current task to first task (reach)
                print(f'Called reset right after task switch to {self.current_task}.')
                self.current_task = 'reach'
                self.fresh_reset = False
            self._env.reset()

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
        Register gymnasium environments
    """
    register(
        id="CompLift-Panda",
        entry_point=CompLiftEnv,
        max_episode_steps=50
    )
    register(
        id="BaselineCompLift-Panda",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'baseline_mode': True}
    )
    register(
        id="CompLift-IIWA",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'IIWA'}
    )
    register(
        id="BaselineCompLift-IIWA",
        entry_point=CompLiftEnv,
        max_episode_steps=50,
        kwargs={'robot': 'IIWA', 'baseline_mode': True}
    )