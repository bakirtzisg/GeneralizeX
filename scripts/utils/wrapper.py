import gymnasium as gym
from utils.util import load_agents

class MDP():
    def __init__(self, env: str, dir: str, policy: str, baseline_mode: bool = False, tasks='all', prefix: str = ''):
        self.env_name = env
        self.env = gym.make(self.env_name) 
        self.dir = dir
        self.policy = policy
        self.tasks = self.env.unwrapped.tasks if tasks == 'all' else tasks
        self.baseline = baseline_mode
        self.agent = load_agents(dir=self.dir, 
                                 policy=self.policy, 
                                 baseline=self.baseline, 
                                 tasks=self.tasks,
                                 prefix=prefix)
        if self.baseline:
            self.state_space_dim = self.env.unwrapped.observation_space
            self.action_space_dim = self.env.unwrapped.action_space
        else:
            self.state_space_dim = {f'{task}': self.env.unwrapped.observation_spaces[f'{task}'].shape[0] for task in self.tasks}
            self.action_space_dim = {f'{task}': self.env.unwrapped.action_spaces[f'{task}'].shape[0] for task in self.tasks}
        # self.horizon = self.env._max_episode_steps

        if isinstance(env, str):
            assert env.lower() in dir.lower(), 'Sanity check to make sure policy matches environment'
        
        assert all(task in self.env.unwrapped.tasks for task in self.tasks), 'Invalid tasks flag!'

class SequentialMDP():
    def __init__(self, mdps, subprocesses):
        pass