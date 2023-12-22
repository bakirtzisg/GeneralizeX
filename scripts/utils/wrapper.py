import gymnasium as gym
from utils.util import load_policy

class MDP():
    def __init__(self, env, dir, policy, baseline_mode=False, tasks='all', prefix=''):
        self.env = env if isinstance(env, gym.Env) else gym.make(env) 
        self.dir = dir
        self.policy = policy
        self.tasks = self.env.unwrapped.tasks if tasks == 'all' else tasks
        self.baseline = baseline_mode
        self.agent = load_policy(dir=self.dir, 
                                 policy=self.policy, 
                                 baseline=self.baseline, 
                                 tasks=self.tasks,
                                 prefix=prefix)
        
        if isinstance(env, str):
            assert env.lower() in dir.lower(), 'Sanity check to make sure policy matches environment'
        
        assert all(task in self.env.unwrapped.tasks for task in self.tasks), 'Invalid tasks flag!'