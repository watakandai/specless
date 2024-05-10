import copy

import gymnasium as gym


class MultiAgentWrapperEnv(gym.Env):
    def __init__(self, env, num_agent):
        self.envs = [copy.deepcopy(env) for i in range(num_agent)]
