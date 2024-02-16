import gymnasium as gym


class TerminateIfNoStrategyWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        if action is None:
            return {}, 0, False, True, {}
        return self.env.step(action)
