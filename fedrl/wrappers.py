import gymnasium as gym


class ScaledActionEnv(gym.ActionWrapper):
    """Multiplies actions by a mutable scale c_i (actuator strength).

    The training loop mutates `.scale` to simulate actuator damage mid-training.
    """

    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def action(self, action):
        return self.scale * action
