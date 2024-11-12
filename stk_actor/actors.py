import gymnasium as gym
import torch
import numpy as np
from bbrl.agents import Agent

class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        return action


class Actor(Agent):
    """BBRL Actor compatible avec les modèles PPO de Stable Baselines3"""

    def __init__(self, model, action_space: gym.Space):
        super().__init__()
        self.model = model
        self.action_space = action_space

    def forward(self, t: int):
        # Prend une action avec le modèle PPO entraîné
        obs = self.get(("env/env_obs", t))
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _states = self.model.predict(obs, deterministic=True)
        self.set(("action", t), torch.LongTensor(action))

class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.TensorDict(dict(self.action_space.sample())))

class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        pass

