import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np

class FilteredObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper permettant de garder uniquement les observations présentes dans keys_to_keep
    """
    def __init__(self, env, keys_to_keep):
        super().__init__(env)
        self.keys_to_keep = keys_to_keep
        
        # Filtrage des dimensions et transformation en espace Box
        low = []
        high = []
        for key in keys_to_keep:
            space = self.env.observation_space[key]
            low.extend(space.low.flatten())
            high.extend(space.high.flatten())
        
        # Définir le nouvel espace d'observation comme un Box 1D
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def observation(self, observation):
        # Filtrage des clés et conversion en vecteur 1D
        filtered_values = [observation[key].flatten() for key in self.keys_to_keep]
        return np.concatenate(filtered_values)
    
class DictToBoxActionWrapper(gym.ActionWrapper):
    """
    Wrapper permettant de passer d'un dictionnaire d'actions à des actions sous forme matricielle pour PPO après
    """
    def __init__(self, env):
        super(DictToBoxActionWrapper, self).__init__(env)
        # Définir l'espace d'action en tant que Box 1D qui combine l'accélération et la direction
        self.action_space = Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def action(self, action):
        # Diviser l'action en 'acceleration' et 'steer' à partir du vecteur 1D
        return {'acceleration': np.array([action[0]], dtype=np.float32), 'steer': np.array([action[1]], dtype=np.float32)}