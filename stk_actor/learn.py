from pathlib import Path
import gymnasium as gym
from pystk2_gymnasium import AgentSpec, FlattenerWrapper, PolarObservations, ConstantSizedObservations
from pystk2_gymnasium.stk_wrappers import OnlyContinuousActionsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch
import numpy as np
from .actors import Actor 
from .wrappers import DictToBoxActionWrapper, FilteredObservationWrapper

# Les clés d'observation à conserver pour le wrapper FilteredObservationWrapper
keys_to_keep = [
    'velocity',
    'center_path_distance',
    'max_steer_angle',
    'front',
    'center_path',
    'paths_start',
    'paths_width',
    'paths_end'
]

# Création de l'environnement avec les wrappers nécessaires
def create_env():
    env = gym.make("supertuxkart/full-v0", agent=AgentSpec(use_ai=False, name="Hatem"), render_mode=None, track='abyss')
    env = OnlyContinuousActionsWrapper(ConstantSizedObservations(PolarObservations(env)))
    env = FilteredObservationWrapper(env, keys_to_keep)
    env = DictToBoxActionWrapper(env)
    return env

if __name__ == "__main__":
    env = create_env()
    check_env(env, warn=True)

    # Configuration et lancement de PPO
    model = PPO(
        "MlpPolicy",    
        env,            
        verbose=1,      
        n_steps=2048,   
        batch_size=64,  
        gae_lambda=0.95,
        gamma=0.99,     
        ent_coef=0.01,  
        learning_rate=2.5e-4, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model.learn(total_timesteps=1000)

    # Sauvegarde du modèle dans le dossier `models`
    model_path = Path("models/test")
    model.save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")
