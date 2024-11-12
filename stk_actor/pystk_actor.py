import gymnasium as gym
from bbrl.agents import Agent
import torch
from .actors import Actor, SamplingActor, ArgmaxActor, MyWrapper
from stable_baselines3 import PPO

# Spécifications de l'environnement et du joueur
env_name = "supertuxkart/flattened_multidiscrete-v0"
player_name = "Hatem"

# Chargement du modèle et préparation de l'agent
def get_actor(state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space) -> Agent:
    model_path = "models/pystk_actor.zip"
    model = PPO.load(model_path)

    actor = Actor(model,action_space)
    if state is None:
        # Utiliser un acteur aléatoire pour le test si aucun état n'est chargé
        return SamplingActor(action_space)

    # Charger les paramètres de l'acteur si un état est fourni
    actor.load_state_dict(state)
    return Agent(actor)

def get_wrappers():
    return [lambda env: MyWrapper(env, option="1")]
