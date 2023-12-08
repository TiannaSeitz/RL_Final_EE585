#python

#################################
# File:     training.py
# Purpose:  will be used to train model
#           using coppeliaSim

# Author:   Tianna Seitz
# Released: 12/07/2023
#
# Notes: Version is incomplete
#
#################################

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import math
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from random import randint

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from sim_code import amigoEnv #verify


import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomPolicy():
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, *args, **kwargs)

        # Extract information about the different subspaces
        location_space = observation_space['location']
        old_location_space = observation_space['old_location']
        proximity_sensor_space = observation_space['proximity_sensor']
        actions_taken_space = observation_space['actions_taken']
        orientation_space = observation_space['orientation']

        # Define neural network architecture for each subspace
        self.location_network = nn.Sequential(
            nn.Linear(location_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, location_space.n)
        )

        self.old_location_network = nn.Sequential(
            nn.Linear(old_location_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, old_location_space.n)
        )

        self.proximity_sensor_network = nn.Sequential(
            nn.Linear(proximity_sensor_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, proximity_sensor_space.shape[0])
        )

        self.actions_taken_network = nn.Sequential(
            nn.Linear(actions_taken_space.n, 64),
            nn.ReLU(),
            nn.Linear(64, actions_taken_space.n)
        )

        self.orientation_network = nn.Sequential(
            nn.Linear(orientation_space.n, 64),
            nn.ReLU(),
            nn.Linear(64, orientation_space.n)
        )

    def forward(self, obs, deterministic=False):
        # Extract observations from the dictionary
        location_obs = obs['location']
        old_location_obs = obs['old_location']
        proximity_sensor_obs = obs['proximity_sensor']
        actions_taken_obs = obs['actions_taken']
        orientation_obs = obs['orientation']

        # Forward pass for each subspace
        location_output = self.location_network(location_obs)
        old_location_output = self.old_location_network(old_location_obs)
        proximity_sensor_output = self.proximity_sensor_network(proximity_sensor_obs)
        actions_taken_output = self.actions_taken_network(F.one_hot(actions_taken_obs, num_classes=50))  # one-hot encode discrete action
        orientation_output = self.orientation_network(F.one_hot(orientation_obs, num_classes=91))  # one-hot encode discrete action

        # Combine the outputs if necessary
        combined_output = torch.cat([location_output, old_location_output, proximity_sensor_output, actions_taken_output, orientation_output], dim=-1)

        return combined_output  # Return the combined output



env = amigoEnv()
# Wrap the environment with DummyVecEnv for compatibility
env = DummyVecEnv([lambda: env])


log_path = "~/tjs1980/RL_Final_EE585/tensorboard/test1"
print("init model")
# model =  PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)
model =  PPO(CustomPolicy, env, verbose = 1)
model.learn(total_timesteps = 10)
# Tianna, make a new dir to save your models in!!!
save_path = "~/tjs1980/RL_Final_EE585/saved_models/test1"
model.save(save_path)
evaluate_policy(model, env, n_eval_episodes = 10, render = True)
env.close()