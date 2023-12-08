#python

#################################
# File:     deployment_sim.py
# Purpose:  will be used to test model
#           using coppeliaSim

# Author:   Tianna Seitz
# Released: 12/06/2023
#
# Notes: Version is incomplete

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

env = amigoEnv()
save_path = "~/tjs1980/RL_Final_EE585/saved_models"
model = PPO.load(save_path, env = env)

episodes = 5

for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        env.render() 
        action, _ = model.predict(obs)
        obs, reward, done = env.step(action)
        score += reward
        print('Episode:{} Score: {}'.format(episode, score))
