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

env = amigoEnv()
# Wrap the environment with DummyVecEnv for compatibility
env = DummyVecEnv([lambda: env])


log_path = "~/tjs1980/RL_Final_EE585/tensorboard/test1"
print("init model")
# model =  PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)
model =  PPO('MultiInputPolicy', env, verbose = 1, n_steps= 500, policy_kwargs=dict{'net_arc':[256, 256, 256]})
model.learn(total_timesteps = 10)
# Tianna, make a new dir to save your models in!!!
save_path = "~/tjs1980/RL_Final_EE585/saved_models/test1"
model.save(save_path)
evaluate_policy(model, env, n_eval_episodes = 10, render = True)
env.close()