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
import gymnasium
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from random import randint

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from sim_code import amigoEnv #verify


env = amigoEnv()
# Wrap the environment with DummyVecEnv for compatibility
# env = DummyVecEnv([lambda: env])
save_path = "/home/mabl/tianna_ws/RL_Final_EE585/saved_models/test2"
log_path = "/home/mabl/tianna_ws/RL_Final_EE585/tensorboard/test2"
print("init model")
model =  PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)

# for step in range(200):
# 	# env.render()
# 	# take random action
# 	obs, reward, done, info, _ = env.step(env.action_space.sample())
# 	print(reward, done)


n_episodes = 200
for episode in range(n_episodes):
    model.learn(total_timesteps=21, reset_num_timesteps=False)
    model.save(save_path)
env.close()

# alright, so it gets a reward, but the position doesn't update in model.learn
# like it does in env.step. We need to find out why.

