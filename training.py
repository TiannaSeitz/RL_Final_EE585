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
save_path = "/home/mabl/tianna_ws/RL_Final_EE585/saved_models/test6/mymodel.zip"
log_path = "/home/mabl/tianna_ws/RL_Final_EE585/tensorboard/test6"
print("init model")
model =  PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)

model.learn(total_timesteps=1, reset_num_timesteps=False)
episodes = 1
for ep in range(0, episodes):
	obs = env.reset()
	done = False
	while done == False:
		print(f"episodes = {episodes}")
		action, _states = model.predict(obs)
		obs, rewards, done, info, _ = env.step(action)
		model.learn(total_timesteps=1, reset_num_timesteps=False)
		if done == True:
			break
	model.save(save_path)

evaluate_policy(model, env, n_eval_episodes = 2, render = False)
env.close()
# for sets in range(0, 200):
#     obs = env.reset()
#     for episode in range(0, 21):
#         obs, reward, done, info, _ = env.step(env.action_space.sample())
#         model.learn(total_timesteps=1, reset_num_timesteps=False)
#         if done == True:
#             obs = env.reset()
#         model.save(save_path)
#     env.close()

# alright, so it gets a reward, but the position doesn't update in model.learn
# like it does in env.step. We need to find out why.

