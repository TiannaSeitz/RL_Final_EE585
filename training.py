#python

#################################
# File:     training.py
# Purpose:  will be used to train model
#           using coppeliaSim

# Author:   Tianna Seitz
# Released: 12/11/2023
#
# Notes: Version is Trainable
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
# DummyVecEnv wraps environment automatically
save_path = "/home/mabl/tianna_ws/RL_Final_EE585/saved_models/test6/mymodel.zip"
log_path = "/home/mabl/tianna_ws/RL_Final_EE585/tensorboard/test6"
print("init model")
model =  PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)

# model should update policy after every action
model.learn(total_timesteps=1)

# model.learn loops indefinitely so it never reaches this point 
episodes = 200
for ep in range(0, episodes):
	obs = env.reset()
	done = False
	while done == False:
		print(f"episodes = {ep}")
		action, _states = model.predict(obs)
		obs, rewards, done, info, _ = env.step(action)
		# model.learn(total_timesteps=1, reset_num_timesteps=False)
		if done == True:
			break
	model.save(save_path)

evaluate_policy(model, env, n_eval_episodes = 2, render = False)
env.close()

