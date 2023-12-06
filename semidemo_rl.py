#python

#################################
# File:     semidemo_rl.py
# Purpose:  proof of concept reinforcement learning

# Author:   Tianna Seitz
# Released: 12/06/2023
#
#################################


import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from random import randint

# 0: move forward
# 1: move backwards
# 2: turn right
# 3: turn left

class amigoEnv(gym.Env):
    def __init__ (self):
        super(amigoEnv, self).__init__()
        self.observation_space = Box(low = -10, high = 10, shape=(2,), dtype = np.float32) # this is how the env observations is produced
        self.action_space = Discrete(4) 
        self.position = np.array([0,0])
        self.actions_taken = 0
        self.actions_max = 20
        self.reward = 0
        self.destination = np.array([randint(1,10), randint(1,10)])
        print(f"destination{self.destination}")
        # self.gamma = .01 # this is to try and get it to optimize route later
        print("setting up")

    def reset(self):
        self.position = np.array([0,0])
        self.actions_taken = 0
        self.actions_max = 20
        self.reward = 0
        # self.destination = np.array([randint(1,10), randint(1,10)])
        print(f"destination{self.destination}")
        print("resetting")
        return self.position

    def step(self, action):
        self.actions_taken += 1
        reward = self.reward

        if action == 0:
            self.position[1] += 1
            reward += 1
            # reward = reward + (1*(self.gamma**self.actions_taken))

        elif action == 1:
            self.position[1] -= 1
            reward -= 1
            # reward = reward - 10

        elif action == 2:
            self.position[0] += 1
            reward += 1
            # reward = reward + (1*(self.gamma**self.actions_taken))

        elif action == 3:
            self.position[0] -= 1
            reward += 1
            # reward = reward - 10

        else: 
            print("not an action!!")

        if self.actions_taken == self.actions_max: # if we run out of actions, thats really bad
            reward -= 50
            print("ran out of time")
            done = True
        else:
            done = False
        
        if np.array_equal(self.position, self.destination): # end goal is to end up at this x,y location
            reward += 50
            print("reached location")
            done = True

        info = {}
        
        return self.position, reward, done, {}


env = DummyVecEnv([lambda: amigoEnv()])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=70000)


obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    print(f"obs: {obs}")
    print(f"reward: {reward}")
    if done == True:
        break


