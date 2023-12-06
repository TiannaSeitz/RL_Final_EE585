#python

#################################
# File:     environment.py
# Purpose:  create an reinforcement learning environment
#           where an AmigoBot can be trained to navigate 
#           to a designated location while avoiding
#           obstacles

# Author:   Tianna Seitz
# Released: 12/06/2023
#
# Notes: Version is incomplete
#
#################################


'''
3 files:

train file
- this is where we will call our class here for training
-
-

test file (x2)?
- will need one for sim
- will need one for deployment
-

environment in coppeliaSim file
- api is: zmqRemoteApi import RemoteAPIClient
- import gym
- when we want to move the joints in coppeliaSim we will use our setVelocity whatever in our actions with ssim.getObect for each controllable
- we can use sim.checkProximity or whatever to get the proximity.
- we'll add our client info in def load(self) in our environment set
- we can add our cubes n shit in here if we want
- we'll need a variable to distinguish if we use ros or coppeliasim movements
'''

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import math
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from random import randint

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


usensors=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
noDetectionDist=.75
maxDetectionDist=0.3
detect=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
braitenbergL=[-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
braitenbergR=[-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
v0=2

'''
We keep track of the direction of the robot so that we know to turn if we are not facing the direction of motion
mostly needed for if we are facing forward and want to turn right and vice versa
Actions:
0 - forward
1 - back
2 - left
3 - right
'''


class amigoEnv(gym.Env):
    def __init__ (self):
        super(amigoEnv, self).__init__()
        # gaurenteed this will need to be fixed
        self.ultrasonic = Box(low = noDetectionDist, high = maxDetectionDist, shape=(len(usensors),), dtype = np.float32) # this is how the env observations is produced
        self.orientation = 90
        self.position = MultiDiscrete([10,10])
        self.old_location = [0,0]
        self.action_space = Discrete(4) 
        self.actions_taken = 0
        self.actions_max = 20
        self.reward = 0
        self.destination = np.array([randint(1,10), randint(1,10)])

        self.observation_space = Dict({
            'location': self.position,
            'old_location': self.old_location,
            'proximity_sensor': self.ultrasonic,
            'actions_taken': self.actions_taken,
            'destination': self.destination,
            'orientation': self.orientation
        })

        print(f"destination{self.destination}")
        # self.gamma = .01 # this is to try and get it to optimize route later
        print("setting up")

    def reset(self): # fix this
        self.position = np.array([0,0]) # especially fix this
        self.orientation = 90
        self.old_location = [0,0]
        self.actions_taken = 0
        self.reward = 0
        # self.destination = np.array([randint(1,10), randint(1,10)])
        print(f"destination{self.destination}")
        print("resetting")
        return self.position
    
    def step(self, action):  

        offset = 0.05
        if self.actions_taken == self.actions_max: # if we run out of actions, thats really bad
            reward -= 50
            print("ran out of time")
            done = True
        else:
            done = False


        self.actions_taken += 1
        reward = self.reward
        self.old_location[0] = self.position[0]
        self.old_location[1] = self.position[0]

        # sim.setJointTargetVelocity(sim.getObject("./leftMotor"),vLeft)
        # sim.setJointTargetVelocity(sim.getObject("./rightMotor"),vRight)
        
        # we will need a function to help the robot turn if it is not facing the direction it wants to move
        if action == 0: # move forward
            if self.orientation != 90:
                # put code to turn robot
                orientation = 90
            # move forward
            self.position[1] = self.old_location[1]+1

        elif action == 1: # move backwards
            if self.orientation != 90:
                # put code to turn robot
                orientation = 90
            # move backwards
            self.position[1] = self.old_location[1]-1

        elif action == 2: # move left
            if self.orientation != 0:
                # put code to turn robot
                orientation = 0
            # move backwards (left)
            self.position[0] = self.old_location[0]-1

        elif action == 3: # move right
            if self.orientation != 0:
                # put code to turn robot
                orientation = 0
            # move foward (right)
            self.position[0] = self.old_location[0]+1

        else: 
            print("not an action!!")

        if np.array_equal(self.position, self.destination): # end goal is to end up at this x,y location
            reward += 50
            print("reached location")
            done = True

        # are we too close to an object?
        if self.ultrasonic < maxDetectionDist-offset:
            # avoiding objects is the most important task consequence of reward must be high
            reward -= 10
        else:
            reward += 1

        # did we advance towards the x coordinate of location?
        # it's okay for robot to move away from final dest to avoid object, so consequence will be lower
        if (self.destination[0]-self.position[0]) < (self.destination[0]-self.old_location[0]):
            reward += 1
        else:
            reward -= 1

        # did we advance towards the y coordinate of location?
        if (self.destination[1]-self.position[1]) < (self.destination[1]-self.old_location[1]):
            reward +=1
        else:
            reward -=1
        
        # does this have to be a specific order?
        return self.actions_taken, self.position, self.old_location, self. orientation, reward, done, {}

    def load_scene(self):
        print("placeholder")
        self.api = RemoteAPIClient('localhost', 23000)
        self.sim = self.api.getObject('sim')
        # do we need stepping?? maybe?
        self.sim.loadScene('~/tjs1980/seitz_csim/scenes/tjs1980_final_env.ttt')
        # we might need to add shapes?
        self.sim.startSimulation()     

    def add_obstacle(self):
        print("we'll randomly add cubes later if time")

    def getProximity(self):
        # likely won't need anything other than res to see if something was detected and dist if something was
        res, dist, point, _, _ = self.sim.handleProximitySensor(ultrasonicSensor[0])
        # I think maybe we don't need this either
        # res, dist, point, _, _ = sim.readProximitySensor
        return res, dist, point


# we will likely not need what is below!

def sysCall_init():
    sim = require('sim')
    robot=sim.getObject('.')
    obstacles=sim.createCollection(0)
    sim.addItemToCollection(obstacles,sim.handle_all,-1,0)
    sim.addItemToCollection(obstacles,sim.handle_tree,robot,1)
    global usensors
    for i in range (0,16):
        print(i)
        usensors[i]=sim.getObject("./ultrasonicSensor",[i])
        sim.setObjectInt32Param(usensors[i],sim.proxintparam_entity_to_detect,obstacles)

#def sysCall_cleanup(): 
 

def sysCall_actuation(): 
    global noDetectionDist
    global maxDetectionDist
    global detect
    global braitenbergL
    global braitenbergR
    global v0
    
    for i in range (0,16):
        res=sim.readProximitySensor(usensors[i])[0]
        dist=sim.readProximitySensor(usensors[i])[1]
        if (res>0) and (dist<noDetectionDist):
            if (dist<maxDetectionDist):
                dist=maxDetectionDist
            detect[i]=1-((dist-maxDetectionDist)/(noDetectionDist-maxDetectionDist))
        else:
            detect[i]=0
        
    vLeft=v0
    vRight=v0
    
    for i in range (0,16):
        vLeft=vLeft+braitenbergL[i]*detect[i]/2
        vRight=vRight+braitenbergR[i]*detect[i]
    
    sim.setJointTargetVelocity(sim.getObject("./leftMotor"),vLeft)
    sim.setJointTargetVelocity(sim.getObject("./rightMotor"),vRight)


# alright. Question of the century. putting it all together... How though?