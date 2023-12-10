#python

#################################
# File:     environment.py
# Purpose:  create an reinforcement learning environment
#           where an AmigoBot can be trained to navigate 
#           to a designated location while avoiding
#           obstacles

# Author:   Tianna Seitz
# Released: 12/07/2023
#
# Notes: Version is ready to test/debug in lab
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
from gym.spaces import Discrete, Box

import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from time import sleep

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
        print("initalizing")
        self.noDetectionDist = 0.75
        self.minDetectionDist = 0.3

        self.destination_x = 1
        self.destination_y = 1
        
        self.action_space = Discrete(4) 
        self.actions_max = 20

        self.reward = 0
        self.done = False
        self.observations = np.zeros((8,1))

        client = RemoteAPIClient('localhost', 23000)
        self.sim = client.getObject('sim')
        client.setStepping(True)
        self.sim.stopSimulation()
        sleep(1)
        self.sim.closeScene()
        sleep(1)
        # do we need stepping?? maybe?
        self.sim.loadScene('/home/mabl/tianna_ws/RL_Final_EE585/tjs1980_final_env.ttt')
        # we might need to add shapes?
        self.sim.startSimulation()  

        self.initProximity()

        pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
        print(pos_of_sensor)

        self.position_x = pos_of_sensor[0]
        self.position_y = pos_of_sensor[1]
        self.orientation =  0
        self.old_location_x = pos_of_sensor[1]
        self.old_location_y = pos_of_sensor[0]
        self.actions_taken =  0
        # okay so we don't declare what the starting actions are here, we just set up the spaces for the stuff to be put into.
        
        self.observation_space = Box(low = -255, high = 255, shape = (8,1), dtype = np.float32) # may need to specify type!

    def reset(self): # may need to be revised...
        print(f"reward = {self.reward}")
        self.sim.stopSimulation()
        sleep(1)
        client = RemoteAPIClient('localhost', 23000)
        self.sim = client.getObject('sim')
        client.setStepping(True)
        sleep(1)
        self.sim.closeScene()
        sleep(1)
        # do we need stepping?? maybe?
        self.sim.loadScene('/home/mabl/tianna_ws/RL_Final_EE585/tjs1980_final_env.ttt')
        self.sim.startSimulation()  

        self.initProximity()
        pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
        print(pos_of_sensor)

        self.position_x = pos_of_sensor[0]
        self.position_y = pos_of_sensor[1]
        self.orientation =  0
        self.old_location_x = pos_of_sensor[1]
        self.old_location_y = pos_of_sensor[0]
        self.actions_taken =  0
        self.reward = 0
        
        self.observation_hold = [self.position_x, self.position_y, self.orientation, self.old_location_x, self.old_location_y, self.actions_taken, self.detect[3], self.detect[4]]
        # self.observations = np.zeros((22,1))

        for i in range(0,8):
            self.observations[i] = self.observation_hold[i]
        return self.observations, {}
    
    def step(self, action):
        self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
        self.done = False
        done = self.done

        offset = 0.05
        if self.actions_taken == self.actions_max: # if we run out of actions, thats really bad
            self.reward -= 50
            print("ran out of time")
            done = True
        else:
            done = False

        self.actions_taken += 1

        
        self.old_location_x = self.position_x
        self.old_location_y = self.position_y

        # we will need a function to help the robot turn if it is not facing the direction it wants to move
        # move forward will be positive, backwards is negative motion
        # move right is positive, move left is negative. 
        # robot front will either face forward or right.
        # moving backwards or left will always be reverse motion

        # left right time

        if action == 0: # move forward
            print('forward')
            if self.orientation != 90:
                # put code to turn robot
                self.move(-1, 1, 2.9)
                orientation = 90
            # move forward
            self.move(1, 1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_y = self.pos_of_sensor[1]

        elif action == 1: # move backwards
            print('backwards')
            if self.orientation != 90:
                # put code to turn robot
                self.move(-1, 1, 2.9)
                orientation = 90
            # move backwards
            self.move(-1, -1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_y = self.pos_of_sensor[1]

        elif action == 2: # move left
            print('left')
            if self.orientation != 0:
                # put code to turn robot
                self.move(1, -1, 2.9)
                orientation = 0
            # move backwards (left)
            self.move(-1, -1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_x = self.pos_of_sensor[0]

        elif action == 3: # move right
            print('right')
            if self.orientation != 0:
                # put code to turn robot
                self.move(1, -1, 2.9)
                orientation = 0
            # move foward (right)
            self.move(1, 1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_x = self.pos_of_sensor[0]

        # self.client.step() # I have no clue why this is here.
        ultrasonic_result = self.getProximity()

        if self.position_x == self.destination_x and self.position_y == self.destination_y: # end goal is to end up at this x,y location
            self.reward += 500
            print("reached location")
            done = True

        # are we too close to an object?
 
        if ultrasonic_result[3] < self.minDetectionDist-offset:
            # avoiding objects is the most important task consequence of reward must be high
            self.reward -= 500
            done = True
        else:
            self.reward += 10

        # did we advance towards the x coordinate of location?
        # it's okay for robot to move away from final dest to avoid object, so consequence will be lower
        if (self.destination_x-self.position_x) < (self.destination_x-self.old_location_x):
            self.reward += 5
        else:
            self.reward -= 1

        # did we advance towards the y coordinate of location?
        if (self.destination_y-self.position_y) < (self.destination_y-self.old_location_y):
            self.reward +=5
        else:
            self.reward -=1

        # idk how to make sure that it doesnt fall off the edge
        # if(self.position_x>=4) or self.position_x<=-5:
        #     self.reward -= 500

        # if(self.position_y>=7) or self.position_y<=-2:
        #     self.reward -= 500


        self.observation_hold = [self.position_x, self.position_y, self.orientation, self.old_location_x, self.old_location_y, self.actions_taken] + list(self.detect)
        # self.observations = np.zeros((22,1))

        for i in range(0,8):
            self.observations[i] = self.observation_hold[i]

        pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
        print(pos_of_sensor)

        print(self.observations, self.reward)

        return (self.observations, self.reward, done, {}, {})

    def add_obstacle(self):
        print("we'll randomly add cubes later if time")

    def initProximity(self):
        # likely won't need anything other than res to see if something was detected and dist if something was
        
        self.usensors=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.detect=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        for i in range (0,16):
            self.usensors[i]=self.sim.getObject("./ultrasonicSensor",[i])
            # self.sim.setObjectInt32Param(usensors[i],_,_) # fix this error.

    def getProximity(self):
        for i in range (0,len(self.usensors)):
            res=self.sim.readProximitySensor(self.usensors[i])[0]
            dist=self.sim.readProximitySensor(self.usensors[i])[1]
            # if there is a detection (res is not 0) and we are close enough to object (dist < noDetectionDist)
            if (res>0) and (dist<self.noDetectionDist):
                if (dist<self.minDetectionDist):
                    dist=self.minDetectionDist
                    # I am not sure that we need these calculations. May just add dist?
                self.detect[i]=1-((dist-self.minDetectionDist)/(self.noDetectionDist-self.minDetectionDist))
            else:
                self.detect[i]=6 # this is how we ensure it doesnt mistake no detection for 0!!

        return self.detect
    
# speed of 1 for 10 seconds for 1 block forward
# speed of 1 for 2.9 seconds for 90 degree turn. 
# turns left if left wheel is negative speed

    def move(self, vLeft, vRight, move_time):
        # we'll need to specify vLeft and vRight
        # will also need to specify time we do this
        leftMotor = self.sim.getObject("./leftMotor")
        rightMotor = self.sim.getObject("./rightMotor")
        # no we use wait in this house
        # self.sim.backUntilTime = self.simg.getSimulationTime() + time

       
        self.sim.setJointTargetVelocity(leftMotor,vLeft)
        self.sim.setJointTargetVelocity(rightMotor,vRight)

        sleep(move_time)

        self.sim.setJointTargetVelocity(leftMotor,0)
        self.sim.setJointTargetVelocity(rightMotor,0)
        sleep(1)


    def close(self):
        self.sim.stopSimulation()
        self.sim.closeSene()

    def render(self):
            print("placeholder I guess")