#python

#################################
# File:     environment.py
# Purpose:  create an reinforcement learning environment
#           where an AmigoBot can be trained to navigate 
#           to a designated location while avoiding
#           obstacles

# Author:   Tianna Seitz
# Released: 12/11/2023
#
# Notes: Version is ready for use
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
        # this range is estimate recieved from real simulation coordinates
        self.destination_x = 1.6
        self.destination_y =  -0.35
        self.destination_x_max = 2.6
        self.destination_y_max = 0.6
        
        self.action_space = Discrete(4) # 4 possible actions 0-3
        self.actions_max = 15 # terminate after 15 actions

        self.reward = 0
        self.done = False
        self.observations = np.zeros((6,1))

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
        self.orientation =  90
        self.actions_taken =  0

        # okay so we don't declare what the starting actions are here, we just set up the spaces for the stuff to be put into.
        
        self.observation_space = Box(low = -255, high = 255, shape = (6,1), dtype = np.float32) # may need to specify type!

    def reset(self): # may need to be revised...
        print(f"final reward = {self.reward}")
        with open('test.txt', 'a') as file:
            stringInput = str(self.reward)
            file.write("\n")
            file.write(stringInput)

        file.close()
        self.sim.stopSimulation()
        sleep(1)
        self.sim.closeScene()
        sleep(1)

        # do we need an explicit stepping call or is that already taken care of in init?

        self.sim.loadScene('/home/mabl/tianna_ws/RL_Final_EE585/tjs1980_final_env.ttt')
        self.sim.startSimulation()  

        self.initProximity()
        self.getProximity()
        pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
        print(pos_of_sensor)

        self.position_x = pos_of_sensor[0]
        self.position_y = pos_of_sensor[1]
        self.orientation =  90
        self.actions_taken =  0
        self.reward = 0
        
        self.observation_hold = [self.position_x, self.position_y, self.orientation, self.actions_taken, self.detect[3], self.detect[4]]

        # put observations into array compatible with space
        for i in range(0,6):
            self.observations[i] = self.observation_hold[i]
        return self.observations, {}
    
    def step(self, action):
        self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])

        self.old_location_y = self.pos_of_sensor[1]
        self.old_location_x = self.pos_of_sensor[0]

        self.done = False
        done = self.done

        offset = 0.05

        if self.actions_taken == self.actions_max: # if we run out of actions, thats really bad
            self.reward -= 250
            print("ran out of time")
            done = True
        else:
            done = False

        self.actions_taken += 1

        # we will need a function to help the robot turn if it is not facing the direction it wants to move
        # move forward will be positive, backwards is negative motion
        # move right is positive, move left is negative. 
        # robot front will either face forward or right.
        # moving backwards or left will always be reverse motion

        # left right time
        if action == 0: # move forward
            if self.orientation != 90:
                # turn robot
                self.move(-1, 1, 2.9)
                orientation = 90
            # move forward
            self.move(1, 1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_y = self.pos_of_sensor[1]
            self.position_x = self.pos_of_sensor[0]

        elif action == 1: # move backwards
            if self.orientation != 90:
                # turn robot
                self.move(-1, 1, 2.9)
                orientation = 90
            # move backwards
            self.move(-1, -1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_y = self.pos_of_sensor[1]
            self.position_x = self.pos_of_sensor[0]

        elif action == 2: # move left
            if self.orientation != 0:
                # turn robot
                self.move(1, -1, 2.9)
                orientation = 0
            # move backwards (left)
            self.move(-1, -1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_y = self.pos_of_sensor[1]
            self.position_x = self.pos_of_sensor[0]

        elif action == 3: # move right
            if self.orientation != 0:
                # turn robot
                self.move(1, -1, 2.9)
                orientation = 0
            # move foward (right)
            self.move(1, 1, 10)
            self.pos_of_sensor = self.sim.getObjectPosition(self.usensors[3])
            self.position_y = self.pos_of_sensor[1]
            self.position_x = self.pos_of_sensor[0]

        # self.client.step() # I have no clue why this is here. Is it necessary?
        ultrasonic_result = self.getProximity()
        pos_done_x = False
        pos_done_y = False

        if self.position_x > self.destination_x and self.position_x < self.destination_x_max: # end goal is to end up at this x,y location
            self.reward += 50
            print("reached location x")
            pos_done_x = True
        else:
            if abs(abs(self.destination_x_max)-abs(self.position_x)) < abs(abs(self.destination_x_max)-abs(self.old_location_x)):
                self.reward += 20
            else:
                self.reward -= 30 # striggles to reach x so a greater punishment is given for moving away from x

            
        if self.position_y > self.destination_y and self.position_y < self.destination_y_max: # end goal is to end up at this x,y location # end goal is to end up at this x,y location
            self.reward += 50
            print("reached location y")
            pos_done_y = True
        else:
            # did we advance towards the y coordinate of location?
            if abs(abs(self.destination_y_max)-abs(self.position_y)) < abs(abs(self.destination_y_max)-abs(self.old_location_y)):
                self.reward +=20
            else:
                self.reward -=10 

        # check to see if we are at our final destination or just one component of our destinations
        if pos_done_y == True and pos_done_x == True:
            self.reward + 500
            done = True
        else:
            if pos_done_x == True and pos_done_y == False:
                self.reward -= 10

            elif pos_done_x == False and pos_done_y == True:
                self.reward -= 10

        # are we too close to an object?
        if ultrasonic_result[3] < self.minDetectionDist-offset:
            # avoiding objects is the most important task consequence of reward must be high
            self.reward -= 500
            done = True
        else:
            self.reward += 5

        if ultrasonic_result[4] < self.minDetectionDist-offset:
            # avoiding objects is the most important task consequence of reward must be consequential
            self.reward -= 500
            done = True
        else:
            self.reward += 10

        # making sure robot doesnt fall off the edge... (or get stuck on something for too long)
        if self.position_y<= -4.7 or self.position_y>=4.7:
            self.reward -= 500

        if self.position_x<= -4.7 or self.position_x>=4.7:
            self.reward -= 500

        # self.observation_hold = [self.position_x, self.position_y, self.orientation, self.actions_taken] + list(self.detect)
        self.observation_hold = [self.position_x, self.position_y, self.orientation, self.actions_taken, self.detect[3], self.detect[4]]
        # self.observations = np.zeros((22,1))

        for i in range(0,6):
            self.observations[i] = self.observation_hold[i]

        print(f"{self.position_x}, {self.position_y}")
        print(self.observations, self.reward)

        return (self.observations, self.reward, done, {}, {})

    def initProximity(self):
        # likely won't need anything other than res to see if something was detected and dist if something was
        
        self.usensors=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.detect=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        for i in range (0,16):
            self.usensors[i]=self.sim.getObject("./ultrasonicSensor",[i])

    def getProximity(self):
        for i in range (0,len(self.usensors)):
            res=self.sim.readProximitySensor(self.usensors[i])[0]
            dist=self.sim.readProximitySensor(self.usensors[i])[1]
            # if there is a detection (res is not 0) and we are close enough to object (dist < noDetectionDist)
            if (res>0) and (dist<self.noDetectionDist):
                if (dist<self.minDetectionDist):
                    dist=self.minDetectionDist
                    # The following calculations came from coppeliaSim
                self.detect[i]=1-((dist-self.minDetectionDist)/(self.noDetectionDist-self.minDetectionDist))
            else:
                self.detect[i]=6 # this is how we ensure it doesnt mistake no detection for 0!!

        return self.detect
    
# speed of 1 for 10 seconds for 1 block forward
# speed of 1 for 2.9 seconds for 90 degree turn. 
# turns left if left wheel is negative speed

    def move(self, vLeft, vRight, move_time):
        # we'll need to pass in vLeft and vRight
        # will also need to pass in duration me we do this
        leftMotor = self.sim.getObject("./leftMotor")
        rightMotor = self.sim.getObject("./rightMotor")
       
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
            print("placeholder for later I guess")