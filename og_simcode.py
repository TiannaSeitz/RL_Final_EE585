#python
import math
import numpy as np


usensors=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
noDetectionDist=.75
maxDetectionDist=0.3
detect=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
braitenbergL=[-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
braitenbergR=[-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
v0=2

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
