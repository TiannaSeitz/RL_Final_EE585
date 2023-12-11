# RL_Final_EE585
Final project for Principles of Robotics class. Goal: train an AmigoBot to avoid obstacles using reinforcement learning.

## Dependencies:
- Stable Baselines 3 (extra)
- OpenAI Gym
- CoppeliaSim API

## Files:
tjs1980_final_env.ttt - simulation environment for CoppeliaSim
sim_code.py - openAI Gym environment which contains step and reset functions for moving the robot an rewards
training.py - file run to train agent using sim_code environment and CoppeliaSim
deployment_sim.py - tests model (NOTE: THIS FILE IS UNTESTED)
semidemo_rl.py - proof of concept reinforcement learning file without coppeliaSim

## Known Issues:
Models seem to train indefinitely and do not save.
