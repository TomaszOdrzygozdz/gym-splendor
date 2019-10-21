#This example shows how to use gym-Splendor environment.
import gym

#First we create the environment
env = gym.make('gym_splendor_code:splendor-v0')

#Let's see what is the observation space:
print('Observation space: \n {} \n'.format(env.observation_space))
#Let's see what is the action space:
print('Action space: \n {} \n'.format(env.action_space))

env.render()