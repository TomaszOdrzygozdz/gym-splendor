import gym

from agent import Agent


class RandomAgent(Agent):

    def __init__(self, distribution='uniform_on_types'):
        """:param:
        distribution: argument determining how action is chosen at random. Possible """

        self.distribution = distribution
        #we create own gym-splendor enivronemt to have access to its functionality
        self.env = gym.make('gym_splendor_code:splendor-v0')


    def choose_action(self, observation):
        pass
