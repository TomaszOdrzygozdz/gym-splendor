from abc import abstractmethod

import gym


class Agent:
    """An abstract class to create agents for playing Splendor."""
    """Every agent must have specified name."""

    def __init__(self, environment_id: str = 'gym_splendor_code:splendor-v0') -> None:
        """Every agent has its private environment to check legal actions, make simulations etc."""
        self.env = gym.make(environment_id)
        self.name = 'Abstract agent'

    @abstractmethod
    def choose_action(self, observation):
        """This method chooses one action to take, based on the provided observation. This method should not have
        access to the original gym-splendor environment - you can create your own environment for example to do
        simulations of game, or have access to environment methods."""
        raise NotImplementedError