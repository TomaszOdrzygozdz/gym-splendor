from abc import abstractmethod

import gym


class Agent:
    """An abstract class to create agents for playing Splendor."""
    """Every agent must have specified name."""
    agents_created = 0
    def __init__(self, environment_id: str = 'gym_splendor_code:splendor-v0') -> None:
        """Every agent has its private environment to check legal actions, make simulations etc."""
        self.env = gym.make(environment_id)
        Agent.agents_created += 1
        self.name = 'Abstract agent ' + str(Agent.agents_created)

    @abstractmethod
    def choose_action(self, observation):
        """This method chooses one action to take, based on the provided observation. This method should not have
        access to the original gym-splendor environment - you can create your own environment for example to do
        simulations of game, or have access to environment methods."""
        raise NotImplementedError


    def __repr__(self):
        return self.name
