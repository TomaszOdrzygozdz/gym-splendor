from abc import abstractmethod
from typing import List


import gym
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State


class Agent:
    """An abstract class to create agents for playing Splendor."""
    """Every agent must have specified name."""
    agents_created = 0
    def __init__(self, environment_id: str = 'gym_splendor_code:splendor-v0', multi_process: bool = False,
                 mpi_comunicator=None) -> None:

        self.multi_process = multi_process
        self.mpi_communicator = mpi_comunicator
        if mpi_comunicator is None:
            self.main_process = True
        else:
            self.main_process = mpi_comunicator.Get_rank() == 0

        """Every agent has its private environment to check legal actions, make simulations etc."""
        if self.main_process:
            self.env = gym.make(environment_id)
        self.name = 'Abstract agent '
        #id is uded to distinguish between two agents of the same type
        if self.main_process:
            Agent.agents_created += 1
            self.id = Agent.agents_created


    @abstractmethod
    def choose_action(self, observation, previous_actions : List[Action]):
        """This method chooses one action to take, based on the provided observation. This method should not have
        access to the original gym-splendor environment - you can create your own environment for example to do
        simulations of game, or have access to environment methods."""
        raise NotImplementedError

    def deterministic_choose_action(self, state: State, previous_actions : List[Action]):
        """This method is used for games with no randomness."""
        raise NotImplementedError

    def __repr__(self):
        return self.name

    def my_name_with_id(self):
        return self.name + ' (' + str(self.id) + ')'


    def set_communicator(self, mpi_communicator):
        if self.multi_process:
            self.mpi_communicator = mpi_communicator
        else:
            print('This agent does not need mpi communicator')

    def finish_game(self):
        pass