import random
from copy import deepcopy

from agent import Agent

from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.splendor import SplendorEnv, POINTS_TO_WIN




class SimpleMCTSAgent(Agent):

    def __init__(self,
                 mcts_algorithm):

        super().__init__()
        self.mcts_algorithm = mcts_algorithm
        # we create own gym-splendor enivronemt to have access to its functionality
        # We specify the name of the agent
        self.name = 'MCTS'


    def choose_action(self, observation):
        state_to_eval = self.env.observation_space.observation_to_state(observation)
        print('Root state:')
        self.mcts_algorithm.create_root(state_to_eval)
        self.mcts_algorithm.run_simulation()
        return self.mcts_algorithm.choose_best_action()
