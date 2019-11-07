import random
from copy import deepcopy

from agent import Agent
from mcts_alogrithms.poor_mcts import MCTS

from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.splendor import SplendorEnv, POINTS_TO_WIN


class VanillaMCTSAgent(Agent):

    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        # we create own gym-splendor enivronemt to have access to its functionality
        # We specify the name of the agent
        self.name = 'MCTS {} steps' + str(self.steps)


    def choose_action(self, observation):
        state_to_eval = self.env.observation_space.observation_to_state(observation)
        state_intreface_mcts_to_eval = StateInterfaceMCTS(state_to_eval)
        local_mcts = MCTS(iterationLimit=self.steps)
        action = local_mcts.search(initialState=state_intreface_mcts_to_eval)
        print(action)
        return action

    def choose_from_state(self, state_to_eval):
        state_intreface_mcts_to_eval = StateInterfaceMCTS(state_to_eval)
        local_mcts = MCTS(iterationLimit=self.steps)
        print('hej')
        return local_mcts.search(initialState=state_intreface_mcts_to_eval)
