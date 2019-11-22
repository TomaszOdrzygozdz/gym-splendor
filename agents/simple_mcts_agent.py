import random
from copy import deepcopy

from agent import Agent

from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.splendor import SplendorEnv, POINTS_TO_WIN
from mcts_alogrithms.tree_renderer.tree_visualizer import TreeVisualizer
from mcts_alogrithms.vanilla_mcts import FullStateVanillaMCTS


class SimpleMCTSAgent(Agent):

    action_number = 0

    def __init__(self,
                 iteration_limit = 10):

        super().__init__()
        self.mcts_algorithm = FullStateVanillaMCTS(iteration_limit=iteration_limit)
        self.mcts_started = False
        # we create own gym-splendor enivronemt to have access to its functionality
        # We specify the name of the agent
        self.name = 'MCTS'
        self.visualizer = TreeVisualizer(show_unvisited_nodes=False)


    def choose_action(self, observation):
        state_to_eval = self.env.observation_space.observation_to_state(observation)
        self.mcts_algorithm.create_root(state_to_eval)
        self.mcts_algorithm.run_simulation()
        return self.mcts_algorithm._select_best_child()

    def choose_action_knowing_state(self, state, opponents_action):
        if not self.mcts_started:
            self.mcts_algorithm.create_root(state)
            self.mcts_started = True
        if opponents_action is not None:
            print('MOVING OPPONENT')
            self.mcts_algorithm.move_root(opponents_action)
        SimpleMCTSAgent.action_number+= 1
        self.mcts_algorithm.run_simulation()
        self.visualizer.generate_html(self.mcts_algorithm.root, 'renders\\action_{}.html'.format(SimpleMCTSAgent.action_number))
        best_action = self.mcts_algorithm.choose_action()
        self.mcts_algorithm.move_root(best_action)
        self.draw_final_tree()
        print('Best action is:')
        print(best_action)
        return best_action

    def draw_final_tree(self):
        self.visualizer.generate_html(self.mcts_algorithm.original_root, 'renders\\full_game.html')

