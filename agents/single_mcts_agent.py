import time
import neptune

from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from monte_carlo_tree_search.mcts_algorithms.multi_process.multi_mcts import MultiMCTS
from monte_carlo_tree_search.mcts_algorithms.single_process.single_mcts import SingleMCTS
from monte_carlo_tree_search.rollout_policies.abstract_rolluot_policy import RolloutPolicy
from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer
from renders.render_paths import RENDER_DIR
import os

class SingleMCTSAgent(Agent):

    def __init__(self,
                 iteration_limit,
                 evaluation_policy: EvaluationPolicy = None,
                 exploration_parameter: float = 0.4,
                 create_visualizer: bool=True,
                 show_unvisited_nodes = False,
                 log_to_neptune: bool = False):

        super().__init__()
        self.evaluation_policy = evaluation_policy
        self.iteration_limit = iteration_limit
        self.mcts_started = False
        self.mcts_initialized = False
        self.name = 'Single Process MCTS'
        self.visualize = create_visualizer
        self.exploration_parameter = exploration_parameter
        if self.visualize:
            self.visualizer = TreeVisualizer(show_unvisited_nodes=show_unvisited_nodes)

        self.previous_root_state = None
        self.previous_game_state = None
        self.actions_taken_so_far = 0
        self.color = 0
        self.self_play_mode = False
        self.log_to_neptune = log_to_neptune
        if self.log_to_neptune:
            self.action_number = 0


    def initialize_mcts(self):
        self.mcts_algorithm = SingleMCTS(iteration_limit=self.iteration_limit,
                                         exploration_parameter=self.exploration_parameter,
                                         evaluation_policy=self.evaluation_policy)

        self.mcts_initialized = True

    def load_weights(self, weights_file):
        self.mcts_algorithm.evaluation_policy.load_weights(weights_file)

    def dump_weights(self, weights_file):
        self.mcts_algorithm.evaluation_policy.dump_weights(weights_file)


    def deterministic_choose_action(self, observation : DeterministicObservation, previous_actions):

        if self.log_to_neptune:
            start_time = time.time()
        assert observation.name == 'deterministic', 'You must provide deterministic observation'
        ignore_previous_action = False
        if not self.mcts_initialized:
            self.initialize_mcts()
        if not self.mcts_started:
            self.mcts_algorithm.create_root(observation)
            self.mcts_started = True
            ignore_previous_action = True

        if not self.self_play_mode:
            if not ignore_previous_action and self.main_process:
                if previous_actions[0] is not None and self.main_process:
                    if self.mcts_started and self.main_process:
                        if not self.mcts_algorithm.return_root().terminal:
                            self.mcts_algorithm.move_root(previous_actions[0])


        root_is_terminal = self.mcts_algorithm.return_root().terminal
        if not root_is_terminal:
            self.mcts_algorithm.run_simulation(self.iteration_limit)
            best_action = self.mcts_algorithm.choose_action()
            if self.visualize and self.main_process:
                RENDER_FILE_ACTION = os.path.join(RENDER_DIR, 'color_{}_action_{}.html'.format(self.color, self.actions_taken_so_far))
                self.visualizer.generate_html(self.mcts_algorithm.return_root(), RENDER_FILE_ACTION)

            if best_action is not None:
                self.mcts_algorithm.move_root(best_action)
                self.actions_taken_so_far += 1
                self.draw_final_tree()

            if self.log_to_neptune:
                self.action_number += 1
                neptune.log_metric('Time for action',x = self. action_number, y=time.time() - start_time)
            return best_action
        else:
            if self.log_to_neptune:
                self.action_number += 1
                neptune.log_metric('Time for action',x = self. action_number, y=time.time() - start_time)
            return None

    # def judge_observation(self, observation : DeterministicObservation):
    #
    #     self.mcts_algorithm.create_root(observation)
    #
    #     root_is_terminal = None
    #     if self.main_process:
    #         root_is_terminal = self.mcts_algorithm.return_root().terminal
    #     root_is_terminal = self.mpi_communicator.bcast(root_is_terminal, root=0)
    #     if not root_is_terminal:
    #         self.mcts_algorithm.run_simulation(self.iteration_limit,self.rollout_repetition, self.only_best)
    #         if self.visualize and self.main_process:
    #             RENDER_FILE_ACTION = os.path.join(RENDER_DIR, 'color_{}_decision_{}.html'.format(self.color, self.actions_taken_so_far))
    #             self.visualizer.generate_html(self.mcts_algorithm.return_root(), RENDER_FILE_ACTION)
    #
    #         if self.main_process:
    #             return self.mcts_algorithm.mcts.root.value_acc.get()
    #         else:
    #             return None
    #     else:
    #         return None

    def draw_final_tree(self):
        if self.visualize and self.main_process:
            RENDER_FILE = os.path.join(RENDER_DIR, 'color_{}_full_game.html'.format(self.color))
            self.visualizer.generate_html(self.mcts_algorithm.original_root, RENDER_FILE)

    def finish_game(self):
        '''When game is finished we need to clear out tree.'''
        if self.main_process:
            self.mcts_started = False
            self.actions_taken_so_far = 0
            self.previous_root_state = None
            self.previous_game_state = None
            self.actions_taken_so_far = 0




