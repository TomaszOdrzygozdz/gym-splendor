from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from monte_carlo_tree_search.mcts_algorithms.multi_process.multi_mcts import MultiMCTS
from monte_carlo_tree_search.rollout_policies.abstract_rolluot_policy import RolloutPolicy
from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer
from renders.render_paths import RENDER_DIR
import os

class MultiMCTSAgent(Agent):

    def __init__(self,
                 iteration_limit,
                 only_best,
                 rollout_policy: RolloutPolicy = None,
                 evaluation_policy: EvaluationPolicy = None,
                 exploration_coefficient: float = 0.4,
                 rollout_repetition = 10,
                 create_visualizer: bool=True,
                 show_unvisited_nodes = False):

        super().__init__(multi_process=True)
        self.rollout_policy = rollout_policy
        self.evaluation_policy = evaluation_policy
        self.iteration_limit = iteration_limit
        self.mcts_started = False
        self.mcts_initialized = False
        self.only_best = only_best
        self.name = 'Multi Process MCTS'
        self.visualize = create_visualizer
        self.rollout_repetition = rollout_repetition
        self.exploration_ceofficient = exploration_coefficient
        if self.visualize:
            self.visualizer = TreeVisualizer(show_unvisited_nodes=show_unvisited_nodes)

        self.previous_root_state = None
        self.previous_game_state = None
        self.actions_taken_so_far = 0
        self.color = 0
        self.self_play_mode = False

    def initialize_mcts(self):
        assert self.mpi_communicator is not None, 'You have to set mpi communiactor befor initializing MCTS.'
        self.mcts_algorithm = MultiMCTS(self.mpi_communicator, rollout_repetition=self.rollout_repetition,
                                        rollout_policy=self.rollout_policy,
                                        evaluation_policy=self.evaluation_policy,
                                        exploration_ceofficient=self.exploration_ceofficient)
        self.mcts_initialized = True
        self.main_process = self.mpi_communicator.Get_rank() == 0


    def deterministic_choose_action(self, observation : DeterministicObservation, previous_actions):

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

        root_is_terminal = None
        if self.main_process:
            root_is_terminal = self.mcts_algorithm.return_root().terminal
        root_is_terminal = self.mpi_communicator.bcast(root_is_terminal, root=0)
        if not root_is_terminal:
            self.mcts_algorithm.run_simulation(self.iteration_limit,self.rollout_repetition, self.only_best)
            if self.visualize and self.main_process:
                RENDER_FILE_ACTION = os.path.join(RENDER_DIR, 'color_{}_action_{}.html'.format(self.color, self.actions_taken_so_far))
                self.visualizer.generate_html(self.mcts_algorithm.return_root(), RENDER_FILE_ACTION)
            best_action = self.mcts_algorithm.choose_action()
            if best_action is not None:
                self.mcts_algorithm.move_root(best_action)
                self.actions_taken_so_far += 1
                self.draw_final_tree()

            return best_action
        else:
            return None

    def judge_observation(self, observation : DeterministicObservation):

        self.mcts_algorithm.create_root(observation)

        root_is_terminal = None
        if self.main_process:
            root_is_terminal = self.mcts_algorithm.return_root().terminal
        root_is_terminal = self.mpi_communicator.bcast(root_is_terminal, root=0)
        if not root_is_terminal:
            self.mcts_algorithm.run_simulation(self.iteration_limit,self.rollout_repetition, self.only_best)
            if self.visualize and self.main_process:
                RENDER_FILE_ACTION = os.path.join(RENDER_DIR, 'color_{}_decision_{}.html'.format(self.color, self.actions_taken_so_far))
                self.visualizer.generate_html(self.mcts_algorithm.return_root(), RENDER_FILE_ACTION)

            if self.main_process:
                return self.mcts_algorithm.mcts.root.value_acc.get()
            else:
                return None
        else:
            return None

    def draw_final_tree(self):
        if self.visualize and self.main_process:
            RENDER_FILE = os.path.join(RENDER_DIR, 'color_{}_full_game.html'.format(self.color))
            self.visualizer.generate_html(self.mcts_algorithm.original_root(), RENDER_FILE)

    def finish_game(self):
        '''When game is finished we need to clear out tree.'''
        if self.main_process:
            self.mcts_started = False
            self.actions_taken_so_far = 0
            self.previous_root_state = None
            self.previous_game_state = None
            self.actions_taken_so_far = 0


