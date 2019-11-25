from agent import Agent
from monte_carlo_tree_search.deterministic_mcts_multi_process import DeterministicMCTSMultiProcess
from monte_carlo_tree_search.mcts_algorithms.deterministic_vanilla_mcts import DeterministicVanillaMCTS
from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer




class MultiProcessMCTSAgent(Agent):

    def __init__(self,
                 iteration_limit,
                 create_visualizer: bool=True):

        super().__init__(multi_process=True)
        self.iteration_limit = iteration_limit
        self.mcts_started = False
        self.mcts_initialized = False
        self.name = 'Multi Process MCTS'
        self.visualize = create_visualizer
        if self.visualize:
            self.visualizer = TreeVisualizer(show_unvisited_nodes=True)

        self.actions_taken_so_far = 0

    def initialize_mcts(self, mpi_communicator):
        assert self.mpi_communicator is not None, 'You have to set mpi communiactor befor initializing MCTS.'
        self.mcts_algorithm = DeterministicMCTSMultiProcess(mpi_communicator)
        self.mcts_initialized = True
        self.main_process = mpi_communicator.Get_rank() == 0

    def deterministic_choose_action(self, state, previous_actions):

        print('MCTS AGENT CHOOSING ACTION')

        if not self.mcts_initialized:
            self.initialize_mcts(self.mpi_communicator)

        if self.mcts_started and self.main_process:
            print('Root state {}'.format(self.mcts_algorithm.return_root().state_as_dict))
            if not self.mcts_algorithm.return_root().terminal:
                self.mcts_algorithm.move_root(previous_actions[0])

        if not self.mcts_started:
            self.mcts_algorithm.create_root(state)
            self.mcts_started = True

        root_is_terminal = False
        if self.main_process:
            root_is_terminal = self.mcts_algorithm.return_root().terminal
        root_is_terminal = self.mpi_communicator.bcast(root_is_terminal, root=0)

        if not root_is_terminal:

            self.mcts_algorithm.run_simulation(self.iteration_limit,5)

            if self.visualize and self.main_process:
                print('html made')
                self.visualizer.generate_html(self.mcts_algorithm.return_root(), 'renders\\action_{}.html'.format(self.actions_taken_so_far))
            best_action = self.mcts_algorithm.choose_action()
            self.mcts_algorithm.move_root(best_action)
            self.actions_taken_so_far += 1
            return best_action
        else:
            print('Root is terminal')
            return None

    def draw_final_tree(self):
        if self.visualize:
            self.visualizer.generate_html(self.mcts_algorithm.original_root, 'renders\\full_game.html')

