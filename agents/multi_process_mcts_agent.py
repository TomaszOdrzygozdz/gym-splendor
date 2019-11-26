from agent import Agent
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.mcts_algorithms.deterministic_mcts_multi_process import DeterministicMCTSMultiProcess
from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer




class MultiProcessMCTSAgent(Agent):

    def __init__(self,
                 iteration_limit,
                 create_visualizer: bool=True,
                 show_unvisited_nodes = False):

        super().__init__(multi_process=True)
        self.iteration_limit = iteration_limit
        self.mcts_started = False
        self.mcts_initialized = False
        self.name = 'Multi Process MCTS'
        self.visualize = create_visualizer
        if self.visualize:
            self.visualizer = TreeVisualizer(show_unvisited_nodes=show_unvisited_nodes)

        self.previous_root_state = None
        self.previous_game_state = None

        self.actions_taken_so_far = 0

    def initialize_mcts(self, mpi_communicator):
        assert self.mpi_communicator is not None, 'You have to set mpi communiactor befor initializing MCTS.'
        self.mcts_algorithm = DeterministicMCTSMultiProcess(mpi_communicator)
        self.mcts_initialized = True
        self.main_process = mpi_communicator.Get_rank() == 0

    def deterministic_choose_action(self, state, previous_actions):
        print("Start deterministic")
        ignore_previous_action = False
        if not self.mcts_initialized:
            self.initialize_mcts(self.mpi_communicator)
        print("MCTS initialized")
        if not self.mcts_started:
            print(' STARTING MCTS')
            self.mcts_algorithm.create_root(state)
            print('STARTING STATE FOR MCTS = {}'.format(state.to_dict()))
            self.mcts_started = True
            ignore_previous_action = True

        if not ignore_previous_action and self.main_process:
            if previous_actions[0] is not None and self.main_process:
                if self.mcts_started and self.main_process:
                    if not self.mcts_algorithm.return_root().terminal:
                        self.mcts_algorithm.move_root(previous_actions[0])
        print("root moved")
        rootek = self.mcts_algorithm.return_root()
        if self.main_process:
            if rootek.state.to_dict() != state.to_dict():
                import pdb;pdb.set_trace()
                print('Dupa')
                print('PREVIOUS ROOT_STATE')
                assert False, 'COINS DO NOT MATCH'

        if self.main_process:
            #assert self.mcts_algorithm.return_root().state.board.gems_on_board == state.board.gems_on_board, 'Coins do not match'
            if not self.mcts_algorithm.return_root().state.board.gems_on_board == state.board.gems_on_board:
                print('a')
                assert False

        root_is_terminal = None
        if self.main_process:
            root_is_terminal = self.mcts_algorithm.return_root().terminal
        root_is_terminal = self.mpi_communicator.bcast(root_is_terminal, root=0)
        print('is root terminal?')
        if not root_is_terminal:
            self.mcts_algorithm.run_simulation(self.iteration_limit,1)
            if self.visualize and self.main_process:
                self.visualizer.generate_html(self.mcts_algorithm.return_root(), 'renders\\action_{}.html'.format(self.actions_taken_so_far))
            best_action = self.mcts_algorithm.choose_action()

            self.mcts_algorithm.move_root(best_action)
            self.actions_taken_so_far += 1
            self.draw_final_tree()

            if self.main_process:
                print('STATE OF MCTS ROOT AFTER TAKING ACTION = {} \n ACTION DONE BY MCTS IS = {} '.format(self.mcts_algorithm.return_root().state.to_dict(), best_action))

            return best_action
        else:
            return None

    def draw_final_tree(self):
        if self.visualize and self.main_process:
            self.visualizer.generate_html(self.mcts_algorithm.original_root(), 'renders\\full_game.html')

