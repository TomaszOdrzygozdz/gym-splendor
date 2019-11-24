from monte_carlo_tree_search.deterministic_mcts_multi_process import DeterministicMCTSMultiProcess


class MCTSMultiProcess:

    def __init__(self, comm):
        self.multi_process_mcts = DeterministicMCTSMultiProcess(comm)
        self.main_process = comm.Get_rank() == 0
        self.iterations_done_so_far = 0

    def create_root(self, state):
        self.multi_process_mcts.create_root(state)

    def return_root(self):
        return self.multi_process_mcts.mcts.root

