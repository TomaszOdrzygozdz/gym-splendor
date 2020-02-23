from mpi4py import MPI

from agents.random_agent import RandomAgent
from agents.value_function_agent import ValueFunctionAgent
from arena.arena_multi_thread import ArenaMultiThread
from arena.multi_arena import MultiArena

comm = MPI.COMM_WORLD
main_process = comm.Get_rank() == 0

class ValueFunctionOptimizer:
    def __init__(self):
        self.arena = ArenaMultiThread()
        self.vf_agent = ValueFunctionAgent()
        self.opp = RandomAgent(distribution='first_buy')

    def eval_metrics(self, n_games):
        results = self.arena.run_many_games('deterministic', [self.vf_agent, self.opp], n_games)
        if main_process:
            _, _, win_rate = results.return_stats()
            return win_rate/n_games
        return None

    def eval_gradient(self, n_games, idx):
        pass




