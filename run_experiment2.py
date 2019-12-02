from mpi4py import MPI

from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from agents.random_agent import RandomAgent
from arena.arena import Arena
from arena.multi_arena import MultiArena


my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

agent1 = RandomAgent(distribution='uniform')
agent2 = RandomAgent(distribution='uniform')
agent3 = RandomAgent(distribution='first_buy')
agent4 = MultiProcessMCTSAgent(200, 5, True)

arek = MultiArena()

fuf = arek.run_many_duels('deterministic', [agent4, agent3],n_games=1, n_proc_per_agent=10)
if main_process:
    print(fuf)