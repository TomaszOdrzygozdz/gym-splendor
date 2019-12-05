from mpi4py import MPI

from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena

my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

agent1 = RandomAgent(distribution='first_buy')
agent1a = RandomAgent(distribution='first_buy')
agent2 = MinMaxAgent(collect_stats=True)

arek = MultiArena()

stats = arek.run_many_duels('deterministic', [agent1, agent2], n_games=20, n_proc_per_agent=1)

if main_process:
    print(stats)

agent2.dump_action_scores('minmax_data')
