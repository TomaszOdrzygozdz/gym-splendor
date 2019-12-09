import time

from mpi4py import MPI

from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena


my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

agent1 = RandomAgent(distribution='uniform')
agent2 = RandomAgent(distribution='uniform')
agent3 = RandomAgent(distribution='first_buy')
agent4 = MultiProcessMCTSAgent(10, 2, False)

# agent5 = MultiProcessMCTSAgent(3, 5, True)
# agent6 = GeneralMultiProcessMCTSAgent(10, 2, True, False,
#                                         mcts = "rollout",
#                                         param_1 = "random",
#                                         param_2 = "uniform")


arek = MultiArena()
time_s = time.time()

import cProfile
pro = cProfile.Profile()

wyn = pro.run('arek.run_many_duels(\'deterministic\', [agent3, agent4], n_games=1, n_proc_per_agent=10)')
#fuf = arek.run_many_duels('deterministic', [agent3, agent4], n_games=1, n_proc_per_agent=10)
# if main_process:
#     print(fuf)
#     print('Time = {}'.format(time.time() - time_s))
wyn.dump_stats('stat.prof')