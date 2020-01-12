import time

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent

from mpi4py import MPI

from agents.state_judger import Judger
from arena.arena_multi_thread import ArenaMultiThread

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
print(my_rank)


from agents.random_agent import RandomAgent

#
agent1a = RandomAgent(distribution = 'uniform')
agent1b = RandomAgent(distribution = 'first_buy')
agent2a = GreedyAgentBoost()
agent2b = GreedyAgentBoost()
agent3a = GreedySearchAgent()
agent3b = GreedySearchAgent()
# agent3 = RandomAgent(distribution = 'first_buy')
#
#
multi_arena = ArenaMultiThread()
multi_arena.start_collecting_states()
results = multi_arena.all_vs_all('deterministic', [agent3a, agent3b], n_games=1200)
multi_arena.dump_collected_states('part2')
multi_arena.collected_states_to_csv('part2')
multi_arena.stop_collecting_states()
print(results)


# fufix = Judger(5)
# time_s = time.time()
# fufix.judge_dataframe('part2', 2)
# print('Time taken = {}'.format(time.time() - time_s))
# #
# x = obs0

# res = fufix.judge_observation(x, 5)
# print(res)

# r = []
# for i in range(50):
#     fufix.judge_observation(x)
#     r =s

