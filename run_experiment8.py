import time

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent

from mpi4py import MPI

from agents.randomized_agent import RandomizedAgent
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
agent4a = RandomizedAgent(epsilon=0.7)
agent4b = RandomizedAgent(epsilon = 0.7)
# agent3 = RandomAgent(distribution = 'first_buy')
#
#
# multi_arena = ArenaMultiThread()
# multi_arena.start_collecting_states()
# results = multi_arena.all_vs_all('deterministic', [agent4a, agent4b], n_games=300)
# multi_arena.dump_collected_states('epsilon_07')
# multi_arena.collected_states_to_csv('epsilon_07')
# multi_arena.stop_collecting_states()
# print(results)


fufix = Judger(5)
time_s = time.time()
fufix.judge_dataframe('epsilon_07', 2)
print('Time taken = {}'.format(time.time() - time_s))


#
# res = fufix.judge_observation(x, 5)
# print(res)

# r = []
# for i in range(50):
#     fufix.judge_observation(x)
#     r =s

