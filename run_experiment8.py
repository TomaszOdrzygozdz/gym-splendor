import time

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.multi_process_mcts_agent import MultiMCTSAgent

from mpi4py import MPI

from agents.state_judger import Judger
from arena.multi_arena import MultiArena
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from trainers.states_list import state_4, obs4, obs3, obs2, obs1, obs1_1, obs1_2, obs0

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
print(my_rank)


from agents.random_agent import RandomAgent
from arena.arena_multi_thread import ArenaMultiThread



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
# multi_arena = ArenaMultiThread()
# multi_arena.start_collecting_states()
# results = multi_arena.all_vs_all('deterministic', [agent3a, agent3b], n_games=2)
# multi_arena.dump_collected_states('dupix')
# multi_arena.collected_states_to_csv('dupix')
# multi_arena.stop_collecting_states()
# print(results)


# multi_arena = MultiArena()
# multi_arena.run_many_duels('deterministic', [agent2, agent3], 1, 5, False, )

fufix = Judger(5)
time_s = time.time()
fufix.judge_dataframe('dupix', 2)
print('Time taken = {}'.format(time.time() - time_s))
#
# x = obs0

# res = fufix.judge_observation(x, 5)
# print(res)

# r = []
# for i in range(50):
#     fufix.judge_observation(x)
#     r =s

