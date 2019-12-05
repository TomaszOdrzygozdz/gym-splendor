from mpi4py import MPI

from agents.dense_nn_agent import DenseNNAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena

my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

agent1 = RandomAgent(distribution='first_buy')
agent1a = RandomAgent(distribution='first_buy')
agent2 = MinMaxAgent(collect_stats=True)
agent3 = DenseNNAgent('agents\weights\minmax_20_games_depth_3.h5')
#agent3a = DenseNNAgent()
agent4 = GreedyAgentBoost()

arek = MultiArena()

#training

print('making data with random first buy')



arek.run_many_duels('deterministic', [agent1, agent2], n_games=2, n_proc_per_agent=24)
agent2.dump_action_scores('data/200_games_against_random_first_buy_my_rank_{}'.format(my_rank))

print('making data with greedy boost')
arek.run_many_duels('deterministic', [agent4, agent2], n_games=2, n_proc_per_agent=24)
agent2.dump_action_scores('data/200_games_against_greedy_boost_my_rank_{}'.format(my_rank))


#testing
# stats = arek.run_many_duels('deterministic', [agent3, agent4], n_games=20, n_proc_per_agent=1)
#
# if main_process:
#     print(stats)
