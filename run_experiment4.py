from mpi4py import MPI

from agents.dense_nn_agent import DenseNNAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
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
agent5 = GreedySearchAgent()
agent6 = MinMaxAgent(collect_stats=False)

arek = MultiArena()

#training

arek.run_many_duels('deterministic', [agent5, agent2], n_games=240, n_proc_per_agent=1)
agent2.dump_action_scores('nn_models/data/200_games_against_greedy_search_my_rank_{}'.format(my_rank))

arek.run_many_duels('deterministic', [agent6, agent2], n_games=240, n_proc_per_agent=1)
agent2.dump_action_scores('nn_models/data/200_games_against_minmax_my_rank_{}'.format(my_rank))


#testing
# stats = arek.run_many_duels('deterministic', [agent3, agent4], n_games=20, n_proc_per_agent=1)
#
# if main_process:
#     print(stats)
