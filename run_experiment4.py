import os

from agents.multi_process_mcts_agent import MultiMCTSAgent
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from nn_models.tree_data_collector import TreeDataCollector


from mpi4py import MPI

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena

my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

agent1 = RandomAgent(distribution='first_buy')
agent2 = GreedyAgentBoost()

#diffetent mcts_levels:

mcts_lvl_1 = MultiMCTSAgent(20, rollout_policy=RandomRollout(), rollout_repetition=10, create_visualizer=True, only_best=None)
mcts_lvl_2 = MultiMCTSAgent(60, rollout_policy=RandomRollout(), rollout_repetition=10, create_visualizer=True, only_best=None)
mcts_lvl_3 = MultiMCTSAgent(150, rollout_policy=RandomRollout(), rollout_repetition=10, create_visualizer=True, only_best=None)
mcts_lvl_4 = MultiMCTSAgent(480, rollout_policy=RandomRollout(), rollout_repetition=10, create_visualizer=True, only_best=None)

list_of_mcts_agents = [mcts_lvl_1, mcts_lvl_2, mcts_lvl_3, mcts_lvl_4]

arek = MultiArena()

n_repetition = 1

for i, mcts_agent in enumerate(list_of_mcts_agents):
    current_repetitions = n_repetition + max(20-5*i, 0)
    for rep in range(current_repetitions):
        # arek.run_many_duels('deterministic', [mcts_agent, agent1], n_games=1, n_proc_per_agent=400)
        # if main_process:
        #     data_collector = TreeDataCollector()
        #     data_collector.setup_root(mcts_agent.mcts_algorithm.original_root())
        #     data_collector.generate_dqn_data()
        #     data_collector.dump_data('lvl_{}_against_random_{}_'.format(i, rep))
        print('lvl_{}_against_random_{}_'.format(i, rep))

        # arek.run_many_duels('deterministic', [mcts_agent, agent2], n_games=1, n_proc_per_agent=400)
        # if main_process:
        #     data_collector = TreeDataCollector()
        #     data_collector.setup_root(mcts_agent.mcts_algorithm.original_root())
        #     data_collector.generate_dqn_data()
        #     data_collector.dump_data('lvl_{}_against_greedyBoost_{}_'.format(i, rep))
