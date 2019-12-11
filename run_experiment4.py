import os

from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
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
agent2 = MinMaxAgent(collect_stats=False)
agent3 = GreedyAgentBoost()
agent4 = GreedySearchAgent()
agent_mcts = MultiProcessMCTSAgent(2, rollout_policy=RandomRollout(), rollout_repetition=1, create_visualizer=True)


arek = MultiArena()

n_repetition = 1

for rep in range(n_repetition):
    arek.run_many_duels('deterministic', [agent_mcts, agent1], n_games=1, n_proc_per_agent=400)
    if main_process:
        data_collector = TreeDataCollector(agent_mcts.mcts_algorithm.original_root())
        data_collector.dump_data('against_random_{}_'.format(n_repetition))

    arek.run_many_duels('deterministic', [agent_mcts, agent2], n_games=1, n_proc_per_agent=400)
    if main_process:
        data_collector = TreeDataCollector(agent_mcts.mcts_algorithm.original_root())
        data_collector.dump_data('against_minmax_{}_'.format(n_repetition))

    arek.run_many_duels('deterministic', [agent_mcts, agent3], n_games=1, n_proc_per_agent=400)
    if main_process:
        data_collector = TreeDataCollector(agent_mcts.mcts_algorithm.original_root())
        data_collector.dump_data('against_greedy_boost_{}_'.format(n_repetition))

    arek.run_many_duels('deterministic', [agent_mcts, agent2], n_games=1, n_proc_per_agent=400)
    if main_process:
        data_collector = TreeDataCollector(agent_mcts.mcts_algorithm.original_root())
        data_collector.dump_data('against_greedy_search_{}_'.format(n_repetition))
