from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
from arena.arena import Arena
from monte_carlo_tree_search.evaluation_policies.heuristic_value_policy import HeuristicValuePolicy
from nn_models.tree_data_collector import TreeDataCollector

data_collector = TreeDataCollector()

mcts_agent = MultiMCTSAgent(10, 1, None, HeuristicValuePolicy(), 0.4, 0, True, False)
opp_agent = RandomAgent(distribution='first_buy')

arena = Arena()
results = arena.run_one_duel('deterministic', [mcts_agent, opp_agent])

data_collector.setup_root(mcts_agent.mcts_algorithm.original_root())
tree_data = data_collector.generate_all_tree_data()
data_collector.dump_data('danko.csv')

print(results)
print(tree_data)
