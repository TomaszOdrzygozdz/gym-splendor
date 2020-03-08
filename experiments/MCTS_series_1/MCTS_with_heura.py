from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
from arena.arena import Arena
from monte_carlo_tree_search.evaluation_policies.heuristic_value_policy import HeuristicValuePolicy

mcts_agent = MultiMCTSAgent(50, 1, None, HeuristicValuePolicy(), 0.4, 0, True, False)
opp_agent = RandomAgent(distribution='first_buy')

arena = Arena()
results = arena.run_one_duel('deterministic', [mcts_agent, opp_agent])
print(results)