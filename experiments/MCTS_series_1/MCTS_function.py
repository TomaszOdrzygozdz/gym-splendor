from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from agents.single_mcts_agent import SingleMCTSAgent
from arena.arena import Arena
from monte_carlo_tree_search.evaluation_policies.heura_val import HeuraEvaluator

arek = Arena()
a1 = GreedyAgentBoost()
a2 = SingleMCTSAgent(150, HeuraEvaluator(), 0.4, True, False)

results = arek.run_many_duels('deterministic', [a1, a2], 1, True)
print(results)