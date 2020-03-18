from agents.greedy_agent_boost import GreedyAgentBoost
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena

arena = MultiArena()
a1 = RandomAgent(distribution='first_buy')
a2 = GreedyAgentBoost()
a3 = MinMaxAgent()

results = arena.run_many_duels('deterministic', [a2, a3], 10, 1)
print(results)