from agents.random_agent import RandomAgent
from agents.value_function_agent import ValueFunctionAgent
from arena.arena import Arena

a1 = RandomAgent(distribution='uniform')
a2 = ValueFunctionAgent()

arek = Arena()
results = arek.run_many_duels('deterministic', [a1, a2], 20)
print(results)