#from experiments.vanilla_mcts_experiments.vanilla_mcts_experiment_v1 import run
from agents.random_agent import RandomAgent
from arena.multi_process.multi_arena import DeterministicMultiProcessArena
arek = DeterministicMultiProcessArena()

agent1 = RandomAgent(distribution='first_buy')
agent2 = RandomAgent(distribution='uniform_on_types')

arek.run_many_duels([agent1, agent2], 3, 3)


# from experiments.baseline_comparison.baseline_comparison_v1 import run_baseline_comparison_v1
#
# run_baseline_comparison_v1()
