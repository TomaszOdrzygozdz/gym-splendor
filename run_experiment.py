#from experiments.vanilla_mcts_experiments.vanilla_mcts_experiment_v1 import run
from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from agents.random_agent import RandomAgent
from arena.multi_process.multi_arena import DeterministicMultiProcessArena
arek = DeterministicMultiProcessArena()

agent1 = RandomAgent(distribution='first_buy')
agent2 = RandomAgent(distribution='uniform')
agent3 = MultiProcessMCTSAgent(iteration_limit=2, rollout_repetition=1, create_visualizer=False)

print('ping')

arek.run_one_duel([agent2, agent3])


# from experiments.baseline_comparison.baseline_comparison_v1 import run_baseline_comparison_v1
#
# run_baseline_comparison_v1()
