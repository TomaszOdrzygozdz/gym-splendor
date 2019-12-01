#from experiments.vanilla_mcts_experiments.vanilla_mcts_experiment_v1 import run
#from experiments.baseline_comparison.baseline_comparison_v3 import run_baseline_comparison_v3

#run_baseline_comparison_v3(10)
import time

from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from agents.general_multi_process_mcts_agent import GeneralMultiProcessMCTSAgent
from agents.random_agent import RandomAgent

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from arena.multi_process.arena_multi_thread import ArenaMultiThread
from arena.multi_process.multi_arena import MultiArena
from arena.single_process.arena import Arena
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State

from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer

agent1 = GreedySearchAgent()
agent2 = RandomAgent(distribution='first_buy')
agent3 = RandomAgent(distribution='uniform_on_types')
agent4 = GeneralMultiProcessMCTSAgent(100, 5, False, False,
                                        mcts = "evaluation")
                                        #param_1 = "random"/ "greedy" :global method
                                        #param_2 = "first_buy", etc / "weight", "depth", "breadth, "decay", ... : parameter or list of paramters


import cProfile



duper = MultiArena()
fuf = duper.run_many_duels('deterministic', [agent2, agent4], n_games=1, n_proc_per_agent=1)
print(fuf)

#bumek.run('duper.run_many_duels(\'deterministic\',[agent2, agent4], n_games=1, n_proc_per_agent=1)')
#bumek.dump_stats('stats.prof')
#duper.run_many_duels('deterministic',[agent2, agent4], n_games=1, n_proc_per_agent=10)

# profek = cProfile.Profile()
# tm = time.time()
# bubu = DeterministicVanillaMCTS(150)
# stanek = State()
# bubu.create_root(DeterministicObservation(stanek))
# bubu.run_simulation(100)
# # profek.run('bubu.run_simulation(10)')
# # profek.dump_stats('stats.prof')
# #
# # print('Time taken = {}'.format(time.time() - tm))
# #
# fufu = TreeVisualizer(show_unvisited_nodes=False)
# fufu.generate_html(bubu.root, 'TUKAN.html')
