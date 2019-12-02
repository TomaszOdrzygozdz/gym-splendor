#from experiments.vanilla_mcts_experiments.vanilla_mcts_experiment_v1 import run
#from experiments.baseline_comparison.baseline_comparison_v3 import run_baseline_comparison_v3

#run_baseline_comparison_v3(10)
import time

from agents.general_multi_process_mcts_agent import GeneralMultiProcessMCTSAgent
from agents.random_agent import RandomAgent

from arena.multi_arena import MultiArena

#agent1 = GreedySearchAgent()
agent1 = RandomAgent(distribution='uniform')
#agent3 = RandomAgent(distribution='uniform_on_types')
agent1a = GeneralMultiProcessMCTSAgent(10, 2, False, False,
                                        mcts = "rollout",
                                        param_1 = "random",
                                        param_2 = "uniform")

agent1b = GeneralMultiProcessMCTSAgent(100, 5, False, False,
                                        mcts = "rollout",
                                        param_1 = "random",
                                        param_2 = "first_buy")

agent1c = GeneralMultiProcessMCTSAgent(100, 5, False, False,
                                        mcts = "rollout",
                                        param_1 = "greedy")

agent1d = GeneralMultiProcessMCTSAgent(100, 5, False, False,
                                        mcts = "evaluation",
                                        param_2 = [[100,2,2,1,0.1], 0.9, 3, 2])

agent1e = GeneralMultiProcessMCTSAgent(100, 5, False, False,
                                        mcts = "evaluation",
                                        param_2 = [[100,2,2,1,0.1], 0.9, 4, 1])
                                        #param_1 = "random"/ "greedy" :global method
                                        #param_2 = "first_buy", etc / "weight", "depth", "breadth, "decay", ... : parameter or list of paramters

arena = MultiArena()

t0 = time.time()
result = arena.run_many_duels('deterministic', [agent1, agent1a], n_games = 20, n_proc_per_agent=24)
print(result)
print("Time",time.time() - t0)

t0 = time.time()
result = arena.run_many_duels('deterministic', [agent1, agent1b], n_games=20, n_proc_per_agent=24)
print(result)
print("Time",time.time() - t0)

t0 = time.time()
result = arena.run_many_duels('deterministic', [agent1, agent1c], n_games=20, n_proc_per_agent=24)
print(result)
print("Time",time.time() - t0)

t0 = time.time()
result = arena.run_many_duels('deterministic', [agent1, agent1d], n_games=20, n_proc_per_agent=24)
print(result)
print("Time",time.time() - t0)

t0 = time.time()
result = arena.run_many_duels('deterministic', [agent1, agent1e], n_games=20, n_proc_per_agent=24)
print(result)
print("Time",time.time() - t0)

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
