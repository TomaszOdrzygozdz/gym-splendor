import time

from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from agents.general_multi_process_mcts_agent import GeneralMultiProcessMCTSAgent
from agents.random_agent import RandomAgent

from arena.multi_arena import MultiArena

agent1 = RandomAgent(distribution='uniform')
agent1a = GeneralMultiProcessMCTSAgent(10, 2, False, False,
                                        mcts = "rollout",
                                        param_1 = "random",
                                        param_2 = "uniform")

agent1b = GeneralMultiProcessMCTSAgent(100, 5, False, False,
                                        mcts = "rollout",
                                        param_1 = "random",
                                        param_2 = "first_buy")

agent1c = GeneralMultiProcessMCTSAgent(100, 1, False, False,
                                        mcts = "rollout",
                                        param_1 = "greedy")

agent1d = GeneralMultiProcessMCTSAgent(100, 1, False, False,
                                        mcts = "evaluation",
                                        param_2 = [[100,2,2,1,0.1], 0.9, 3, 2])

agent1e = GeneralMultiProcessMCTSAgent(100, 1, False, False,
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