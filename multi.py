# from agents.minmax_agent import MinMaxAgent
# from agents.random_agent import RandomAgent
# from agents.simple_mcts_agent import SimpleMCTSAgent
#
# from mpi4py import MPI
# #from arena.many_vs_many import ManyVsManyStatistics
# from arena.deterministic_arena import DeterministicArena
#
# comm = MPI.COMM_WORLD
# my_rank = MPI.COMM_WORLD.Get_rank()
# main_thread = my_rank==0
#
# # agent1 = GreedyAgentBoost(name = "Greedy Paper(mod)", weight = [100,2,2,1,0.1])
# # agent2 = GreedyAgentBoost(weight = [100,1.5,2.5,1,0.1])
# # agent3 = GreedyAgentBoost(weight = [0.99954913, 0.01997425, 0.02001405, 0.01004779, 0.00101971])
# # agent4 = GreedyAgentBoost(weight = [0.99953495, 0.02010871, 0.02010487, 0.01095619, 0.00113329])
# # agent5 = GreedyAgentBoost(name = "Greedy Paper", weight = [100,2,2,1,1])
# # agent6 = MinMaxAgent(weight = [100,2,2,1,0.1])
# # agent7 = MinMaxAgent(name = "MinMax Paper(mod)",weight = [100,1.5,2.5,1,0.1])
# #agent8 = MinMaxAgent(weight = [100,2,2,1,0.1], decay = 0.7)
# # agent9 = MinMaxAgent(name = "MinMax Paper", weight = [100,1.5,2.5,1,1])
# # agent10 = MinMaxAgent(weight = [100,2,2,1,0.1], decay = 1)
# # agent11 = GreedySearchAgent(depth = 9, weight = [100,2,2,1,0.1], decay = 0.95)
# # agent12 = GreedySearchAgent(depth = 7, weight = [100,2,2,1,0.1])
# # agent13 = GreedySearchAgent(depth = 7, weight = [100,1.5,2.5,1,1], decay = 0.7)
# # agent14 = GreedySearchAgent(depth = 7, weight = [100,1.5,2.5,1,1])
# # agent15 = GreedySearchAgent(depth = 7, weight = [0.99953495, 0.02010871, 0.02010487, 0.01095619, 0.00113329], decay = 0.95)
# #agent1 = RandomAgent(distribution='uniform')
# #agent2 = RandomAgent(distribution='uniform_on_types')
# agent1 = GreedyAgentBoost(weight = [100,2.5,1.5,2,0.02])
# agent2 = RandomAgent(distribution='first_buy')
#
# #agent_mcts = VanillaMCTSAgent(steps = 50)
#
#
# agent_mcts = SimpleMCTSAgent(2)
# #
# # arena = Arena()
# #
#  multi_arena = ArenaMultiThread()
#
# n_games = 4
# list_of_agents = [agent_mcts, agent1]
# #list_of_agents = [agent8, agent1]
#
#
# arek = DeterministicArena()
# resu = arek.run_one_duel(list_of_agents, 0, render_game=False)
#
# #if main_thread:
# #    print(resu)
# # results = multi_arena.all_vs_all(list_of_agents, n_games)
#
# from matplotlib  import pyplot as plt
# import pickle
# n_games = 250
# with open('reports/results.pickle', 'rb') as f:
#     results = pickle.load(f)
#
# results
#  if main_thread:
#      print(' \n \n {}'.format(results.to_pandas()))
#      print('\n \n \n')
#      print(results)
#      wins = results.to_pandas(param='wins').to_csv('wins.csv')
#      vic_points = results.to_pandas(param='victory_points').to_csv('victory_points.csv')
#      rewards = results.to_pandas(param='reward').to_csv('reward.csv')
#
#      plt.title('Average win rate over {} games per pair:'.format(2*n_games))
#      wins_pic = results.create_heatmap(param='wins', average=True, p1 = 15, p2 =1)
#      plt.savefig('reports/wins.png')
#      plt.clf()
#
#      plt.title('Average reward over {} games per pair:'.format(2*n_games))
#      reward_pic = results.create_heatmap('reward', average=True, p1 = 15, p2 =1)
#      plt.savefig('reports/reward.png')
#      plt.clf()
#
#      plt.title('Average victory points over {} games per pair:'.format(2*n_games))
#      vic_points_pic = results.create_heatmap('victory_points', average=True, p1 = 15, p2 =1)
#      plt.savefig('reports/victory_points.png')
#      plt.clf()
#
#      plt.title('Average win rate over {} games per pair:'.format(2*n_games))
#      wins_pic = results.create_heatmap(param='games', average = True, p1 = 15, p2 =1, n_games = n_games)
#      plt.savefig('reports/games.png')
#      plt.clf()
#
#
# multi_arena.all_vs_all([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7], 2)
#
#  dupa = ManyVsManyStatistics([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7])
#  print(dupa)
#
#  fu = Arena()
#  x = fu.run_one_duel([agent1, agent2])
#  print(x)
