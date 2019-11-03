from agents.greedy_agent import GreedyAgent
from agents.greedysearch_agent import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
from arena.arena import Arena
from arena.arena_multi_thread import ArenaMultiThread

from mpi4py import MPI
from matplotlib import pyplot as plt
import time

#from arena.many_vs_many import ManyVsManyStatistics
from arena.leaderboard import LeaderBoard

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank==0

agent0 = RandomAgent(distribution='first_buy')
agent1 = GreedyAgent()
agent2 = GreedySearchAgent()
agent3 = MinMaxAgent()

arena = Arena()

multi_arena = ArenaMultiThread()

n_games = 10
list_of_agents = [agent0, agent1, agent2]

results = multi_arena.all_vs_all(list_of_agents, n_games)



if main_thread:
    print(' \n \n {}'.format(results.to_pandas()))

    print('\n \n \n')
    print(results)
    wins = results.to_pandas(param='wins').to_csv('wins.csv')
    vic_points = results.to_pandas(param='victory_points').to_csv('victory_points.csv')
    rewards = results.to_pandas(param='reward').to_csv('reward.csv')

    leader_board = LeaderBoard(list_of_agents)
    #leader_board.load_from_file()
    leader_board.register_from_games_statistics(results)
    print(leader_board)
    leader_board.save_to_file()

    plt.title('Average win rate over {} games per pair:'.format(2*n_games))
    wins_pic = results.create_heatmap(param='wins', average=True)
    plt.savefig('reports/wins.png')
    plt.clf()

    plt.title('Average reward over {} games per pair:'.format(2*n_games))
    reward_pic = results.create_heatmap('reward', average=True)
    plt.savefig('reports/reward.png')
    plt.clf()

    plt.title('Average victory points over {} games per pair:'.format(2*n_games))
    vic_points_pic = results.create_heatmap('victory_points', average=True)
    plt.savefig('reports/victory_points.png')
    plt.clf()

#multi_arena.all_vs_all([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7], 2)

# dupa = ManyVsManyStatistics([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7])
# print(dupa)

# fu = Arena()
# x = fu.run_one_duel([agent1, agent2])
# print(x)