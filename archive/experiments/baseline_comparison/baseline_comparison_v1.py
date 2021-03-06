from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent2 import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from mpi4py import MPI

from arena.arena_multi_thread import ArenaMultiThread


comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank == 0
import matplotlib.pyplot as plt


def run_baseline_comparison_v1():

    experiment_name = 'baseline_comparison_v1'

    agent1 = GreedyAgentBoost(weight = [100,1.5,2.5,1,0.1])
    agent2 = GreedyAgentBoost(weight = [0.99953495, 0.02010871, 0.02010487, 0.01095619, 0.00113329])
    agent3 = MinMaxAgent(weight = [100,2,2,1,0.1])
    # agent4 = MinMaxAgent(weight=[100, 2, 2, 1, 0.1], decay=0.7)
    # agent5 = GreedySearchAgent(depth = 9, weight = [100,2,2,1,0.1], decay = 0.95)
    # agent6 = GreedySearchAgent(depth = 7, weight = [100,2,2,1,0.1])
    agent7 = GreedySearchAgent(depth = 7, weight = [0.99953495, 0.02010871, 0.02010487, 0.01095619, 0.00113329], decay = 0.95)
    # agent8 = RandomAgent(distribution='first_buy')

    multi_arena = ArenaMultiThread()

    n_games = 1
    list_of_agents = [agent1, agent2, agent3, agent7]
    results = multi_arena.all_vs_all(list_of_agents, n_games)

    if main_thread:
        print(' \n \n {}'.format(results.to_pandas()))
        print('\n \n \n')
        print(results)
        wins = results.to_pandas(param='wins').to_csv('wins.csv')
        vic_points = results.to_pandas(param='victory_points').to_csv('victory_points.csv')
        rewards = results.to_pandas(param='reward').to_csv('reward.csv')

        #leader_board = LeaderBoard(list_of_agents)
        #leader_board.load_from_file()
        #leader_board.register_from_games_statistics(results)
        #print(leader_board)
        #leader_board.save_to_file()

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
