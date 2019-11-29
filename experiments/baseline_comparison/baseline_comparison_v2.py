from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent2 import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
from mpi4py import MPI
import matplotlib.pyplot as plt
from arena.multi_process.arena_multi_thread import ArenaMultiThread


comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank == 0


def run_baseline_comparison_v2(n_games = 10):

    experiment_name = 'baseline_comparison_v2'

    agent1 = GreedyAgentBoost(weight = [100, 1.5, 2.5, 1, 0.1])
    agent2 = GreedyAgentBoost(weight = [100, 2, 2, 1, 0.1])
    agent3 = MinMaxAgent(weight = [100, 2, 2, 1, 0.1], depth = 2)
    agent4 = MinMaxAgent(weight = [100, 2, 2, 1, 0.1])
    agent5 = MinMaxAgent(weight = [100, 1.5, 2.5, 1, 0.1])
    agent6 = MinMaxAgent(weight = [100, 2, 2, 1, 0.1], decay = 0.7)
    agent7 = MinMaxAgent(weight = [100, 2, 2, 1, 0.1], decay = 1.2)
    agent8 = GreedySearchAgent(depth = 3, breadth = 1, weight = [100, 2, 2, 1, 0.1])
    agent9 = GreedySearchAgent(depth = 3, weight = [100, 2, 2, 1, 0.1])
    agent10 = GreedySearchAgent(depth = 3, weight = [100, 2.5, 1.5, 1, 0.1])
    agent11 = GreedySearchAgent(depth = 4, weight = [100, 2, 2, 1, 0.1])
    agent12 = GreedySearchAgent(depth = 5, breadth = 1, weight = [100, 2, 2, 1, 0.1])
    agent13 = RandomAgent(distribution = 'uniform')
    agent14 = RandomAgent(distribution = 'uniform_by_types')
    agent15 = RandomAgent(distribution = 'first_buy')


    multi_arena = ArenaMultiThread()

    list_of_agents = [agent1,
                      agent2,
                      agent3,
                      agent4,
                      agent5,
                      agent6,
                      agent7,
                      agent8,
                      agent9,
                      agent10,
                      agent11,
                      agent12,
                      agent13,
                      agent14,
                      agent15]

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

        plt.title('Average win rate over {} games per pair:'.format(2 * n_games))
        wins_pic = results.create_heatmap(param='wins', average=True)
        plt.savefig('reports/wins.png')
        plt.clf()

        plt.title('Average reward over {} games per pair:'.format(2 * n_games))
        reward_pic = results.create_heatmap('reward', average=True)
        plt.savefig('reports/reward.png')
        plt.clf()

        plt.title('Average victory points over {} games per pair:'.format(2*n_games))
        vic_points_pic = results.create_heatmap('victory_points', average=True)
        plt.savefig('reports/victory_points.png')
        plt.clf()

        plt.title('Average games played over {} games per pair:'.format(2*n_games))
        vic_points_pic = results.create_heatmap('games', average=True)
        plt.savefig('reports/games.png')
        plt.clf()
