from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from arena.arena import Arena
from arena.arena_multi_thread import ArenaMultiThread

from mpi4py import MPI
import time

from arena.game_statistics_duels import GameStatisticsDuels
#from arena.many_vs_many import ManyVsManyStatistics
from arena.leaderboard import LeaderBoard

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank==0

agent0 = RandomAgent()
agent1 = GreedyAgent(weight=0.1)
agent2 = GreedyAgent(weight=0.2)
agent3 = GreedyAgent(weight=0.3)
agent4 = GreedyAgent(weight=0.4)
agent5 = GreedyAgent(weight=0.5)
agent6 = GreedyAgent(weight=0.6)
agent7 = GreedyAgent(weight=0.7)


multi_arena = ArenaMultiThread()

results = multi_arena.all_vs_all([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7], n_games=5)



if main_thread:
    print(' \n \n {}'.format(results.to_pandas()))

    print('\n \n \n')
    print(results)
    results.to_pandas().to_csv('wyniczki.csv')

    # leader_board = LeaderBoard([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7])
    # leader_board.load_from_file()
    # leader_board.register_from_games_statistics(results)
    # print(leader_board)
    # leader_board.save_to_file()



#multi_arena.all_vs_all([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7], 2)

# dupa = ManyVsManyStatistics([agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7])
# print(dupa)

# fu = Arena()
# x = fu.run_one_duel([agent1, agent2])
# print(x)