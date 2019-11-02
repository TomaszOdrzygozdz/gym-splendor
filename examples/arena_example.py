from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.minmax_agent import MinMaxAgent

import time

from arena.arena import Arena
import cProfile
fight_pit = Arena()

goku = RandomAgent(distribution='first_buy')
goku2 = RandomAgent(distribution='uniform')
#gohan = RandomAgent(distribution='uniform_on_types')
#gohan = RandomAgent(distribution='uniform')
#goku = GreedyAgen/t(weight = 0.3)
gohan = GreedyAgent(weight = 0.1)
goku = MinMaxAgent(name = "MinMax", depth = 3)


# profi = cProfile.Profile()
#
# profi.run('(fight_pit.run_many_games([goku, gohan], number_of_games=50))')
# profi.dump_stats('profi2.prof')
time_dupa = time.time()
fight_pit.run_one_game([goku, gohan], render_game=False)
time.time() - time_dupa
# for i in range(100):
#     print(i)
#     fight_pit = Arena()
#     fight_pit.run_one_game([goku, gohan], starting_agent_id=0)
# print(time.time() - time_dupa)
