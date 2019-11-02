from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from arena.arena import Arena
import cProfile
fight_pit = Arena()

goku = RandomAgent(distribution='first_buy')
goku2 = RandomAgent(distribution='uniform')
#gohan = RandomAgent(distribution='uniform_on_types')
#gohan = RandomAgent(distribution='uniform')
#goku = GreedyAgent(weight = 0.3)
gohan = GreedyAgent(weight = 0.1)


# profi = cProfile.Profile()
#
# profi.run('(fight_pit.run_many_duels([goku, gohan], number_of_games=50))')
# profi.dump_stats('profi2.prof')

fight_pit.run_one_duel([goku, gohan], render_game=True)

# time_dupa = time.time()
# for i in range(100):
#     print(i)
#     fight_pit = Arena()
#     fight_pit.run_one_duel([goku, gohan], starting_agent_id=0)
# print(time.time() - time_dupa)
