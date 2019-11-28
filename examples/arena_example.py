from agents.random_agent import RandomAgent
from agents.minmax_agent import MinMaxAgent
from agents.greedysearch_agent2 import GreedySearchAgent

from arena.single_process.arena import Arena

environment_id = 'gym_splendor_code:splendor-v1'
fight_pit = Arena(environment_id)

goku = RandomAgent(distribution='first_buy')
goku2 = RandomAgent(distribution='uniform')
#gohan = RandomAgent(distribution='uniform_on_types')
#gohan = RandomAgent(distribution='uniform')
#goku = GreedyAgen/t(weight = 0.3)
gohan = GreedySearchAgent(depth = 5)
goku = MinMaxAgent(name = "MinMax", depth = 3)
gohan.name = "g2"
goku.name = "g1"
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
