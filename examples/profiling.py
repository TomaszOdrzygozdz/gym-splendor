import cProfile

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from arena import Arena
from tqdm import tqdm
import time

fight_pit = Arena()

goku = RandomAgent(distribution='first_buy')
goku2 = RandomAgent(distribution='uniform')
#gohan = RandomAgent(distribution='uniform_on_types')
#gohan = RandomAgent(distribution='uniform')
#goku = GreedyAgent(weight = 0.3)
gohan = GreedyAgent(weight = 0.1)


time_profile = cProfile.Profile()
time_profile.run('fight_pit.run_many_games_single_process([goku, gohan], number_of_games=100)')
time_profile.dump_stats('optimization1.prof')

# for i in tqdm(range(100)):
#     fight_pit.run_one_game([goku, gohan], starting_agent_id=0, render_game=False)
# fight_pit.run_one_game([goku, gohan], starting_agent_id=0, render_game=False)
# fight_pit.run_one_game([goku, gohan], starting_agent_id=0, render_game=True)


#print(fight_pit.run_many_games_single_process([goku, gohan], number_of_games=100))

# time_dupa = time.time()
# for i in range(100):
#     print(i)
#     fight_pit = Arena()
#     fight_pit.run_one_game([goku, gohan], starting_agent_id=0)
# print(time.time() - time_dupa)
