from agents.random_agent import RandomAgent
from arena import Arena
import time

fight_pit = Arena()

goku = RandomAgent(distribution='uniform_on_types')
gohan = RandomAgent(distribution='uniform')

time_dupa = time.time()
fight_pit.run_one_game([goku, gohan], starting_player_id=0)
print(time.time() - time_dupa)