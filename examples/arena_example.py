from agents.random_agent import RandomAgent
from arena import Arena

fight_pit = Arena()

goku = RandomAgent(distribution='uniform_on_types')
gohan = RandomAgent(distribution='uniform')

fight_pit.run_one_game([goku, gohan], starting_player_id=0)