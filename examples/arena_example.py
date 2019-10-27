from agents.random_agent import RandomAgent
from arena import Arena
import cProfile, pstats

fight_pit = Arena()

goku = RandomAgent(distribution='uniform_on_types')
gohan = RandomAgent(distribution='uniform')


fight_pit.run_one_game([goku, gohan], starting_player_id=0)

# pr = cProfile.Profile()
# pr.run('fight_pit.run_one_game([goku, gohan], starting_player_id=0)')
# filename = 'profile.prof'  # You can change this if needed
# pr.dump_stats(filename)

fight_pit.run_many_games([goku, gohan], number_of_games=10)