from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.minmax_agent import MinMaxAgent
from arena.arena import Arena

fight_pit = Arena()

goku = RandomAgent(distribution='uniform')
goku2 = RandomAgent(distribution='uniform')
#gohan = RandomAgent(distribution='uniform_on_types')
#gohan = RandomAgent(distribution='uniform')
#goku = GreedyAgent(weight = 0.3)
gohan = GreedyAgent(weight = 0.1)

goku = MinMaxAgent(name = "MinMax", depth = 3)


# time_profile = cProfile.Profile()
# time_profile.run('fight_pit.run_many_duels([goku, gohan], number_of_games=100)')
# time_profile.dump_stats('optimization1.prof')

# for i in tqdm(range(100)):
#     fight_pit.run_one_duel([goku, gohan], starting_agent_id=0, render_game=False)
# fight_pit.run_one_duel([goku, gohan], starting_agent_id=0, render_game=False)
# fight_pit.run_one_duel([goku, gohan], starting_agent_id=0, render_game=True)


print(fight_pit.run_many_games([goku, gohan], number_of_games = 100, shuffle_agents=False))

# time_dupa = time.time()
# for i in range(100):
#     print(i)
#     fight_pit = Arena()
#     fight_pit.run_one_duel([goku, gohan], starting_agent_id=0)
# print(time.time() - time_dupa)
