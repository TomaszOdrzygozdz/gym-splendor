from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost

from agents.minmax_agent import MinMaxAgent
from agents.greedysearch_agent import GreedySearchAgent
from arena.arena import Arena

fight_pit = Arena()


# time_profile = cProfile.Profile()
# time_profile.run('fight_pit.run_many_duels([goku, gohan], number_of_games=100)')
# time_profile.dump_stats('optimization1.prof')


n_games = 10

gohan = GreedyAgentBoost(weight = [100,2,2,1,0.1])
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels("deterministic",[goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


gohan = GreedyAgentBoost(weight = [100,2.5,1.5,1,0.1])
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))



gohan = MinMaxAgent(name = "MinMax", depth = 2)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


gohan = MinMaxAgent(name = "MinMax", depth = 3)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))



gohan = MinMaxAgent(name = "MinMax", weight = [100,2.5,1.5,1,0.1], depth = 3)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


gohan = MinMaxAgent(name = "MinMax", decay =0.7, depth = 3)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))



gohan = MinMaxAgent(name = "MinMax", decay = 1.2, depth = 3)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


gohan = GreedySearchAgent(name = "GreedySearch", depth = 3)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))




gohan = GreedySearchAgent(name = "GreedySearch", weight = [100,2.5,1.5,1,0.1], depth = 3)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


gohan = GreedySearchAgent(name = "GreedySearch", depth = 5)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


gohan = GreedySearchAgent(name = "GreedySearch", depth = 4, breadth = 2)
print(gohan.name)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
