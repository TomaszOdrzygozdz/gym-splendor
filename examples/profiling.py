from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.greedy_agent_boost import GreedyAgentBoost

from agents.minmax_agent import MinMaxAgent
from agents.greedysearch_agent2 import GreedySearchAgent
from arena.arena import Arena

fight_pit = Arena()

goku = RandomAgent(distribution='uniform')
goku = RandomAgent(distribution='first_buy')
#gohan = RandomAgent(distribution='uniform_on_types')
#gohan = RandomAgent(distribution='uniform')
#goku = GreedyAgent(weight = 0.3)
gohan = GreedyAgent(weight = 0.1)
goku = GreedySearchAgent(name = "GS", depth = 6)
goku = MinMaxAgent(name = "MinMax", depth = 3)


# time_profile = cProfile.Profile()
# time_profile.run('fight_pit.run_many_duels([goku, gohan], number_of_games=100)')
# time_profile.dump_stats('optimization1.prof')


n_games = 1000
gohan = MinMaxAgent(name = "MinMax", weight = [100,2.5,1.5,1,0.1], depth = 3)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
479/778
12752/778

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
837/838
1431/838
13697/838
goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
894/903
3357/903
14663/903

n_games = 1000
gohan = MinMaxAgent(name = "MinMax", decay =0.7, depth = 3)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
500/807
13305/807
goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
853/854
1500/854
13913/854
goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
925/938
3522/938
15236/938
n_games = 1000
gohan = MinMaxAgent(name = "MinMax", decay = 1.2, depth = 3)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
513/734
12020/734
goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
1410/787
12860/787

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

868/880
3292/880
14283/880

n_games = 1000
gohan = GreedySearchAgent(name = "GreedySearch", depth = 3)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))




n_games = 1000
gohan = GreedySearchAgent(name = "GreedySearch", weight = [100,2.5,1.5,1,0.1], depth = 3)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
615/824
13503/824
goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
1850/891
14654/891

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
911/921
3537/921
15193/921

n_games = 1000
gohan = GreedySearchAgent(name = "GreedySearch", depth = 7)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


n_games = 100
gohan = GreedySearchAgent(name = "GreedySearch", depth = 4, breadth = 2)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))



n_games = 1000
gohan = GreedySearchAgent(name = "GreedySearch", depth = 4, breadth = 2)

goku = RandomAgent(distribution='uniform')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution='uniform_on_types')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

goku = RandomAgent(distribution = 'first_buy')
print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
