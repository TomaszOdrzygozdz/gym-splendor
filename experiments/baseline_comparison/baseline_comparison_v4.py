from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost

from arena.arena import Arena

fight_pit = Arena()

def run_comparison(n_games = 1000):

    gohan = GreedyAgentBoost()

    goku = RandomAgent(distribution='uniform')
    print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

    goku = RandomAgent(distribution='uniform_on_types')
    print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

    goku = RandomAgent(distribution = 'first_buy')
    print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))


    gohan = GreedyAgentBoost(weight = [100,2.5,1.5,1,0.1])

    goku = RandomAgent(distribution='uniform')
    print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

    goku = RandomAgent(distribution='uniform_on_types')
    print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))

    goku = RandomAgent(distribution = 'first_buy')
    print(fight_pit.run_many_duels([goku, gohan], number_of_games = n_games, shuffle_agents=True))
