"""Arena holds games between players, measures their performance and calculates ELO rating. This version
___Single process version___
hold only 1 vs 1 games."""


#This package provides ELO rating for players
#You can get this from: https://github.com/HankSheehan/EloPy
import elopy

from typing import List, Dict

import gym

from agent import Agent
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES


class Arena:

    def __init__(self, environment_id: str = 'gym_splendor_code:splendor-v0') -> None:
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)


    def run_one_game(self,
                     list_of_agents: List[Agent], starting_player_id,
                     max_number_of_moves: int = MAX_NUMBER_OF_MOVES)->Dict:

        #prepare the game
        self.env.reset()
        self.env.set_active_player(starting_player_id)
        #set players names:
        self.env.set_players_names([agent.name for agent in list_of_agents])
        is_done = False
        #set the initial values:
        active_agent_id = starting_player_id
        observation = self.env.show_observation()
        number_of_actions_taken = 0
        full_round_finished = False
        results_dict = {agent : 0 for agent in list_of_agents}
        #Here happens the game:
        while not is_done and not full_round_finished and number_of_actions_taken <= max_number_of_moves:
            action = list_of_agents[active_agent_id].choose_action(observation)
            observation, reward, is_done, info = self.env.step(action)
            active_agent_id = (active_agent_id + 1)%len(list_of_agents)
            number_of_actions_taken += 1
            full_round_finished = active_agent_id == (starting_player_id - 1)%len(list_of_agents)
            if is_done:
                results_dict[list_of_agents[active_agent_id]] = reward
                print(reward)

        return results_dict


    def run_many_games(self,
                       list_of_agents: List[Agent],
                       number_of_games: int,
                       max_number_of_moves: int = MAX_NUMBER_OF_MOVES,
                       ):

        cum_results = {agent : 0 for agent in list_of_agents}

        for i in range(number_of_games):
            print('Game number {}:'.format(i))
            results_dict = self.run_one_game(list_of_agents,i%2)
            for agent in results_dict.keys():
                cum_results[agent] += results_dict[agent]

        for agent in cum_results.keys():
            print(agent.name + ' {}'.format(cum_results[agent]))


