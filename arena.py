"""Arena holds games between players, measures their performance and calculates ELO rating. This version

___Single process version___
hold only 1 vs 1 games."""


#This package provides ELO rating for players
#You can get this from: https://github.com/HankSheehan/EloPy
import elopy

from typing import List

import gym

from agent import Agent
from agents.random_agent import RandomAgent


class Arena:

    def __init__(self, environment_id: str = 'gym_splendor_code:splendor-v0') -> None:
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)

    def run_one_game(self,
                     list_of_agents: List[Agent], starting_player_id):

        #prepare the game
        self.env.reset()
        self.env.set_active_player(starting_player_id)
        #set players names:
        self.env.set_players_names([agent.name for agent in list_of_agents])
        is_done = False
        #set the initial agent id
        active_agent_id = starting_player_id
        #set the initial observation
        observation = self.env.show_observation()
        while not is_done:
            action = list_of_agents[active_agent_id].choose_action(observation)
            observation, reward, is_done, info = self.env.step(action)
            active_agent_id = (active_agent_id + 1)%len(list_of_agents)
            if is_done:
                print('The winner is: {}'.format(list_of_agents[active_agent_id].name))
