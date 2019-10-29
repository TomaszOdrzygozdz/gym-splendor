"""Arena holds games between players, measures their performance and calculates ELO rating. This version

___Single process version___
hold only 1 vs 1 games."""


#This package provides ELO rating for players
#You can get this from: https://github.com/HankSheehan/EloPy
#import elopy

from typing import List

import gym

from agent import Agent
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES


class Arena:

    def __init__(self, environment_id: str = 'gym_splendor_code:splendor-v0') -> None:
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)
        self.env.setup_state()

    def run_one_game(self,
                     list_of_agents: List[Agent],
                     starting_agent_id):

        #prepare the game
        self.env.reset()
        self.env.set_active_player(starting_agent_id)
        #set players names:
        self.env.set_players_names([agent.name for agent in list_of_agents])
        is_done = False
        #set the initial agent id
        active_agent_id = starting_agent_id
        #set the initial observation
        observation = self.env.show_observation()
        number_of_actions = 0

        results = {agent : None for agent in list_of_agents}
        round_ended = False
        while not is_done and number_of_actions < MAX_NUMBER_OF_MOVES:
            action = list_of_agents[active_agent_id].choose_action(observation)
            observation, reward, is_done, info = self.env.step(action)
            active_agent_id = (active_agent_id + 1)%len(list_of_agents)
            number_of_actions += 1
            if active_agent_id == starting_agent_id
            if is_done:
                print('The winner is: {}'.format(list_of_agents[active_agent_id].name))
        if  number_of_actions >= MAX_NUMBER_OF_MOVES:
            print('Game unfinished')