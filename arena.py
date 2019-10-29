"""Arena holds games between players, measures their performance and calculates ELO rating. This version

___Single process version___
hold only 1 vs 1 games."""


#This package provides ELO rating for players
#You can get this from: https://github.com/HankSheehan/EloPy
#import elopy

from typing import List, Dict

import gym

from agent import Agent
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES

import time

class Arena:

    def __init__(self, environment_id: str = 'gym_splendor_code:splendor-v0') -> None:
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)
        self.env.setup_state()

    def run_one_game(self,
                     list_of_agents: List[Agent],
                     starting_agent_id: int,
                     render_game: bool=False,
                     render_game_spped: float = 0.1)->Dict:
        """Runs one game between two agents.
        :param:
        list_of_agents: list of agents to play, they will play in the order given by the list
        starting_agent_id: id of the agent who starts the game.
        show_game: If True, GUI will appear showing the game. """

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
        one_game_results = {agent : {'reward': 0, 'points':0} for agent in list_of_agents}
        #Id if the player who first reaches number of points to win
        first_winner_id = None
        checked_all_players_after_first_winner = False

        if render_game:
            self.env.render()
        while  number_of_actions < MAX_NUMBER_OF_MOVES and not (is_done and checked_all_players_after_first_winner):

            action = list_of_agents[active_agent_id].choose_action(observation)
            observation, reward, is_done, info = self.env.step(action)
            if render_game:
                self.env.render()
            if is_done:
                one_game_results[list_of_agents[active_agent_id]]['reward'] = reward
                one_game_results[list_of_agents[active_agent_id]]['points'] = \
                    self.env.points_of_player_by_id(active_agent_id)
                if first_winner_id is None:
                    first_winner_id = active_agent_id
                checked_all_players_after_first_winner = active_agent_id == (first_winner_id-1)%len(list_of_agents)
            active_agent_id = (active_agent_id + 1) % len(list_of_agents)
            number_of_actions += 1

        print(one_game_results)
        return one_game_results