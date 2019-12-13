"""Arena holds games between players, measures their performance and calculates ELO rating. This version

___Single thread version___
hold only 1 vs 1 games."""

from typing import List
from gym_splendor_code.envs.mechanics.game_settings import USE_TQDM
if USE_TQDM:
    from tqdm import tqdm
import random

import gym

from agents.abstract_agent import Agent
from arena.game_statistics_duels import GameStatisticsDuels
from arena.one_agent_statistics import OneAgentStatistics
from gym_splendor_code.envs.graphics.graphics_settings import GAME_INITIAL_DELAY
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES

import time

class Arena:

    def __init__(self,
                 environment_id: str = 'gym_splendor_code:splendor-v0') -> None:
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)

    def run_self_play(self, mode: str,
                      agent: Agent,
                      render_game=False,
                      mpi_communicator = None):

        self.env.reset()
        self.env.set_active_player(0)
        # set players names:
        is_done = False
        # set the initial agent id
        # set the initial observation
        observation = self.env.show_observation(mode)
        number_of_actions = 0
        results_dict = {}
        # Id if the player who first reaches number of points to win
        previous_actions = [None]
        action = None

        agent.set_self_play_mode()

        if mpi_communicator is None:
            local_main_process = True
        if mpi_communicator is not None:
            local_main_process = mpi_communicator.Get_rank() == 0
            if agent.multi_process:
                agent.set_communicator(mpi_communicator)

        if render_game and local_main_process:
            self.env.render()
            time.sleep(GAME_INITIAL_DELAY)

        while number_of_actions < MAX_NUMBER_OF_MOVES and not is_done:
            if agent.multi_process == True:
                action = agent.choose_action(observation, previous_actions)
            if agent.multi_process == False and local_main_process:
                action = agent.choose_action(observation, previous_actions)
            previous_actions = [action]

            if local_main_process:
                observation, reward, is_done, info = self.env.step(mode, action)
                winner_id = info['winner_id']
            if render_game:
                self.env.render()

            number_of_actions += 1
            if mpi_communicator is not None:
                is_done = mpi_communicator.bcast(is_done, root=0)

        if local_main_process:
            self.env.reset()

        agent.unset_self_play_mode()
        print('Self-game done')



    def run_one_duel(self,
                     mode: str,
                     list_of_agents: List[Agent],
                     starting_agent_id: int = 0,
                     render_game: bool=False,
                     mpi_communicator = None)-> GameStatisticsDuels:
        """Runs one game between two agents.
        :param:
        mode: mode of the game (stochastic or deterministic)
        list_of_agents: List of agents to play, they will play in the order given by the list
        starting_agent_id:  Id of the agent who starts the game.
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
        observation = self.env.show_observation(mode)
        number_of_actions = 0
        results_dict = {}
        #Id if the player who first reaches number of points to win
        first_winner_id = None
        checked_all_players_after_first_winner = False
        previous_actions = [None]
        action = None

        if mpi_communicator is None:
            local_main_process = True
        if mpi_communicator is not None:
            local_main_process = mpi_communicator.Get_rank() == 0
            for agent in list_of_agents:
                if agent.multi_process:
                    agent.set_communicator(mpi_communicator)

        if render_game and local_main_process:
            self.env.render()
            time.sleep(GAME_INITIAL_DELAY)

        while  number_of_actions < MAX_NUMBER_OF_MOVES and not is_done:
            #if local_main_process:
                #print('Action number = {}'.format(number_of_actions))
            active_agent = list_of_agents[active_agent_id]
            if active_agent.multi_process == True:
                action = list_of_agents[active_agent_id].choose_action(observation, previous_actions)
            if active_agent.multi_process == False and local_main_process:
                action = list_of_agents[active_agent_id].choose_action(observation, previous_actions)
            previous_actions = [action]

            if local_main_process:
                observation, reward, is_done, info = self.env.step(mode, action)
                winner_id = info['winner_id']
            if render_game:
                self.env.render()

            if is_done and local_main_process:
                results_dict[list_of_agents[active_agent_id].my_name_with_id()] = \
                    OneAgentStatistics(reward, self.env.points_of_player_by_id(active_agent_id),
                                       int(winner_id == active_agent_id))
                previous_agent_id = (active_agent_id-1)%len(list_of_agents)
                results_dict[list_of_agents[previous_agent_id].my_name_with_id()] = \
                    OneAgentStatistics(reward*(-1), self.env.points_of_player_by_id(previous_agent_id),
                                   int(previous_agent_id == winner_id))
                if first_winner_id is None:
                    first_winner_id = active_agent_id
                checked_all_players_after_first_winner = active_agent_id == (first_winner_id-1)%len(list_of_agents)
                #print('ALL MOVED = {}'.format(checked_all_players_after_first_winner))

            active_agent_id = (active_agent_id + 1) % len(list_of_agents)
            number_of_actions += 1
            if mpi_communicator is not None:
                is_done = mpi_communicator.bcast(is_done, root = 0)

        if local_main_process:
            self.env.reset()
            one_game_statistics = GameStatisticsDuels(list_of_agents)
            one_game_statistics.register_from_dict(results_dict)

        for agent in list_of_agents:
            agent.finish_game()

        return one_game_statistics if local_main_process else None


    def run_many_duels(self,
                       mode,
                       list_of_agents: List[Agent],
                       number_of_games: int,
                       shuffle_agents: bool = True,
                       starting_agent_id = 0):

        """Runs many games on a single process.
        :param
        list_of_agents: List of agents to play, they will play in the order given by the list.
        number_of_games: The number of games to play.
        shuffle_agents: If True list of agents (and thus their order in the game will be shuffled after each game).
        starting_agent_id: Id of the agent who starts each game.
        """
        assert number_of_games > 0, 'Number of games must be positive'
        cumulative_results = GameStatisticsDuels(list_of_agents)
        games_ids_to_iterate = tqdm(range(number_of_games)) if USE_TQDM else range(number_of_games)
        for game_id in games_ids_to_iterate:
            if shuffle_agents:
                starting_agent_id = random.choice(range(len(list_of_agents)))
            one_game_results = self.run_one_duel(mode, list_of_agents, starting_agent_id)
            #update results:
            cumulative_results.register(one_game_results)

        cumulative_results.number_of_games = number_of_games
        return cumulative_results
