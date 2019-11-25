"""Arena holds games between players, measures their performance and calculates ELO rating. This version

___Single thread version___
hold only 1 vs 1 games."""

from typing import List, Dict
from tqdm import tqdm
import random

import gym

from agent import Agent
from agents.random_agent import RandomAgent
from arena.game_statistics_duels import GameStatisticsDuels
from arena.leaderboard import LeaderBoard
from arena.one_agent_statistics import OneAgentStatistics
from gym_splendor_code.envs.graphics.graphics_settings import GAME_INITIAL_DELAY
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES

import time

from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict


class MultiProcessSingleDuelArena:

    def __init__(self) -> None:

        self.env_initialized = False
        self.name = 'Multi Process Arena'


    def initialize_env(self, environment_id: str = 'gym_splendor_code:splendor-deterministic-v0'):
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)

    def run_one_duel(self,
                     list_of_agents: List[Agent],
                     starting_agent_id: int = 0,
                     render_game: bool = False) -> GameStatisticsDuels:
        """Runs one game between two agents.
        :param:
        list_of_agents: List of agents to play, they will play in the order given by the list
        starting_agent_id:  Id of the agent who starts the game.
        show_game: If True, GUI will appear showing the game. """

        # prepare the game
        self.env.reset()
        self.env.set_active_player(starting_agent_id)
        # set players names:
        self.env.set_players_names([agent.name for agent in list_of_agents])
        is_done = False
        # set the initial agent id
        active_agent_id = starting_agent_id
        # set the initial observation
        observation = self.env.show_observation()
        number_of_actions = 0
        results_dict = {}
        # Id if the player who first reaches number of points to win
        first_winner_id = None
        checked_all_players_after_first_winner = False
        previous_actions = [None]

        if render_game:
            self.env.render()
            time.sleep(GAME_INITIAL_DELAY)

        while number_of_actions < MAX_NUMBER_OF_MOVES and not (is_done and checked_all_players_after_first_winner):
            action = list_of_agents[active_agent_id].choose_action(observation, previous_actions)
            previous_actions = [action]
            observation, reward, is_done, info = self.env.step(action)
            if render_game:
                self.env.render()
            if is_done:
                results_dict[list_of_agents[active_agent_id].my_name_with_id()] = \
                    OneAgentStatistics(reward, self.env.points_of_player_by_id(active_agent_id), int(reward == 1))
                if first_winner_id is None:
                    first_winner_id = active_agent_id
                checked_all_players_after_first_winner = active_agent_id == (first_winner_id - 1) % len(list_of_agents)
            active_agent_id = (active_agent_id + 1) % len(list_of_agents)
            number_of_actions += 1

        one_game_statistics = GameStatisticsDuels(list_of_agents)
        one_game_statistics.register_from_dict(results_dict)

        print(time.time())
        return one_game_statistics

    def run_one_duel_multi_process_deterministic(self,
                                                 mpi_communicator,
                                                 list_of_agents: List[Agent],
                                                 starting_agent_id: int = 0,
                                                 render_game: bool = False):


        # determine which agents need multi processing:
        agents_needing_multi_processing = [agent for agent in list_of_agents if agent.multi_process == True]

        print('Agents needing multi process: {}'.format(agents_needing_multi_processing))

        for agent_needing_multi_processing in agents_needing_multi_processing:
            agent_needing_multi_processing.set_communicator(mpi_communicator)

        my_rank = mpi_communicator.Get_rank()
        main_process = my_rank == 0

        if not self.env_initialized:
            self.initialize_env()

            # prepare the game
            self.env.reset()
            self.env.set_active_player(starting_agent_id)
            # set players names:
            self.env.set_players_names([agent.name for agent in list_of_agents])

            # set the initial observation
            full_state = self.env.current_state_of_the_game
            results_dict = {}
            # Id if the player who first reaches number of points to win
            first_winner_id = None


            if render_game:
                self.env.render()
                time.sleep(GAME_INITIAL_DELAY)

            previous_actions = [None]
            is_done = False
            checked_all_players_after_first_winner = False
            number_of_actions = 0
            active_agent_id = starting_agent_id

            while number_of_actions < MAX_NUMBER_OF_MOVES and not (
                    is_done and checked_all_players_after_first_winner):

                if main_process:
                    print('Number of actions : {}'.format(number_of_actions))

                if list_of_agents[active_agent_id].multi_process == True:
                    action = list_of_agents[active_agent_id].deterministic_choose_action(full_state, previous_actions)
                    if main_process:
                        print('Agent {} taking action'.format(list_of_agents[active_agent_id].name))
                if list_of_agents[active_agent_id].multi_process == False and main_process:
                    action = list_of_agents[active_agent_id].deterministic_choose_action(full_state, previous_actions)
                    if main_process:
                        print('Agent {} taking action'.format(list_of_agents[active_agent_id].name))

                if main_process:
                    if action is None:
                        print('None action by {}'.format(list_of_agents[active_agent_id].name))
                        print('Current state of the game')
                        state_str = StateAsDict(self.env.current_state_of_the_game)
                        print(state_str)
                    previous_actions = [action]
                    if render_game:
                        self.env.render()

                    full_state, reward, is_done, info = self.env.deterministic_step(action)

                    if is_done:
                        results_dict[list_of_agents[active_agent_id].my_name_with_id()] = \
                            OneAgentStatistics(reward, self.env.points_of_player_by_id(active_agent_id),
                                               int(reward == 1))
                        if first_winner_id is None:
                            first_winner_id = active_agent_id
                        checked_all_players_after_first_winner = active_agent_id == (first_winner_id - 1) % len(
                            list_of_agents)

                active_agent_id = (active_agent_id + 1) % len(list_of_agents)
                number_of_actions += 1

                #broadcast to all processes

                is_done = mpi_communicator.bcast(is_done, root=0)
                number_of_actions = mpi_communicator.bcast(number_of_actions, root=0)
                checked_all_players_after_first_winner = mpi_communicator.bcast(checked_all_players_after_first_winner, root=0)
                active_agent_id =  mpi_communicator.bcast(active_agent_id, root=0)

            if main_process:
                one_game_statistics = GameStatisticsDuels(list_of_agents)
                one_game_statistics.register_from_dict(results_dict)
            if not main_process:
                one_game_statistics = None


            return one_game_statistics



