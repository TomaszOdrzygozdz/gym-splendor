import random
from itertools import product
from typing import List

import gym

from agents.abstract_agent import Agent
from arena.game_statistics_duels import GameStatisticsDuels
from arena.one_agent_statistics import OneAgentStatistics
from gym_splendor_code.envs.graphics.graphics_settings import GAME_INITIAL_DELAY
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
n_proc = MPI.COMM_WORLD.Get_size()
my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank == 0

class DeterministicMultiProcessArena:

    def __init__(self) -> None:

        self.env_initialized = False
        self.name = 'Multi Process Arena'
        self.initialize_env()


    def initialize_env(self, environment_id: str = 'gym_splendor_code:splendor-deterministic-v0'):
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)

    def run_one_duel(self,
                     list_of_agents: List[Agent],
                     starting_agent_id: int = 0,
                     mpi_communicator=None,
                     ) -> GameStatisticsDuels:
        """Runs one game between two agents.
        :param:
        list_of_agents: List of agents to play, they will play in the order given by the list
        starting_agent_id:  Id of the agent who starts the game.
        show_game: If True, GUI will appear showing the game. """

        # prepare the game
        mpi_communicator = comm if mpi_communicator is None else mpi_communicator

        #Determine agents needing multi processing

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

        while number_of_actions < MAX_NUMBER_OF_MOVES and not (is_done and checked_all_players_after_first_winner):
            action = list_of_agents[active_agent_id].choose_action(observation, previous_actions)
            previous_actions = [action]
            observation, reward, is_done, info = self.env.step(action)
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

        #Inform all agents that game is finished
        for agent in list_of_agents:
            agent.finish_game()

        return one_game_statistics

    def run_many_duels_one_pair(self, list_of_agents: List[Agent], n_games: int, n_proc_per_agent:int, shuffle: bool = True):


        assert n_games > 0, 'Number of games must be positive.'
        assert len(list_of_agents) == 2, 'This method can run on exactly two agents.'

        n_process = comm.Get_size()
        my_rank = comm.Get_rank()
        n_proc_per_agent = max(min(n_proc_per_agent, n_proc),1)

        n_parallel_games = int(n_process / n_proc_per_agent)
        remaining_processes = n_process % n_proc_per_agent
        extra_process_per_game = int(remaining_processes / n_parallel_games)
        remaining_processes_after_all = remaining_processes % n_parallel_games

        colors = []
        for i in range(n_parallel_games):
            if i < remaining_processes_after_all:
                processes_to_add = n_proc_per_agent + extra_process_per_game + 1
                colors += [i]*processes_to_add
            if i >= remaining_processes_after_all:
                processes_to_add = n_proc_per_agent + extra_process_per_game
                colors += [i] * processes_to_add

        my_color = colors[my_rank]
        if main_process:
            print(colors)

        #create communicators:
        new_communicator = comm.Split(my_color)

        #prepare jobs for each group of processes
        n_games_for_one_communicator = int(n_games / n_parallel_games)
        remaining_games = n_games % n_parallel_games

        if my_color < remaining_games:
            my_games = n_games_for_one_communicator + 1
        if my_color >= remaining_games:
            my_games = n_games_for_one_communicator

        local_main = new_communicator.Get_rank() == 0
        if local_main:
            print('My color = {} I have to take = {} games'.format(my_color, my_games ))

        local_results = GameStatisticsDuels(list_of_agents[:1], list_of_agents[1:])

        if local_main:
            for _ in range(my_games):
                if shuffle:
                    starting_agent_id = random.choice(range(2))
                one_game_results = self.run_one_duel(list_of_agents)
                local_results.register(one_game_results)

        #Gather all results:
        combined_results_list = comm.gather(local_results, root=0)

        if main_process:
            global_results = GameStatisticsDuels(list_of_agents[:1], list_of_agents[1:])
            for local_result in combined_results_list:
                global_results.register(local_result)

            print(global_results)