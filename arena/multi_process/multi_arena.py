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

        #Inform all agents that game is finished
        for agent in list_of_agents:
            agent.finish_game()

        return one_game_statistics

    def run_many_duels(self, list_of_agents: List[Agent], n_games: int, n_proc_per_agent:int, shuffle: bool = True):

        n_process = comm.Get_size()
        #Determine number of games that can take place at the same time:
        n_parallel_games = int(n_process / n_proc_per_agent)
        remainder = n_process % n_proc_per_agent

        if main_process:
            print('n_parallel_games = {}, remainder = {}'.format(n_parallel_games, remainder))

        if main_process:
            print('n _ Proc per agent {}'.format(n_proc_per_agent))
        remainder = n_process % n_parallel_games
        #prepare communicators:
        list_of_processes = list(range(n_process))

        # communicators = []
        # start = 0
        # for i in range(n_parallel_games+1):
        #     if i < remainder:
        #         processes_to_add = n_proc_per_game + 1
        #         communicators.append(list_of_processes[start: start + processes_to_add])
        #         start += processes_to_add
        #     if i >= remainder:
        #         processes_to_add = n_proc_per_game
        #         communicators.append(list_of_processes[start: start + n_process])
        #         start+= n_process
        #
        # if main_process:
        #     print(communicators)

        # #create all pairs to fightd
        # all_pairs = list(product(list_of_agents1, list_of_agents2))
        # pairs_to_duel = [pair for pair in all_pairs if pair[0] != pair[1]]
        # #create list of jobs:
        # list_of_jobs = pairs_to_duel*games_per_duel
        # #calculate jobs per thread:
        # jobs_per_thread = int(len(list_of_jobs) / n_proc)
        # remaining_jobs = len(list_of_jobs) % n_proc
        # #create local arena
        # local_arena = Arena()
        # local_results = GameStatisticsDuels(list_of_agents1, list_of_agents2)
        # add_remaining_job = int(my_rank < remaining_jobs)
        #
        # #create progress bar
        # self.create_progress_bar(len(list_of_jobs))
        #
        # for game_id in range(0, jobs_per_thread + add_remaining_job):
        #     pair_to_duel = list_of_jobs[game_id*n_proc + my_rank]
        #     if shuffle:
        #         starting_agent_id = random.choice(range(2))
        #     one_game_results = local_arena.run_one_duel(list(pair_to_duel))
        #     local_results.register(one_game_results)
        #     if main_thread:
        #         self.set_progress_bar((game_id+1)*n_proc)
        #
        # #gather all results
        # cumulative_results_unprocessed = comm.gather(local_results, root=0)
        # if main_thread:
        #     cumulative_results = GameStatisticsDuels(list_of_agents1, list_of_agents2)
        #     for one_thread_results in cumulative_results_unprocessed:
        #         cumulative_results.register(one_thread_results)
        #
        #     return cumulative_results
