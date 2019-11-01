import random
from copy import deepcopy
from itertools import combinations

import gym
from mpi4py import MPI
from typing import List

from tqdm import tqdm

from agent import Agent
from arena.arena import Arena
from arena.game_statistics import GameStatistics
from arena.one_agent_statistics import OneAgentStatistics

comm = MPI.COMM_WORLD
n_proc = MPI.COMM_WORLD.Get_size()
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank == 0


class ArenaMultiThread:

    def __init__(self,
                 environment_id='gym_splendor_code:splendor-v0'):

        self.environment_id = environment_id

    def create_progress_bar(self, lenght):
        if main_thread:
            self.progress_bar = tqdm(total=lenght, postfix=None)

    def set_progress_bar(self, value):
        if main_thread:
            self.progress_bar.n = value
            self.progress_bar.update()

    def run_many_games_single_thread(self,
                                     list_of_agents: List[Agent],
                                     n_games: int,
                                     shuffle_agents: bool = True,
                                     starting_agent_id=0):

        local_arena = Arena()
        assert n_games > 0, 'Number of games must be positive'
        cumulative_results = GameStatistics()
        cumulative_results.create_from_list_of_agents(list_of_agents)

        for game_id in range(n_games):
            if shuffle_agents:
                random.shuffle(list_of_agents)
            one_game_results = local_arena.run_one_game(list_of_agents, starting_agent_id)
            # update results:
            cumulative_results = cumulative_results + one_game_results
            if main_thread:
                self.set_progress_bar(min((game_id + 1) * n_proc, self.progress_bar.total - 1))

        cumulative_results.number_of_games = n_games
        return cumulative_results

    def run_many_games_multi_thread(self,
                                    list_of_agents: List[Agent],
                                    number_of_games: int,
                                    shuffle_agents: bool = True,
                                    starting_agent_id=0):

        # prepare jobs for all threads
        games_per_thread = int(number_of_games / n_proc)
        remaining_games = number_of_games % n_proc

        self.create_progress_bar(number_of_games)
        # results = None
        if my_rank < remaining_games:
            results = self.run_many_games_single_thread(list_of_agents, games_per_thread + 1)
        if my_rank >= remaining_games:
            results = self.run_many_games_single_thread(list_of_agents, games_per_thread)

        # send all results to the main thread:
        collected_results = comm.gather(results, root=0)

        if main_thread:
            # sum all results
            all_threads_results = GameStatistics()
            all_threads_results.create_from_list_of_agents(list_of_agents)
            for one_thread_results in collected_results:
                all_threads_results = all_threads_results + one_thread_results

            return all_threads_results

        if my_rank > 0:
            return None

    def everybody_vs_everybody(self, list_of_agents: List[Agent], games_for_each_set, agents_per_game: int = 2):

        # first we create all sets of players:
        all_subsets_of_players = list(combinations(list_of_agents, agents_per_game))
        # for option in enumerate(all_subsets_of_players):
        #     print(option)

        # divide tasks between processes:
        jobs_per_thread = int(len(all_subsets_of_players) / n_proc)
        remaining = len(all_subsets_of_players) % n_proc

        local_arena = Arena()
        if main_thread:
            print('Total {} games on {} threads.'.format(len(all_subsets_of_players)*games_for_each_set, n_proc))

        if my_rank < remaining:

            for game_setup_id in range(jobs_per_thread + 1):
                agents_list = all_subsets_of_players[game_setup_id * n_proc + my_rank]
                self.run_many_games_single_thread()

        if my_rank >= remaining:
            all_jobs = []
            for game_setup_id in range(jobs_per_thread):
                agents_list = all_subsets_of_players[game_setup_id * n_proc + my_rank]
                all_jobs.append(agents_list)
