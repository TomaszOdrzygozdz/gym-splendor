from mpi4py import MPI
from typing import List

import random
import numpy as np
from itertools import combinations
from tqdm import tqdm

from agent import Agent
from arena.arena import Arena
from arena.game_statistics import GameStatistics
from arena.many_vs_many import ManyVsManyStatistics
from arena.one_agent_statistics import OneAgentStatistics

comm = MPI.COMM_WORLD
n_proc = MPI.COMM_WORLD.Get_size()
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank == 0


class ArenaMultiThread:

    def __init__(self,
                 environment_id='gym_splendor_code:splendor-v0'):

        self.environment_id = environment_id
        self.progress_bar = None

    def create_progress_bar(self, lenght):
        if main_thread:
            self.progress_bar = tqdm(total=lenght, postfix=None)

    def set_progress_bar(self, value):
        if main_thread:
            self.progress_bar.n = min(value, self.progress_bar.total-1)
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
                starting_agent_id = random.choice(range(len(list_of_agents)))
            one_game_results = local_arena.run_one_game(list_of_agents, starting_agent_id)
            # update results:
            cumulative_results = cumulative_results + one_game_results
            if main_thread and self.progress_bar is not None:
                self.set_progress_bar((game_id + 1) * n_proc)

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
            results = self.run_many_games_single_thread(list_of_agents, games_per_thread + 1, shuffle_agents)
        if my_rank >= remaining_games:
            results = self.run_many_games_single_thread(list_of_agents, games_per_thread, shuffle_agents)

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

    def all_vs_all(self, list_of_agents: List[Agent], games_for_each_combination, agents_per_game: int = 2):

        # first we create all sets of players:
        list_of_jobs = list(combinations(list_of_agents, agents_per_game))*games_for_each_combination
        #create progress bar:
        self.create_progress_bar(len(list_of_jobs))
        # divide tasks between processes:
        jobs_per_thread = int(len(list_of_jobs) / n_proc)
        remaining = len(list_of_jobs) % n_proc

        #in 1vs1 variant we can store results in an array:

        results = ManyVsManyStatistics(list_of_agents)

        add_remaining = int(my_rank < remaining)
        #create local arena (for one thread)
        local_arena = Arena()


        for my_job_id in range(jobs_per_thread + add_remaining):
            agents_list = list(list_of_jobs[my_job_id * n_proc + my_rank])
            local_arena.run_one_game(agents_list)
            if main_thread:
                self.set_progress_bar((my_job_id+1)*n_proc)

        if main_thread:
            print(results)
