from mpi4py import MPI
from typing import List

import random
from itertools import product
from gym_splendor_code.envs.mechanics.game_settings import USE_TQDM
if USE_TQDM:
    from tqdm import tqdm


from agents.abstract_agent import Agent
from arena.single_process.arena import Arena
from arena.game_statistics_duels import GameStatisticsDuels

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
        if main_thread and USE_TQDM:
            self.progress_bar = tqdm(total=lenght, postfix=None)

    def set_progress_bar(self, value):
        if main_thread and USE_TQDM:
            self.progress_bar.n = min(value, self.progress_bar.total-1)
            self.progress_bar.update()

    def one_group_vs_other_duels(self,
                                 mode,
                                 list_of_agents1: List[Agent],
                                 list_of_agents2: List[Agent],
                                 games_per_duel: int,
                                 shuffle: bool = True):

        #create all pairs to fightd
        all_pairs = list(product(list_of_agents1, list_of_agents2))
        pairs_to_duel = [pair for pair in all_pairs if pair[0] != pair[1]]
        #create list of jobs:
        list_of_jobs = pairs_to_duel*games_per_duel
        #calculate jobs per thread:
        jobs_per_thread = int(len(list_of_jobs) / n_proc)
        remaining_jobs = len(list_of_jobs) % n_proc
        #create local arena
        local_arena = Arena()
        local_results = GameStatisticsDuels(list_of_agents1, list_of_agents2)
        add_remaining_job = int(my_rank < remaining_jobs)

        #create progress bar
        self.create_progress_bar(len(list_of_jobs))

        for game_id in range(0, jobs_per_thread + add_remaining_job):
            pair_to_duel = list_of_jobs[game_id*n_proc + my_rank]
            if shuffle:
                starting_agent_id = random.choice(range(2))
            one_game_results = local_arena.run_one_duel(mode, list(pair_to_duel))
            local_results.register(one_game_results)
            if main_thread:
                self.set_progress_bar((game_id+1)*n_proc)

        #gather all results
        cumulative_results_unprocessed = comm.gather(local_results, root=0)
        if main_thread:
            cumulative_results = GameStatisticsDuels(list_of_agents1, list_of_agents2)
            for one_thread_results in cumulative_results_unprocessed:
                cumulative_results.register(one_thread_results)

            return cumulative_results

    def run_many_games(self, mode, list_of_agents, n_games):
        return self.one_group_vs_other_duels(mode, [list_of_agents[0]], [list_of_agents[1]], games_per_duel=n_games)

    def all_vs_all(self, mode, list_of_agents, n_games):
        return self.one_group_vs_other_duels(mode, list_of_agents, list_of_agents, games_per_duel=n_games)
