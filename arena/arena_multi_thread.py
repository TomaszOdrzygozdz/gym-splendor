from copy import deepcopy

import gym
from mpi4py import MPI
from typing import List

from agent import Agent
from arena.arena import Arena
from arena.game_statistics import GameStatistics
from arena.one_agent_statistics import OneAgentStatistics

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

class ArenaMultiThread:

    def __init__(self,
                 environment_id='gym_splendor_code:splendor-v0'):

        self.environment_id = environment_id

    def run_many_games_multi_threads(self,
                                      list_of_agents: List[Agent],
                                      number_of_games: int,
                                      shuffle_agents: bool = True,
                                      starting_agent_id=0):

        #prepare jobs for all threads
        games_per_thread = int(number_of_games / size)
        remaining_games = number_of_games % size

        #results = None
        if rank < remaining_games:
            local_arena = Arena(self.environment_id)
            results = local_arena.run_many_games_single_thread(list_of_agents, games_per_thread + 1)

        if rank >= remaining_games:
            local_arena = Arena(self.environment_id)
            results = local_arena.run_many_games_single_thread(list_of_agents, games_per_thread)



        #send all results to the main thread:
        # collected_results = comm.gather(results, root=0)
        #
        # if rank==0:
        #     #sum all results
        #     all_threads_results = GameStatistics()
        #     all_threads_results.create_from_list_of_agents(list_of_agents)
        #     for one_thread_results in collected_results:
        #         all_threads_results = all_threads_results + one_thread_results
        #
        #     return all_threads_results
        #
        # if rank > 0:
        #     return None




