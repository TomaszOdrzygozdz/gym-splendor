import random
from typing import List

import gym

from agents.abstract_agent import Agent
from arena.game_statistics_duels import GameStatisticsDuels
from arena.arena import Arena

from mpi4py import MPI

comm = MPI.COMM_WORLD
n_proc = MPI.COMM_WORLD.Get_size()
my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank == 0

class MultiArena:

    def __init__(self) -> None:

        self.env_initialized = False
        self.name = 'Multi Process Arena'

    def initialize_env(self, environment_id: str = 'gym_splendor_code:splendor-deterministic-v0'):
        """Arena has its private environment to run the game."""
        self.env = gym.make(environment_id)

    def run_multi_process_self_play(self, mode, agent: Agent, render_game = False):
        local_arena = Arena()
        local_arena.run_self_play(mode, agent, render_game=render_game, mpi_communicator=comm)

    def run_many_duels(self, mode, list_of_agents: List[Agent], n_games: int, n_proc_per_agent:int, shuffle: bool = True):

        assert n_games > 0, 'Number of games must be positive.'
        assert len(list_of_agents) == 2, 'This method can run on exactly two agents.'

        local_arena = Arena()
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
        #set agents colors:
        for agent in list_of_agents:
            agent.set_color(my_color)

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
            print('My color = {} I have to take = {} games'.format(my_color, my_games))
        local_results = GameStatisticsDuels(list_of_agents[:1], list_of_agents[1:])

        for _ in range(my_games):
            if shuffle:
                starting_agent_id = random.choice(range(2))
            one_game_results = local_arena.run_one_duel(mode, list_of_agents, mpi_communicator=new_communicator)
            if local_main:
                local_results.register(one_game_results)

        #Gather all results:
        combined_results_list = comm.gather(local_results, root=0)

        if main_process:
            global_results = GameStatisticsDuels(list_of_agents[:1], list_of_agents[1:])
            for local_result in combined_results_list:
                global_results.register(local_result)

            return global_results
