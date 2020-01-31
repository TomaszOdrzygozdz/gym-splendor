from copy import deepcopy

from agents.greedysearch_agent import GreedySearchAgent
from agents.random_agent import RandomAgent
import numpy as np
from arena.arena_multi_thread import ArenaMultiThread
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from nn_models.utils.vectorizer import Vectorizer
import pickle

from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank == 0


def produce_data(when_to_start, dump_p, n_games, filename):
    agent1 = RandomAgent()
    agent2 = GreedySearchAgent()

    arek = ArenaMultiThread()
    arek.start_collecting_states()
    arek.collect_only_from_middle_game(when_to_start, dump_p)
    arek.run_many_games('deterministic', [agent1, agent2], n_games=n_games)
    arek.dump_collected_states(filename)
