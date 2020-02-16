import os

import random
from copy import deepcopy

from tqdm import tqdm

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
import numpy as np

from agents.state_evaluator_heuristic import StateEvaluatorHeuristic
from arena.arena_multi_thread import ArenaMultiThread
from gym_splendor_code.envs.mechanics.game_settings import USE_TQDM
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.utils.vectorizer import Vectorizer
import pickle

from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank == 0

def produce_data(when_to_start, dump_p, n_games, filename, folder):
    list_of_agents = [RandomAgent(), GreedyAgentBoost(), MinMaxAgent()]

    arek = ArenaMultiThread()
    arek.start_collecting_states()
    arek.collect_only_from_middle_game(when_to_start, dump_p)
    arek.all_vs_all('deterministic', list_of_agents, n_games)
    arek.dump_collected_states(filename, folder)

def flip_states(list_of_states, list_of_values):
    rev_states = []
    rev_values = []
    for i in range(len(list_of_states)):
        rev_state = deepcopy(list_of_states[i])
        rev_state.change_active_player()
        rev_states.append(rev_state)
        rev_values.append(-list_of_values[i])
    return rev_states, rev_values

def evaluate_states(files_dir, dump_dir):
    evaluator = StateEvaluatorHeuristic()
    list_of_files = os.listdir(files_dir)
    list_to_iterate = tqdm(list_of_files) if USE_TQDM else list_of_files
    for file_name in list_of_files:
        X = []
        Y = []
        with open(os.path.join(files_dir, file_name), 'rb') as f:
            X, _ = pickle.load(f)
            X = X[:len(X)//2]
            Y = []
            for x in X:
                state_to_eval = StateAsDict(x).to_state()
                Y.append(evaluator.evaluate(state_to_eval))

        with open(os.path.join(dump_dir, file_name), 'wb') as f:
            pickle.dump((X, Y), f)
            print(len(X))
        del X
        del Y


def pick_data_for_training(epochs_range, files_dir, dump_dir):
    states = []
    values = []

    files_list = os.listdir(files_dir)
    for epoch in epochs_range:
        print(f'Epoch = {epoch}')
        for file_name in files_list:
            print(f'Current file = {file_name}')
            with open(os.path.join(files_dir, file_name), 'rb') as f:
                one_file_data = pickle.load(f)
                for key in one_file_data:
                    if one_file_data[key]['states']:
                        random_idx = random.randint(0, len(one_file_data[key]['states']) - 1)
                        states.append(one_file_data[key]['states'][random_idx])
                        values.append(one_file_data[key]['values'][random_idx])
                del one_file_data

        print('\n Flipping \n')
        states_rev, values_rev = flip_states(states, values)
        print('States flipped')
        states = states + states_rev
        values = values + values_rev
        del states_rev
        del values_rev
        print('Ready to save')
        with open(os.path.join(dump_dir, f'epoch_{epoch}.pickle'), 'wb') as f:
            pickle.dump((states, values), f)
            del states
            del values
            states = []
            values = []


def flatten_data_from_games(source_file, target_file):
    with open(source_file, 'rb') as f:
        one_file_data = pickle.load(f)
    states = []
    values = []
    for key in one_file_data:
        states += one_file_data[key]['states']
        values += one_file_data[key]['values']
    with open(target_file, 'wb') as f:
        pickle.dump((states, values), f)

def load_data_for_model(file):
    with open(file, 'rb') as f:
        data_to_return = pickle.load(f)
    return data_to_return

