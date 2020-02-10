import os

import random
from copy import deepcopy

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
import numpy as np
from arena.arena_multi_thread import ArenaMultiThread
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.utils.vectorizer import Vectorizer
import pickle

from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_thread = my_rank == 0

def produce_data(when_to_start, dump_p, n_games, filename, folder):
    list_of_agents = [RandomAgent(), RandomAgent(distribution='first_buy'), GreedyAgentBoost(), MinMaxAgent(),
                      GreedySearchAgent()]


    arek = ArenaMultiThread()
    arek.start_collecting_states()
    arek.collect_only_from_middle_game(when_to_start, dump_p)
    #arek.run_many_games('deterministic', li, n_games=n_games)
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

def pick_data_for_training(epochs_range, files_dir, dump_dir):
    states = {f'ep_{ep}' : [] for ep in epochs_range}
    values = {f'ep_{ep}' : [] for ep in epochs_range}

    files_list = os.listdir(files_dir)

    for file_name in files_list:
        print(f'Current file = {file_name}')
        with open(os.path.join(files_dir, file_name), 'rb') as f:
            one_file_data = pickle.load(f)
            for epoch in epochs_range:
                for key in one_file_data:
                    if one_file_data[key]['states']:
                        random_idx = random.randint(0, len(one_file_data[key]['states']) - 1)
                        states[f'ep_{epoch}'].append(one_file_data[key]['states'][random_idx])
                        values[f'ep_{epoch}'].append(one_file_data[key]['values'][random_idx])

    print('\n Flipping \n')
    for epoch in epochs_range:
        states_rev, values_rev = flip_states(states[f'ep_{epoch}'], values[f'ep_{epoch}'])
        states[f'ep_{epoch}'] = states[f'ep_{epoch}'] + states_rev
        values[f'ep_{epoch}'] = values[f'ep_{epoch}'] + values_rev


        with open(os.path.join(dump_dir, f'epoch_{epoch}.pickle'), 'wb') as f:
            pickle.dump((states[f'ep_{epoch}'], values[f'ep_{epoch}']), f)

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

#produce_data(10, 0.25, 1, 'all_vs_all.pickle' , '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl1')

# X = load_data_for_model('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl1/proc_0_all_vs_all.pickle')
# for y in X:
#     print(StateAsDict(X[y]['states'][0]))
#     print(X[y]['values'][0])
#     print('\n ****** \n')
#


# produce_data(4, 0.8, 200, 'validation.pickle', '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/')
#
# flatten_data_from_games('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/validation.pickle',
#                         '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/validation_flat.pickle')


#pick_data_for_training(range(15, 20))


#X, Y = load_data_for_model('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/ep_0.pickle')

#print(len(Y))

# def data_process(list_of_sates, list_of_values, validation_size):
#     for state in list_of_sates:
#         new_state = deepcopy(state)
#         new_state.ch
