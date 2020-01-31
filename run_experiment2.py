import pickle

from training_data.data_generation.gen_data_lvl0 import produce_data

produce_data(when_to_start=10, dump_p=0.25, n_games=15, filename='lvl0.pickle', folder='data_lvl_0')