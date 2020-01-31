import pickle

from training_data.data_generation.gen_data_lvl0 import produce_data

#produce_data(when_to_start=10, dump_p=0.25, n_games=10, filename='lvl0.pickle')

file = open('proc_0_lvl0.pickle', 'rb')
x = pickle.load(file)
print(x)