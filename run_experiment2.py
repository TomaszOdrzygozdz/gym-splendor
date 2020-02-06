
from training_data.data_generation.gen_data_lvl0 import produce_data, load_data_for_model
produce_data(10, 0.25, 1, 'all_vs_all.pickle' , '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl1')

# data = load_data_for_model('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/combined.pickle')
#
# list_of_states = len(data[0])
# print(len(data[1]))

