import pickle
import random
import numpy as np

import gin
from tqdm import tqdm

from nn_models.architectures.average_pool_v0 import StateEvaluator
from nn_models.utils.vectorizer import Vectorizer
from training_data.data_generation.gen_data_lvl0 import load_data_for_model

gin.parse_config_file('params_v1.gin')
model = StateEvaluator()

# data = load_data_for_model('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/combined.pickle')
# test_size = 500
# X = Vectorizer().many_states_to_input(data[0])
# Y = np.array(data[1])
# # X_val = Vectorizer().many_states_to_input(data[0][-test_size:])
# # Y_val = np.array(data[1][-test_size:])

model.train_network_on_many_sets('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/ep_',
                                 '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/validation_flat.pickle',
                                 epochs=18)


