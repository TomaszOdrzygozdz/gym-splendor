import pickle
import gin
from keras.optimizers import Adam
import numpy as np

from archive.states_list import state_3, state_1
from nn_models.architectures.average_pool_v0 import StateEvaluator
from nn_models.utils.vectorizer import Vectorizer

gin.parse_config_file('params_v1.gin')

model = StateEvaluator()

with open('/home/tomasz/ML_Research/splendor/gym-splendor/training_data/vectorization_v0/small_data.pickle', 'rb') as f:
    data = pickle.load(f)

vectorizer = Vectorizer()
xxx = vectorizer.many_states_to_input([state_3, state_1, state_3, state_3, state_1])

model.network.compile(Adam(), 'mean_squared_error', metrics=None)

print(data['X'][0].shape)
print(data['Y'].shape)
# model.network.fit(x = xxx, y = np.array([1, 1, 1, 1, 1]).reshape(5,1))
# #model.network.predict(x = xxx)