import pickle
import gin

from nn_models.architectures.average_pool_v0 import StateEvaluator

gin.parse_config_file('params_v1.gin')
model = StateEvaluator()

with open('/home/tomasz/ML_Research/splendor/gym-splendor/training_data/vectorization_v0/half_data.pickle', 'rb') as f:
    data = pickle.load(f)

with open('/home/tomasz/ML_Research/splendor/gym-splendor/training_data/vectorization_v0/small_data.pickle', 'rb') as f:
    valid_data = pickle.load(f)


model.train_network(x_train=data['X'], y_train=data['Y'], validation_data=valid_data)


