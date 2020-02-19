import gin

from gym_splendor_code.envs.mechanics.state import State
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor

gin.parse_config_file('/home/tomasz/ML_Research/splendor/gym-splendor/nn_models/experiments/series_1/experiment_1/params.gin')

x = StateEncoder()

f = State()
print(x.get_value(f))