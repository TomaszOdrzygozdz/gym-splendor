import gin
from nn_models.architectures.average_pool_v0 import StateEvaluator
gin.parse_config_file('/home/tomasz/ML_Research/splendor/gym-splendor/nn_models/experiments/series_1/experiment_1/params_v1.gin')

model = StateEvaluator()
model.train_network_on_many_sets('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl3/train_epochs/epoch_',
                                 '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl3/valid_epochs/valid_0.pickle',
                                 epochs=18, test_games=5)


