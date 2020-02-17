import gin
from nn_models.architectures.average_pool_v0 import StateEvaluator
gin.parse_config_file('/net/people/plgtodrzygozdz/gym-splendor/nn_models/experiments/series_1/experiment_1/params_v1.gin')

model = StateEvaluator()
model.train_network_on_many_sets('/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/train_epochs_eval',
                                 '/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/valid_epoch/valid_0.pickle',
                                 epochs=500, epochs_repeat=11, test_games=10)


# import gin
# from nn_models.architectures.average_pool_v0 import StateEvaluator
# gin.parse_config_file('nn_models/experiments/series_1/experiment_1/params_v1.gin')
#
# model = StateEvaluator()
# model.train_network_on_many_sets('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/test0/train_epochs/epoch_',
#                                  '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/test0/valid_epochs/epoch_0.pickle',
#                                  epochs=20, epochs_repeat=10,  test_games=1)
