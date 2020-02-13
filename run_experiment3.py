import gin
from nn_models.architectures.average_pool_v0 import StateEvaluator
gin.parse_config_file('nn_models/experiments/series_1/experiment_1/run_experiment.py')

model = StateEvaluator()
model.train_network_on_many_sets('/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/train_epochs/epoch_',
                                 '/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/valid_epoch/valid_0.pickle',
                                 epochs=2, test_games=10)


