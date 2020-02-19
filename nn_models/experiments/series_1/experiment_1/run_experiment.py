import gin
from nn_models.architectures.average_pool_v0 import StateEncoder
from nn_models.experiments.series_1.experiment_1.experiment_setup import TRAIN_DIR, VALID_FILE


def run_experiment_1():
    gin.parse_config_file('nn_models/experiments/series_1/experiment_1/params.gin')
    model = StateEncoder()
    model.train_network_on_many_sets(TRAIN_DIR, VALID_FILE, epochs=25,  test_games=2)

