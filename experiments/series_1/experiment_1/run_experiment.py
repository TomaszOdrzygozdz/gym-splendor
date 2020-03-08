import gin
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, DataTransformerExp
from experiments.series_1.experiment_1.experiment_setup import TRAIN_DIR, VALID_FILE


def run_experiment_1():
    gin.parse_config_file('params.gin')
    final_layer = ValueRegressor()
    data_transformer = DataTransformerExp(0.2)
    model = StateEncoder(final_layer=final_layer, data_transformer=data_transformer)
    model.train_network_on_many_sets(TRAIN_DIR, VALID_FILE, epochs=1000,  test_games=100)

