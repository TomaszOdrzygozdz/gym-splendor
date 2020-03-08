import gin
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer
from experiments.series_1.experiment_2.experiment_setup import TRAIN_DIR, VALID_FILE


def run_experiment_1_2():
    gin.parse_config_file('nn_models/experiments/series_1/experiment_2/params.gin')
    final_layer = ValueRegressor()
    data_transformer = IdentityTransformer()
    model = StateEncoder(final_layer=final_layer, data_transformer=data_transformer)
    model.train_network_on_many_sets(TRAIN_DIR, VALID_FILE, epochs=1000,  test_games=100)

