from agents.greedysearch_agent import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent

PARAMS_FILE = 'experiments/MCTS_series_1/params.gin'

import gin

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.utils.cluster_detection import CLUSTER
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer

gin.parse_config_file(PARAMS_FILE)
from monte_carlo_tree_search.trainers.MCTS_value_training import MCTS_value_trainer

def run_experiment():
    trainer = MCTS_value_trainer()
    trainer.include_params_file(PARAMS_FILE)
    trainer.include_params_file('gym_splendor_code/envs/mechanics/game_settings.py')

    if not CLUSTER:
        trainer.run_training_games_multi_process(RandomAgent(distribution='first_buy'),
                                                 epochs=50,
                                                 mcts_passes=50,
                                                 n_test_games=0,
                                                 exploration_ceofficient=0.41,
                                                 experiment_name='MCTS local',
                                                 weights_path='/home/tomasz/ML_Research/splendor/gym-splendor/archive/weights_tt1/',
                                                 neural_network_train_epochs=1,
                                                 confidence_threshold=1,  confidence_limit=4, count_threshold=20,
                                                 replay_buffer_n_games=50,
                                                 use_neptune = True,
                                                 tags=['local-run'])

    if CLUSTER:
        trainer.run_training_games_multi_process(GreedyAgentBoost(),
                                                 epochs=250,
                                                 mcts_passes=50,
                                                 n_test_games=0,
                                                 exploration_ceofficient=0.41,
                                                 experiment_name='MCTS with NN',
                                                 weights_path='/net/archive/groups/plggluna/plgtodrzygozdz/weights_temp/',
                                                 neural_network_train_epochs=1,
                                                 confidence_threshold=1, confidence_limit=2, count_threshold=40,
                                                 replay_buffer_n_games=100,
                                                 use_neptune=True,
                                                 tags=['cluster-run'])