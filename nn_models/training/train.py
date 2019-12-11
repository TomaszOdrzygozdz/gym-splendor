from agents.dense_nn_agent import DenseNNAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena
from nn_models.dense_model import DenseModel
from nn_models.value_dense_model import ValueDenseModel

model = ValueDenseModel()
model.create_network()

model.train_model('E:\ML_research\gym_splendor\\nn_models\data\\tree_data_960_vectorized.csv',
                  'E:\ML_research\gym_splendor\\nn_models\\value_random_rollout_960.h5',
                  epochs = 15)


# mumu = DenseNNAgent(weights_file='E:\ML_research\gym_splendor\agents\weights\minmax_480_games.h5')
# opp = RandomAgent(distribution='first_buy')
#
# arek = MultiArena()
# results = arek.run_many_duels('deterministic', [mumu, opp], n_games=20, )


