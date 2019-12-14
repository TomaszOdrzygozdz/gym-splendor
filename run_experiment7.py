import time

from agents.random_agent import RandomAgent
from agents.value_nn_agent import ValueNNAgent
from arena.multi_arena import MultiArena
from nn_models.value_dense_model import ValueDenseModel

model = ValueDenseModel()
model.create_network()

history = model.train_model(data_file_name='E:\ML_research\gym_splendor\\nn_models\data\\vectorized\combined.csv',
                  output_weights_file_name='E:\ML_research\gym_splendor\\nn_models\weights\experiment7.h5',
                  epochs = 2)

agent1 = ValueNNAgent(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\experiment7.h5')
agent2 = RandomAgent(distribution='first_buy')
agent3 = RandomAgent(distribution='first_buy')

arek = MultiArena()

time_s = time.time()
results = arek.run_many_duels('deterministic', [agent1, agent2], 10, 10)
print(results)


# time_e = time.time()
# print('Time taken = {}'.format(time_e-time_s))
#
#
# #from agents.random_agent import RandomAgent
# # from arena.multi_arena import MultiArena
# # from monte_carlo_tree_search.self_play_trainer import SelfPlayTrainer
# # import time
# #
# # time_s = time.time()
# # fufu = SelfPlayTrainer('dqn', iteration_limit=20, rollout_repetition=5, choose_best=0.5)
# # fufu.full_training(n_repetitions=100, alpha=0.1, epochs=2)
# #
# # time_e = time.time() - time_s
# # print(time_e)


