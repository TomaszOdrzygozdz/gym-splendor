import time

from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
from agents.value_nn_agent import ValueNNAgent
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from nn_models.value_dense_model import ValueDenseModel

model = ValueDenseModel()
model.create_network()

 # history = model.train_model(data_file_name='E:\ML_research\gym_splendor\\nn_models\data\\value_vectorized\combined.csv',
 #                  output_weights_file_name='E:\ML_research\gym_splendor\\nn_models\weights\experiment7.h5',
 #                  epochs = 10)

agent1 = ValueNNAgent(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\experiment7.h5')
eval_policy = ValueEvaluator(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\experiment7.h5')
agent2 = MultiMCTSAgent(iteration_limit=10, only_best=None, rollout_policy=None, evaluation_policy=eval_policy,
                        rollout_repetition=1, create_visualizer=True, show_unvisited_nodes=False)

agent3 = RandomAgent(distribution='first_buy')
agent4 = RandomAgent(distribution='first_buy')

arek = MultiArena()

time_s = time.time()
results = arek.run_many_duels('deterministic', [agent2 , agent3], 1, 6)
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


