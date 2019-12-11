from mpi4py import MPI
import pandas as pd
import numpy as np

from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator
from nn_models.tree_data_collector import TreeDataCollector

my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

class SelfPlayTrainer:

    def __init__(self, mode, iteration_limit):
        assert mode == 'dqn', 'You must provide mode of training'
        self.iteration_limit = iteration_limit
        self.local_arena = MultiArena()
        self.data_collector = TreeDataCollector()

    def prepare_training(self, weights_file:str = None):
        self.eval_policy = evaluation_policy = ValueEvaluator(weights_file)
        self.mcts_agent = MultiProcessMCTSAgent(iteration_limit=self.iteration_limit,
                                                evaluation_policy = self.eval_policy)

    def one_train_iteration(self, alpha=0.1, epochs = 2):

        #run self play:
        self.local_arena.run_multi_process_self_play('deterministic', self.mcts_agent, render_game=False)
        #collect data
        if main_process:
            self.data_collector.setup_rooot(self.mcts_agent.mcts_algorithm.original_root())
            data_collected = self.data_collector.generate_dqn_data()
            #evaluate_data with old network:
            eval_by_old_network_values = []
            eval_to_learn = []
            for index, row in data_collected.iterrows():
                current_eval_by_old_network = self.eval_policy.evaluate_vector(np.array(row[0]))
                current_mcts_value = row[1]
                eval_by_old_network_values.append(current_eval_by_old_network)
                #print('A = {} B = {} C = {}'.format(current_mcts_value, current_eval_by_old_network,(1-alpha)*current_eval_by_old_network + alpha*current_mcts_value ))
                eval_to_learn.append((1-alpha)*current_eval_by_old_network + alpha*current_mcts_value)
            #calculate new values:
            #add columns to dataframe:
            data_collected['old_eval'] = pd.Series(eval_by_old_network_values)
            data_collected['eval_to_learn'] = pd.Series(eval_to_learn)
            self.eval_policy.model.train_model(data_frame=data_collected)

    def full_training(self, n_repetitions, alpha, epochs):
        self.prepare_training()
        for i in range(n_repetitions):
            self.one_train_iteration(alpha=alpha, epochs=epochs)
            self.eval_policy.model.save_weights()

fufu = SelfPlayTrainer('dqn', 100)
fufu.prepare_training()
for
fufu.one_train_iteration()