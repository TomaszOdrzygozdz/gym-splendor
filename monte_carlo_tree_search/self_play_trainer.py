import gym
from mpi4py import MPI
import pandas as pd
import numpy as np

from agents.abstract_agent import Agent
from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from nn_models.tree_data_collector import TreeDataCollector

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

class SelfPlayTrainer:

    def __init__(self, mode, iteration_limit, rollout_repetition, choose_best):
        assert mode == 'dqn', 'You must provide mode of training'
        self.iteration_limit = iteration_limit
        self.rollout_repetition = rollout_repetition
        self.data_collector = TreeDataCollector()
        self.opponent = RandomAgent(distribution='first_buy')
        self.choose_best = choose_best
        self.env = gym.make('splendor-v0')

    def prepare_training(self, weights_file:str = None):
        self.eval_policy = ValueEvaluator(weights_file)
        self.mcts_agent = MultiMCTSAgent(iteration_limit=self.iteration_limit,
                                         evaluation_policy = self.eval_policy, rollout_policy=RandomRollout(),
                                         rollout_repetition=self.rollout_repetition,
                                         only_best=self.choose_best)



    def run_self_play(self, mode: str = 'deterministic', alpha=0.1, epochs=2):

        self.env.reset()
        self.env.set_active_player(0)
        # set players names:
        is_done = False
        # set the initial agent id
        # set the initial observation
        observation = self.env.show_observation(mode)

        number_of_actions = 0
        results_dict = {}
        # Id if the player who first reaches number of points to win
        previous_actions = [None]
        action = None

        self.mcts_agent.set_self_play_mode()
        self.mcts_agent.set_communicator(comm)


        while number_of_actions < MAX_NUMBER_OF_MOVES and not is_done:
            action = self.mcts_agent.choose_action(observation, previous_actions)
            previous_actions = [action]
            self.train_network_iteration(alpha=alpha, epochs=epochs)
            #data collection and training:

            if main_process:
                observation, reward, is_done, info = self.env.step(mode, action)
                winner_id = info['winner_id']

            number_of_actions += 1
            is_done = comm.bcast(is_done, root=0)

        if main_process:
            self.env.reset()

        self.mcts_agent.unset_self_play_mode()
        self.mcts_agent.finish_game()
        print('Self-game done')

    def train_network_iteration(self, alpha=0.1, epochs=2):

        #collect data
        if main_process:
            self.data_collector.setup_root(self.mcts_agent.mcts_algorithm.return_root())
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
            self.eval_policy.model.train_model(data_frame=data_collected, epochs=epochs)

    def return_agent(self):
        return self.mcts_agent

    def full_training(self, n_repetitions, alpha, epochs):

        self.prepare_training()
        for i in range(n_repetitions):
            if main_process:
                print('Game number = {}'.format(i))
            self.run_self_play('deterministic', alpha=alpha, epochs=epochs)
            agent_to_test = self.mcts_agent
            arena = MultiArena()
            results = arena.run_many_duels('deterministic', [agent_to_test, RandomAgent(distribution='first_buy')], 1, 24)
            if main_process:
                print(results)
                self.eval_policy.model.save_weights('Weights_i = {}.h5'.format(i))
                text_file = open("Results_{}.txt".format(i), "w")
                text_file.write(results.__repr__())
                text_file.close()



