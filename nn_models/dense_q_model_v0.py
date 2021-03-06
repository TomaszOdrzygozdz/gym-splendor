import os
from gym_splendor_code.envs.mechanics.game_settings import USE_TENSORFLOW_GPU, USE_LOCAL_TF
from monte_carlo_tree_search.mcts_settings import REWARDS_FOR_HAVING_NO_LEGAL_ACTIONS

if not USE_TENSORFLOW_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if USE_LOCAL_TF:
    import tensorflow_local as tf
if not USE_LOCAL_TF:
    import tensorflow as tf

import keras
import json
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K

import pandas as pd

from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from gym_splendor_code.envs.mechanics.action import Action

from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from archive.vectorization import vectorize_state, vectorize_action




class DenseModel:

    def __init__(self):

        self.session = tf.Session()
        self.network = None

    def set_corrent_session(self):
        K.set_session(self.session)

    def create_network(self, input_size : int = 597, layers_list : List[int] = [600, 600, 600, 600]) -> None:
        '''
        This method creates network with a specific architecture
        :return:
        '''
        self.set_corrent_session()
        entries = Input(shape=(input_size,))

        for i, layer_size in enumerate(layers_list):
            print(layer_size)
            if i == 0:
                data_flow = Dense(layer_size, activation='relu')(entries)
            else:
                data_flow = Dense(layer_size, activation='relu')(data_flow)
        predictions = Dense(1, activation='tanh')(data_flow)

        optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.network = Model(inputs=entries, outputs=predictions)
        self.network.compile(optimizer=optim, loss='mean_squared_error')

        with open('architecture.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.network.summary(print_fn=lambda x: fh.write(x + '\n'))
        self.network.summary()
        self.session.run(tf.global_variables_initializer())

    def train_model(self, data_file_name=None, data_frame=None, output_weights_file_name=None, epochs=2):

        assert self.network is not None, 'You must create network before training'
        self.set_corrent_session()

        X = []
        Y = []

        if data_file_name is not None:
            data = pd.read_csv(data_file_name)

            for i, row in data.iterrows():
                    state_action_concat = json.loads(row[1]) + json.loads(row[2])
                    evaluation = row[3]
                    X.append(state_action_concat)
                    Y.append(evaluation)

        if data_frame is not None:
            data = data_frame

            for i, row in data.iterrows():
                # print('row = {}'.format(row))
                # print('row [0] =  {}'.format(row[0]))
                # print('row [1] =  {}'.format(row[1]))
                # print('row [2] =  {}'.format(row[2]))

                state_action_concat = row[1] + row[0]
                evaluation = row[2]
                X.append(state_action_concat)
                Y.append(evaluation)


        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
        self.network.fit(X_train, Y_train, batch_size=None, epochs=epochs, verbose=1)
        score = self.network.evaluate(X_test, Y_test, verbose=1)
        print('Training score = {}'.format(score))
        if output_weights_file_name is not None:
            self.network.save_weights(output_weights_file_name)

    def load_weights(self, weights_file):
        assert self.network is not None, 'You must create network before loading weights.'
        self.set_corrent_session()
        self.network.load_weights(weights_file)

    def save_weights(self, weights_file):
        assert self.network is not None, 'You must create network before saving weights.'
        self.set_corrent_session()
        self.network.save_weights(weights_file)

    def get_q_value(self, state_as_dict: StateAsDict, action : Action) -> float:
        assert self.network is not None, 'You must create network first.'
        self.set_corrent_session()
        vector_of_state = vectorize_state(state_as_dict)
        vector_of_action = vectorize_action(action)
        input_vec = np.array(vector_of_state + vector_of_action)
        return self.network.predict(x=input_vec.reshape(1, 597))[0]

    def choose_best_action(self, state_as_dict : StateAsDict, list_of_actions: List[Action]) -> Action:
        assert self.network is not None, 'You must create network first.'
        if len(list_of_actions) > 0:
            q_values_predicted = self.evaluate_list(state_as_dict, list_of_actions)
            index_of_best_action = np.argmax(q_values_predicted)
            return list_of_actions[index_of_best_action]
        else:
            return None

    def choose_best_action_with_q_value(self, state_as_dict : StateAsDict, list_of_actions: List[Action]):
        assert self.network is not None, 'You must create network first.'
        if len(list_of_actions) > 0:
            q_values_predicted = self.evaluate_list(state_as_dict, list_of_actions)
            index_of_best_action = np.argmax(q_values_predicted)
            best_value = np.max(q_values_predicted)
            return list_of_actions[index_of_best_action], best_value
        else:
            return None, -1

    def get_max_q_value(self, state_as_dict : StateAsDict, list_of_actions: List[Action]) -> Action:
        assert self.network is not None, 'You must create network first.'
        if len(list_of_actions) > 0:
            q_values_predicted = self.evaluate_list(state_as_dict, list_of_actions)
            return np.argmax(q_values_predicted)
        else:
            return REWARDS_FOR_HAVING_NO_LEGAL_ACTIONS

    def evaluate_list(self, state_as_dict : StateAsDict, list_of_actions: List[Action]):
        assert self.network is not None, 'You must create network first.'
        self.set_corrent_session()
        X = []
        if len(list_of_actions) > 0:
            vector_of_state = vectorize_state(state_as_dict)
            for action in list_of_actions:
                state_action_concat = vector_of_state + vectorize_action(action)
                X.append(state_action_concat)
            X = np.array(X)
            q_values_predicted = self.network.predict(X)
            return q_values_predicted
        else:
            return None