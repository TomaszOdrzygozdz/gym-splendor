import os
from gym_splendor_code.envs.mechanics.game_settings import USE_TENSORFLOW_GPU
from monte_carlo_tree_search.mcts_settings import REWARDS_FOR_HAVING_NO_LEGAL_ACTIONS

if not USE_TENSORFLOW_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
import json
from keras.models import Model
from keras.layers import Input, Dense, Dropout, warnings
from keras import backend as K

import pandas as pd

from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.enums import GemColor

from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.vectorization import vectorize_state, vectorize_action




class ValueDenseModel:

    def __init__(self):

        self.session = tf.Session()
        self.network = None

    def set_corrent_session(self):
        K.set_session(self.session)

    def create_network(self, input_size : int = 498, layers_list : List[int] = [500, 500, 500, 500]) -> None:
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
                data_flow = Dropout(rate=0.5)(data_flow)
        predictions = Dense(1, activation='tanh')(data_flow)

        optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.network = Model(inputs=entries, outputs=predictions)
        self.network.compile(optimizer=optim, loss='mean_squared_error')

        with open('architecture.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.network.summary(print_fn=lambda x: fh.write(x + '\n'))
        self.network.summary()
        self.session.run(tf.global_variables_initializer())

    def train_model(self, data_file_name=None, data_frame=None, output_weights_file_name=None, epochs=5):

        assert self.network is not None, 'You must create network before training'
        self.set_corrent_session()

        X = []
        Y = []

        if data_frame is None:
            assert data_file_name is not None
            data = pd.read_csv(data_file_name)
            for i, row in data.iterrows():
                    state_vector = json.loads(row[1])
                    evaluation = row[-1]
                    X.append(state_vector)
                    Y.append(evaluation)

        if data_frame is not None:
            data = data_frame
            for i, row in data.iterrows():
                state_vector = row[0]
                evaluation = row[-1]
                X.append(state_vector)
                Y.append(evaluation)

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
        self.network.fit(X_train, Y_train, batch_size=None, epochs=epochs, verbose=0)
        score = self.network.evaluate(X_test, Y_test, verbose=0)
        print('Training score = {}'.format(score))
        if output_weights_file_name is not None:
            self.network.save_weights(output_weights_file_name)

    def load_weights(self, weights_file):
        assert self.network is not None, 'You must create network before loading weights.'
        self.set_corrent_session()
        self.network.load_weights(weights_file)

    def save_weights(self, output_weights_file_name):
        self.set_corrent_session()
        self.network.save_weights(output_weights_file_name)

    def get_value(self, state_as_dict: StateAsDict) -> float:
        assert self.network is not None, 'You must create network first.'
        self.set_corrent_session()
        vector_of_state = vectorize_state(state_as_dict)
        input_vec = np.array(vector_of_state)
        return self.network.predict(x=input_vec.reshape(1, 498))[0][0]

    def get_value_of_vector(self, vector_of_state)->float:
        assert self.network is not None, 'You must create network first.'
        self.set_corrent_session()
        return self.network.predict(x=vector_of_state.reshape(1, 498))[0][0]