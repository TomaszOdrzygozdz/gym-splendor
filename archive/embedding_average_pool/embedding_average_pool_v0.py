import os
import gin

from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.game_settings import USE_TENSORFLOW_GPU, Row, GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from archive.neptune_settings import USE_NEPTUNE
from nn_models.abstract_model import AbstractModel

if not USE_TENSORFLOW_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, Dense

import pandas as pd

from typing import List

import numpy as np
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

if USE_NEPTUNE:
    import neptune

    class NeptuneMonitor(Callback):
        def on_epoch_end(self, epoch, logs={}):
            neptune.send_metric('loss', epoch, logs['loss'])
            neptune.send_metric('test loss', epoch, self.model.evaluate(self.validation_data[0], self.validation_data[1]))

        # def on_batch_end(self, epoch, logs={}):
        #     neptune.send_metric('loss [batch]', logs['loss'])
        #     neptune.send_metric('test loss [batch]', self.model.evaluate(self.validation_data[0], self.validation_data[1]))

@gin.configurable
class EmbeddingAveragePool_v0(AbstractModel):
    '''Architecture v1 proposed by Adam'''
    def __init__(self,
                 model_name : str,
                 architecture_params: dict,
                 ):
        super().__init__()
        self.params['Model name'] = model_name

    def create_network(self, input_size : int = 498, layers_list : List[int] = [800, 800, 800, 800]) -> None:
        '''
        This method creates network with a specific architecture
        :return:
        '''

        #board input:
        gems_on_board_input = Input(shape=(36,), name='gems_on_board')
        cards_on_board_input_list = [Input(shape=(51,), name='card_{}th'.format(i)) for i in range(12)]
        nobles_on_board_list = [Input(shape=(25,), name='noble_{}th'.format(i)) for i in range(3)]

        #active_player:
        active_player_discount = Input(shape=(80,), name='active_player_discount')
        active_player_gems = Input(shape=(36,0), name='active_player_gems')
        active_player_reserved_cards = [Input(shape=(51,0), name='active_rsv_card_{}th'.format(i)) for i in range(3)]
        active_player_points = Input(shape = (25,), name='active_player_points')

        #TODO: klasa model  card_encoder
        entries = Input(shape=(input_size,))



        for i, layer_size in enumerate(layers_list):
            print(layer_size)
            if i == 0:
                data_flow = Dense(layer_size, activation='relu')(entries)
            else:
                data_flow = Dense(layer_size, activation='relu')(data_flow)
                #data_flow = Dropout(rate=0.1)(data_flow)
        predictions = Dense(1, activation='tanh')(data_flow)

        self.params['Layers list'] = layers_list

        lr = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = None
        decay=0.0
        amsgrad = False

        optim = keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
        self.params['optimizer_name'] = 'Adam'
        self.params['optimizer lr'] = lr
        self.params['optimizer beta_1'] = beta_1
        self.params['optimizer beta_2'] = beta_2
        self.params['optimizer epsilon'] = None
        self.params['optimizer decay'] = 0.0
        self.params['optimizer amsgrad'] = False


        self.network = Model(inputs=entries, outputs=predictions)
        self.network.compile(optimizer=optim, loss='mean_squared_error')
        self.session.run(tf.global_variables_initializer())


    def train_model(self, data_file_name=None, data_frame=None, epochs=50, output_weights_file_name=None, experiment_name='Training'):

        #training params
        test_size = 0.05
        batch_size = None

        assert self.network is not None, 'You must create network before training'
        self.set_corrent_session()

        X = []
        Y = []

        if data_frame is None:
            assert data_file_name is not None
            data = pd.read_pickle(data_file_name)
            for i, row in data.iterrows():
                    state_vector = self.vectorize_state(row['observation'].observation_dict)
                    value = row['value']
                    X.append(state_vector)
                    Y.append(value)

        if data_frame is not None:
            data = data_frame
            for i, row in data.iterrows():
                state_vector = row[0]
                evaluation = row[-1]
                X.append(state_vector)
                Y.append(evaluation)

        X = np.array(X)
        Y = np.array(Y)

        self.params['Epochs'] = epochs
        self.params['Data set size'] = X.shape[0]

        self.params['Test set size'] = int(test_size*X.shape[0])
        self.params['batch_size'] = batch_size

        self.start_neptune_experiment(experiment_name=experiment_name, description='Training dense network',
                                      neptune_monitor=NeptuneMonitor())

        fit_history = self.network.fit(X, Y, batch_size=None, epochs=epochs, verbose=1, validation_split=test_size, callbacks=[self.neptune_monitor])
        neptune.stop()
        if output_weights_file_name is not None:
            self.network.save_weights(output_weights_file_name)
        return fit_history

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
        vector_of_state = self.vectorize_state(state_as_dict)
        input_vec = np.array(vector_of_state)
        return self.network.predict(x=input_vec.reshape(1, 498))[0][0]

    def get_value_of_vector(self, vector_of_state)->float:
        assert self.network is not None, 'You must create network first.'
        self.set_corrent_session()
        return self.network.predict(x=vector_of_state.reshape(1, 498))[0][0]

x = EmbeddingAveragePool_v0()
ddd = {GemColor.RED: 0, GemColor.GREEN : 1, GemColor.BLUE:2, GemColor.WHITE:3, GemColor.BLACK:7, GemColor.GOLD:0}
f = Card('R19', 10, Row.CHEAP, GemsCollection(gems_values_dict=ddd), discount_profit=GemColor.GREEN, victory_points=2)
b = x.card_to_vector(f)
print(len(b))