import pickle

import gin
import neptune
import logging, os
import numpy as np

from keras.optimizers import Adam

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.minmax_agent import MinMaxAgent
from agents.random_agent import RandomAgent
from agents.value_nn_agent import ValueNNAgent
from archive.states_list import list_of_fixes_states
from arena.arena import Arena
from gym_splendor_code.envs.mechanics.game_settings import MAX_RESERVED_CARDS, \
    NOBLES_ON_BOARD_INITIAL
from nn_models.abstract_model import AbstractModel
from nn_models.utils.useful_keras_layers import CardNobleMasking, TensorSqueeze
from keras.callbacks import Callback

from nn_models.utils.vectorizer import Vectorizer
from training_data.data_generation.gen_data_lvl0 import load_data_for_model

logging.disable(logging.WARNING)

from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense
from nn_models.utils.named_tuples import *
import keras.backend as K

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split

class NeptuneMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.win_rates_history = []

    def reset_epoch_counter(self):
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        neptune.send_metric('epoch loss', self.epoch, logs['loss'])
        neptune.send_metric('epoch test loss', self.epoch, self.model.evaluate(self.validation_data[:62], self.validation_data[62]))
        self.epoch += 1

    def log_win_rates(self, type,  results):
        self.win_rates_history[type].append(results)
        if len(self.win_rates_history) < 5:
            neptune.send_metric(f'win rate {type}', self.epoch, np.mean(self.win_rates_history))
        else:
            neptune.send_metric(f'win rate {type}', self.epoch, np.mean(self.win_rates_history[-4:]))
        #neptune.send_metric('medium win rate', self.epoch, medium_results)
        #neptune.send_metric('hard win rate', self.epoch, hard_results)

    def log_state_values(self, results):
        for i in range(len(results)):
            neptune.send_metric(f'eval of state {i}', self.epoch, results[i])

    def log_histograms(self, file1, file2):
        img1 = Image.open(file1)
        neptune.send_image('train set histogram', img1)
        img2 = Image.open(file2)
        neptune.send_image('val set histogram', img2)


class GemsEncoder:
    def __init__(self, output_dim):
        self.inputs = [Input(batch_shape=(None, 1), name='gems_{}'.format(color).replace('GemColor.', '')) for color in GemColor]
        color_embeddings = [Embedding(input_dim=6,
                                                              name='embd_gem_{}'.format(color).replace('GemColor.', ''),
                                                              output_dim=output_dim)(self.inputs[color.value])
                                                    for color in GemColor]

        gems_concatenated = Concatenate(axis=-1)(color_embeddings)
        gems_concatenated = TensorSqueeze(gems_concatenated)
        self.layer = Model(inputs = self.inputs, outputs = gems_concatenated, name = 'gems_encoder')

    def __call__(self, list_of_gems):
        return self.layer(list_of_gems)

class PriceEncoder:
    def __init__(self, output_dim):
        self.inputs = [Input(batch_shape=(None, None, 1), name='gem_{}'.format(color).replace('GemColor.', '')) for color in GemColor
                       if color != GemColor.GOLD]
        price_embeddings = [Embedding(input_dim=25,
                                                              name='embd_gem_{}'.format(color),
                                                              output_dim=output_dim)(self.inputs[color.value-1])
                                                    for color in GemColor if color != GemColor.GOLD]
        price_concatenated = Concatenate(axis=-1)(price_embeddings)
        self.layer = Model(inputs=self.inputs, outputs=price_concatenated, name='price_encoder')
    def __call__(self, list_of_gems):
        return self.layer(list_of_gems)

class ManyCardsEncoder:
    def __init__(self, seq_dim, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim,  max_points=25):
        self.price_encoder = PriceEncoder(output_dim=price_dim)
        self.inputs = [Input(batch_shape=(None, seq_dim), name='{}'.format(x)) for x in CardTuple._fields] +\
                      [Input(batch_shape=(None, seq_dim), name='cards_mask')]
        profit_embedded = Embedding(input_dim=5, output_dim=profit_dim, name='profit_embedd')(self.inputs[0])
        price_encoded = self.price_encoder.layer(self.inputs[1:-2])
        points_embedded = Embedding(input_dim=max_points, output_dim=points_dim, name='points_embedd')(self.inputs[6])
        cards_mask = self.inputs[7]
        full_cards = Concatenate(axis=-1)([profit_embedded, price_encoded, points_embedded])
        full_cards  = Dense(units=dense1_dim, activation='relu')(full_cards)
        full_cards = Dense(units=dense2_dim, activation='relu')(full_cards)
        full_cards_reduced = CardNobleMasking([full_cards, cards_mask])
        self.layer = Model(inputs = self.inputs, outputs = full_cards_reduced, name = 'card_encoder')
    def __call__(self, card_input_list):
        return self.layer(card_input_list)

class ManyNoblesEncoder:
    def __init__(self, price_dim, dense1_dim, dense2_dim):
        self.price_encoder = PriceEncoder(output_dim=price_dim)
        self.inputs = [Input(batch_shape=(None, NOBLES_ON_BOARD_INITIAL), name=x) for x in PriceTuple._fields] +\
                      [Input(batch_shape=(None, NOBLES_ON_BOARD_INITIAL), name='nobles_mask')]
        price_input = self.inputs[0:5]
        nobles_mask = self.inputs[5]
        price_encoded = self.price_encoder.layer(price_input)
        full_nobles = Dense(dense1_dim, activation='relu')(price_encoded)
        full_nobles = Dense(dense2_dim, activation='relu')(full_nobles)
        full_nobles_averaged = CardNobleMasking([full_nobles, nobles_mask])
        self.layer = Model(inputs = self.inputs, outputs = full_nobles_averaged, name='noble_encoder')
    def __call__(self, noble_input_list):
        return self.layer(noble_input_list)

class BoardEncoder:
    def __init__(self, gems_encoder : GemsEncoder, nobles_encoder: ManyNoblesEncoder, cards_encoder: ManyCardsEncoder, dense_dim1, dense_dim2):

        gems_input = [Input(batch_shape=(None, 1), name='gems_{}'.format(color).replace('GemColor.', '')) for color in GemColor]
        cards_input = [Input(batch_shape=(None, MAX_CARDS_ON_BORD), name='card_{}'.format(x)) for x in CardTuple._fields]
        nobles_input = [Input(batch_shape=(None, NOBLES_ON_BOARD_INITIAL), name = 'noble_{}'.format(x)) for x in NobleTuple._fields]
        cards_mask = [Input(batch_shape=(None, 12), name='cards_mask')]
        nobles_mask = [Input(batch_shape=(None, 3), name='nobles_mask')]
        self.inputs =  gems_input + cards_input + nobles_input + cards_mask + nobles_mask
        gems_encoded = gems_encoder(gems_input)
        cards_encoded = cards_encoder(cards_input + cards_mask)
        nobles_encoded = nobles_encoder(nobles_input + nobles_mask)
        full_board = Concatenate(axis=-1)([gems_encoded, nobles_encoded, cards_encoded])
        full_board = Dense(dense_dim1, activation='relu')(full_board)
        full_board = Dense(dense_dim2, activation='relu')(full_board)
        self.layer = Model(inputs = self.inputs, outputs = full_board, name='board_encoder')
    def __call__(self, board_tensor):
        return self.layer(board_tensor)


class PlayersInputGenerator:
    def __init__(self, prefix):
        gems_input = [Input(batch_shape=(None, 1), name=prefix+'pl_gems_{}'.format(color).replace('GemColor.', '')) for color
                      in GemColor]
        discount_input = [Input(batch_shape=(None, 1), name=prefix+'gem_{}'.format(color).replace('GemColor.', '')) for color
                          in GemColor
                          if color != GemColor.GOLD]
        reserved_cards_input = [Input(batch_shape=(None, MAX_RESERVED_CARDS), name=prefix+'res_card_{}'.format(x)) for x in
                                CardTuple._fields]
        points_input = [Input(batch_shape=(None, 1), name=prefix+'player_points')]
        nobles_number_input = [Input(batch_shape=(None, 1), name=prefix+'player_nobles_number')]
        reserved_cards_mask_input = [Input(batch_shape=(None, MAX_RESERVED_CARDS), name=prefix+'reserved_cards_mask')]
        self.inputs =  gems_input + discount_input +  reserved_cards_input + points_input + nobles_number_input +\
                       reserved_cards_mask_input

class PlayerEncoder:
    def __init__(self, gems_encoder : GemsEncoder, price_encoder : PriceEncoder,
                 reserved_cards_encoder: ManyCardsEncoder, points_dim, nobles_dim, dense_dim1, dense_dim2):

        gems_input = [Input(batch_shape=(None, 1), name='pl_gems_{}'.format(color).replace('GemColor.', '')) for color in GemColor]
        discount_input = [Input(batch_shape=(None, 1), name='gem_{}'.format(color).replace('GemColor.', '')) for color in GemColor
                       if color != GemColor.GOLD]
        reserved_cards_input = [Input(batch_shape=(None, MAX_RESERVED_CARDS), name='res_card_{}'.format(x)) for x in CardTuple._fields]
        points_input = [Input(batch_shape=(None, 1), name='player_points')]
        nobles_number_input = [Input(batch_shape=(None, 1), name='player_nobles_number')]
        reserved_cards_mask_input = [Input(batch_shape=(None, MAX_RESERVED_CARDS), name='reserved_cards_mask')]

        self.inputs =  gems_input + discount_input +  reserved_cards_input + points_input + nobles_number_input +\
                       reserved_cards_mask_input

        gems_encoded = gems_encoder(gems_input)
        discount_encoded = TensorSqueeze(price_encoder(discount_input))
        reserved_cards_encoded = reserved_cards_encoder(reserved_cards_input + reserved_cards_mask_input)
        points_encoded = TensorSqueeze(Embedding(input_dim=30, output_dim=points_dim)(*points_input))
        nobles_number_encoded = TensorSqueeze(Embedding(input_dim=4, output_dim=nobles_dim)(*nobles_number_input))

        full_player = Concatenate(axis=-1)([gems_encoded, discount_encoded, reserved_cards_encoded, points_encoded,
                                            nobles_number_encoded])
        full_player = Dense(dense_dim1, activation='relu')(full_player)
        full_player = Dense(dense_dim2, activation='relu')(full_player)
        self.layer = Model(inputs = self.inputs, outputs = full_player, name = 'player_encoder')

    def __call__(self, player_input):
        return self.layer(player_input)

@gin.configurable
class StateEncoder(AbstractModel):
   def __init__(self,
                gems_encoder_dim : int = None,
                price_encoder_dim : int = None,
                profit_encoder_dim : int = None,
                cards_points_dim: int = None,
                cards_dense1_dim: int = None,
                cards_dense2_dim: int = None,
                board_nobles_dense1_dim : int = None,
                board_nobles_dense2_dim : int = None,
                full_board_dense1_dim: int = None,
                full_board_dense2_dim: int = None,
                player_points_dim: int = None,
                player_nobles_dim: int = None,
                full_player_dense1_dim: int = None,
                full_player_dense2_dim: int = None,
                final_layer= None,
                data_transformer = None,
                network_name: str = None
                ):
       super().__init__()
       self.vectorizer = Vectorizer()
       self.final_layer = final_layer
       self.data_transformer = data_transformer

       self.params['data transormation'] = self.data_transformer.name
       self.params['final layer name'] = self.final_layer.name
       self.params['gems_encoder_dim'] = gems_encoder_dim
       self.params['gems_encoder_dim'] = gems_encoder_dim
       self.params['price_encoder_dim'] = price_encoder_dim
       self.params['profit_encoder_dim'] = profit_encoder_dim
       self.params['cards_points_dim'] = cards_points_dim
       self.params['cards_dense1_dim'] = cards_dense1_dim
       self.params['cards_dense2_dim'] = cards_dense2_dim
       self.params['board_nobles_dense1_dim'] = board_nobles_dense1_dim
       self.params['board_nobles_dense2_dim'] = board_nobles_dense2_dim
       self.params['full_board_dense1_dim']= full_board_dense1_dim
       self.params['full_board_dense2_dim'] = full_board_dense2_dim
       self.params['player_points_dim'] = player_points_dim
       self.params['player_nobles_dim'] = player_nobles_dim
       self.params['full_player_dense1_dim'] = full_player_dense1_dim
       self.params['full_player_dense2_dim']= full_player_dense2_dim


       self.arena = Arena()
       self.network_agent = ValueNNAgent(self)
       self.easy_opp = RandomAgent(distribution='first_buy')
       self.medium_opp = GreedyAgentBoost()
       self.hard_opp = MinMaxAgent()


       self.neptune_monitor = NeptuneMonitor()
       self.network_name = network_name

       self.gems_encoder = GemsEncoder(gems_encoder_dim)
       self.price_encoder = PriceEncoder(price_encoder_dim)
       self.board_encoder = BoardEncoder(self.gems_encoder,
                                          ManyNoblesEncoder(price_encoder_dim,
                                                            board_nobles_dense1_dim,
                                                            board_nobles_dense2_dim),
                                          ManyCardsEncoder(MAX_CARDS_ON_BORD,
                                                           profit_encoder_dim,
                                                           price_encoder_dim,
                                                           cards_points_dim,
                                                           cards_dense1_dim,
                                                           cards_dense2_dim
                                                           ),
                                          full_board_dense1_dim,
                                          full_board_dense2_dim)
       self.player_encoder = PlayerEncoder(self.gems_encoder,
                                            self.price_encoder,
                                            ManyCardsEncoder(MAX_RESERVED_CARDS,
                                                             profit_encoder_dim,
                                                             price_encoder_dim,
                                                             cards_points_dim,
                                                             cards_dense1_dim,
                                                             cards_dense2_dim
                                                             ),
                                            player_points_dim,
                                            player_nobles_dim,
                                            full_player_dense1_dim,
                                            full_player_dense2_dim)
       active_player_input = PlayersInputGenerator('active_').inputs
       other_player_input = PlayersInputGenerator('other_').inputs
       board_input = self.board_encoder.inputs
       self.inputs = board_input + active_player_input + other_player_input
       board_encoded = self.board_encoder(board_input)
       active_player_encoded = self.player_encoder(active_player_input)
       other_player_encoded = self.player_encoder(other_player_input)
       full_state = Concatenate(axis=-1)([board_encoded, active_player_encoded, other_player_encoded])
       full_state = Dense(full_player_dense1_dim, activation='relu')(full_state)
       final_state = Dense(full_player_dense2_dim, activation='relu')(full_state)
       result = self.final_layer(final_state)
       self.layer = Model(inputs = self.inputs, outputs = final_state, name = 'full_state_splendor_estimator')
       self.network = Model(inputs = self.inputs, outputs = result, name = 'full_state_splendor_estimator')
       self.network.compile(Adam(), loss='mean_squared_error')
       self.params['Model name'] = 'Average pooling model'
       self.params['optimizer_name'] = 'Adam'

   def get_value(self, state):
       prediciton = self.network.predict(self.vectorizer.state_to_input(state))
       return self.final_layer.get_value(prediciton)

   def train_on_mcts_data(self, data_frame, train_epochs:int):
       X = data_frame['state']
       Y = data_frame['mcts_value']
       X = self.vectorizer.many_states_to_input(X)
       Y = self.data_transformer.transform_array(Y)
       fit_history = self.network.fit(X, Y, epochs=train_epochs)
       return fit_history


   def train_network_on_many_sets(self, train_dir=None, validation_file=None, epochs=None, batch_size=None,
                                  test_games=1):
       assert self.network is not None, 'You must create network before training'

       with open(validation_file, 'rb') as f:
           X_val, Y_val = pickle.load(f)

       X_val = self.vectorizer.many_states_to_input(X_val)
       Y_val = self.data_transformer.transform_array(Y_val)
       self.neptune_monitor.reset_epoch_counter()
       file1, file2 = self.gather_data_info(train_dir, validation_file)
       self.start_neptune_experiment(experiment_name=self.network_name, description='Training avg_pool arch network',
                                     neptune_monitor=self.neptune_monitor)
       self.neptune_monitor.log_histograms(file1, file2)
       files_for_training = os.listdir(train_dir)
       for epoch in range(epochs):
           print(f'\n Epoch {epoch}: \n')
           file_epoch = epoch % len(files_for_training)
           X, Y = load_data_for_model(os.path.join(train_dir, files_for_training[file_epoch]))
           X = self.vectorizer.many_states_to_input(X)
           Y = self.data_transformer.transform_array(Y)
           self.network.fit(x=X, y=Y, epochs=1, batch_size=batch_size,
                            validation_data=(X_val, Y_val),
                            callbacks=[self.neptune_monitor])
           del X
           del Y

       neptune.stop()

   def dump_weights(self, file_name):
       self.network.save_weights(file_name)

   def load_weights(self, file_name):
       self.network.load_weights(file_name)

   def gather_data_info(self, train_dir, validation_file):
       list_of_files = os.listdir(train_dir)
       example_file = list_of_files[0]
       with open(os.path.join(train_dir, example_file), 'rb') as f1:
           _, Y_ex = pickle.load(f1)
       with open(validation_file, 'rb') as f2:
           _, Y_val = pickle.load(f2)
       self.params['train set size'] = len(Y_ex)
       self.params['valid set size'] = len(Y_val)
       file1 = os.path.join('temp', 'train_hist.png')
       file2 = os.path.join('temp', 'valid_hist.png')
       Y_ex = self.data_transformer.transform_array(Y_ex)
       Y_val = self.data_transformer.transform_array(Y_val)
       plt.hist(Y_ex, bins=100)
       plt.savefig(file1)
       plt.clf()
       plt.hist(Y_val, bins=100)
       plt.savefig(file2)
       return file1, file2

   def check_performance(self, n_games, opponents):
       performance_results = {}
       if 'easy' in opponents:
           easy_results = self.arena.run_many_duels('deterministic', [self.network_agent, self.easy_opp], n_games,
                                                    shuffle_agents=True)
           _, _, easy_win_rate = easy_results.return_stats()
           performance_results['easy'] = easy_win_rate / n_games
       if 'medium' in opponents:
           medium_results = self.arena.run_many_duels('deterministic', [self.network_agent, self.medium_opp], n_games,
                                                    shuffle_agents=True)
           _, _, medium_win_rate = medium_results.return_stats()
           performance_results['medium'] = medium_win_rate / n_games
       if 'hard' in opponents:
           hard_results = self.arena.run_many_duels('deterministic', [self.network_agent, self.hard_opp], n_games,
                                                    shuffle_agents=True)
           _, _, hard_win_rate = hard_results.return_stats()
           performance_results['hard'] = hard_win_rate / n_games
       return performance_results

   def run_test(self, n_games):
       results = self.check_performance(n_games, ['easy'])
       self.neptune_monitor.log_win_rates(['easy'], results)

   def evaluate_fixed_states(self):
       results = [self.get_value(f_state) for f_state in list_of_fixes_states]
       self.neptune_monitor.log_state_values(results)

class DataTransformerExp:
    def __init__(self, exp):
        self.exponent = exp
        self.name = f'f(x) = x^{self.exponent}'

    def transform(self, value):
        if value < 0:
            return -abs(value)**self.exponent
        return value**self.exponent

    def __call__(self, value):
        return self.transform(value)

    def move_towards_identity(self):
        self.exponent = 0.2 + 0.8*self.exponent

    def transform_array(self, values):
        return np.array([self.transform(v) for v in values])

class IdentityTransformer:
    def __init__(self):
        self.name = 'f(x) = x'

    def transform(self, value):
        return value

    def __call__(self, value):
        return self.transform(value)

    def transform_array(self, values):
        return values


class ValueRegressor:
    def __init__(self):
        self.layer = Dense(1, activation='linear')
        self.name = 'Value regression'

    def __call__(self, state_input):
        return self.layer(state_input)

    def get_value(self, network_result):
        return network_result[0][0]

