import gin
import neptune
import logging, os
import numpy as np

from keras.optimizers import Adam

from gym_splendor_code.envs.mechanics.game_settings import MAX_RESERVED_CARDS, \
    NOBLES_ON_BOARD_INITIAL
from nn_models.abstract_model import AbstractModel
from nn_models.utils.useful_keras_layers import CardNobleMasking, TensorSqueeze
from keras.callbacks import Callback

logging.disable(logging.WARNING)
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.metrics import MeanSquaredError
from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense
from nn_models.utils.named_tuples import *
import keras.backend as K

from sklearn.model_selection import train_test_split

class NeptuneMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.mse_metric = MeanSquaredError()

    def on_epoch_end(self, epoch, logs={}):
        neptune.send_metric('epoch loss', epoch, logs['loss'])
 #       print(self.validation_data)
        #d = self.model.predict(x = self.validation_data[0:62])
        neptune.send_metric('epoch test loss', epoch, self.model.evaluate(self.validation_data[:62], self.validation_data[62]))

        self.mse_metric.update_state(self.validation_data[62][1:4000],
                                    np.array([1 - 2*np.random.randint(2)
                                              for _ in range(len(self.validation_data[62][1:4000]))]))
        random_score = K.eval(self.mse_metric.result())
        neptune.send_metric('random_prediction loss', epoch, random_score)

    def on_batch_end(self, batch, logs=None):
        neptune.send_metric('batch_loss', batch, logs['loss'])
        neptune.send_metric('batch_test_loss', batch,
                            self.model.evaluate(self.validation_data[:62], self.validation_data[62]))


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
        points_encoded = TensorSqueeze(Embedding(input_dim=25, output_dim=points_dim)(*points_input))
        nobles_number_encoded = TensorSqueeze(Embedding(input_dim=4, output_dim=nobles_dim)(*nobles_number_input))

        full_player = Concatenate(axis=-1)([gems_encoded, discount_encoded, reserved_cards_encoded, points_encoded,
                                            nobles_number_encoded])
        full_player = Dense(dense_dim1, activation='relu')(full_player)
        full_player = Dense(dense_dim2, activation='relu')(full_player)
        self.layer = Model(inputs = self.inputs, outputs = full_player, name = 'player_encoder')

    def __call__(self, player_input):
        return self.layer(player_input)

@gin.configurable
class StateEvaluator(AbstractModel):
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
                experiment_name: str = None,
                epochs: int = None
                ):
       super().__init__()

       self.neptune_monitor = NeptuneMonitor()
       self.experiment_name = experiment_name
       self.epochs = epochs

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
       full_state = Dense(full_player_dense2_dim, activation='relu')(full_state)
       final_result = Dense(1, activation='tanh', name='final_layer')(full_state)
       self.network = Model(inputs = self.inputs, outputs = final_result, name = 'full_state_splendor_estimator')
       self.network.compile(Adam(), loss='mean_squared_error')
       self.params['Model name'] = 'Average pooling model'
       self.params['optimizer_name'] = 'Adam'

   def train_network(self, x_train, y_train, validation_split=None, validation_data=None, batch_size = None):
       assert self.network is not None, 'You must create network before training'

       self.start_neptune_experiment(experiment_name=self.experiment_name, description='Training dense network',
                                    neptune_monitor=self.neptune_monitor)

       if validation_data is not None:
           self.network.fit(x=x_train, y=y_train, epochs=self.epochs, batch_size=batch_size, validation_data=validation_data,
                            callbacks=[self.neptune_monitor])
       elif validation_split is not None:
           self.network.fit(x=x_train, y=y_train, epochs=self.epochs, batch_size=batch_size, validation_split=validation_split,
                            callbacks=[self.neptune_monitor])
       else:
           pass
       neptune.stop()

