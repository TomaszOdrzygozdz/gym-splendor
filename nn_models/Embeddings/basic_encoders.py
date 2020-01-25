import logging, os
from _ast import Lambda
from copy import deepcopy

from gym_splendor_code.envs.mechanics.game_settings import MAX_CARDS_ON_BORD, MAX_RESERVED_CARDS, \
    NOBLES_ON_BOARD_INITIAL
from nn_models.utils.own_keras_layers import CardNobleMasking, TensorSqueeze

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Layer
from keras.utils import plot_model


from archive.states_list import state_3
from nn_models.utils.named_tuples import *
from nn_models.utils.vectorizer import Vectorizer

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
        full_cards  = Dense(units=dense1_dim)(full_cards)
        full_cards = Dense(units=dense2_dim)(full_cards)
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

# PlayerTuple = namedtuple('player',  tuple_to_str(GemsTuple._fields, 'player_gems_')
#                          + tuple_to_str(CardTuple._fields, 'res_cards_') + ' points nobles')
#

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
        nobles_number_encoded = TensorSqueeze(Embedding(input_dim=3, output_dim=nobles_dim)(*nobles_number_input))

        full_player = Concatenate(axis=-1)([gems_encoded, discount_encoded, reserved_cards_encoded, points_encoded,
                                            nobles_number_encoded])
        full_player = Dense(dense_dim1, activation='relu')(full_player)
        full_player = Dense(dense_dim2, activation='relu')(full_player)
        self.layer = Model(inputs = self.inputs, outputs = full_player, name = 'player_encoder')

    def __call__(self, player_input):
        return self.layer(player_input)
#
class StateEvaluator:
   def __init__(self,
                gems_encoder_dim : int,
                price_encoder_dim : int,
                profit_encoder_dim : int,
                cards_points_dim: int,
                cards_dense1_dim: int,
                cards_dense2_dim: int,
                board_nobles_dense1_dim : int,
                board_nobles_dense2_dim : int,
                full_board_dense1_dim: int,
                full_board_dense2_dim: int,
                player_points_dim: int,
                player_nobles_dim: int,
                full_player_dense1_dim: int,
                full_player_dense2_dim: int
                ):
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
       self.inputs = active_player_input + other_player_input + board_input
       board_encoded = self.board_encoder(board_input)
       active_player_encoded = self.player_encoder(active_player_input)
       other_player_encoded = self.player_encoder(other_player_input)
       full_state = Concatenate(axis=-1)([board_encoded, active_player_encoded, other_player_encoded])
       full_state = Dense(full_player_dense1_dim, activation='relu')(full_state)
       full_state = Dense(full_player_dense2_dim, activation='relu')(full_state)
       final_result = Dense(1, activation='tanh')(full_state)

       self.layer = Model(inputs = self.inputs, outputs = final_result, name = 'full_state_splendor_estimator')


fullik = StateEvaluator(2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 2, 2)
plot_model(fullik.layer, to_file='fullik.png')
#

# pupu = PlayerEncoder(GemsEncoder(2), PriceEncoder(3), ManyCardsEncoder(3, 2, 3, 4, 5, 6), 3, 3, 3, 3)
# pupu.layer.compile(Adam(), 'mean_squared_error')
# xxx = Vectorizer().players_hand_to_input(state_3.active_players_hand())
# wyn = pupu.layer.predict(x = xxx)
# print(wyn)
#plot_model(xuxu.layer, to_file='player_encoder.png')




# bubu = BoardEncoder(GemsEncoder(3), ManyNoblesEncoder(2, 2, 2), ManyCardsEncoder(12, 2, 2, 2, 2, 2), 17, 13)
# bubu.layer.compile(Adam(), 'mean_squared_error')
# plot_model(bubu.layer, to_file='bubu.png', show_shapes=True)
# xxx1 = Vectorizer().board_to_input(state_3.board)
# xxx2 = Vectorizer().board_to_input(state_3.board)
#
# # for susu in xxx:
# #     print(susu.shape)
# wyn = bubu.layer.predict(x=xxx3)
# print(wyn)
#
# card_encoder = CardEncoder(3, 1, 1, 1, 32, 2)
# # model_inputs = Input(batch_shape=(None, 1))
# # model_outputs = card_encoder(model_inputs)
# # real_model = Model(inputs=model_inputs, outputs=model_outputs, name='real_model')
# # optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# # real_model.compile(optimizer=optim, loss='mean_squared_error')
# plot_model(card_encoder.layer, to_file='card_encoder_new.png')


# class CardEncoder:
#     def __init__(self, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim,  max_points=25):
#         self.price_encoder = PriceEncoder(output_dim=price_dim)
#         self.inputs = [Input(batch_shape=(None, 1), name='{}'.format(x)) for x in CardTuple._fields]
#         profit_embedded = Embedding(input_dim=5, output_dim=profit_dim, name='profit_embedd')(self.inputs[0])
#         price_encoded = self.price_encoder.layer(self.inputs[1:-1])
#         price_concatenated = Concatenate(axis=-1)(price_encoded)
#         points_embedded = Embedding(input_dim=max_points, output_dim=points_dim, name='points_embedd')(self.inputs[6])
#         full_card = Concatenate(axis=-1)([profit_embedded, price_concatenated, points_embedded])
#         full_card  = Dense(units=dense1_dim)(full_card)
#         full_card = Dense(units=dense2_dim)(full_card)
#         self.layer = Model(inputs = self.inputs, outputs = full_card, name = 'card_encoder')


#male embedding 32
#densy 128
